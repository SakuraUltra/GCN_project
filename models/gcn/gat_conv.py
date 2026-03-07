"""
GAT (Graph Attention Network) 自实现版本
对应任务卡: VIT-26

设计原则：
  - 不依赖 PyTorch Geometric，与现有 GCNConv 接口完全一致
  - 实现 GATv2 (Brody et al. 2021) 修复了原版 GAT 的静态注意力问题
  - 支持 AMP (FP16/FP32 自动转换)，与现有训练框架兼容
  - 可直接替换 SimpleGCN 中的 GCNConv

参考论文：
  GATv2: Brody et al. "How Attentive are Graph Attention Networks?" ICLR 2022
  GAT:   Veličković et al. "Graph Attention Networks" ICLR 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Conv(nn.Module):
    """
    GATv2 单层卷积 - 自实现版本

    GATv2 公式:
        e_ij = a^T * LeakyReLU(W_l * h_i + W_r * h_j)   # 动态注意力
        α_ij = softmax_j(e_ij)
        h_i' = σ(Σ_j α_ij * W_r * h_j)

    与 GCNConv 接口完全兼容:
        输入: x (N, C_in), edge_index (2, E)
        输出: (N, C_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = False,       # False: 多头取均值，输出 out_channels
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.heads        = heads
        self.concat       = concat
        self.negative_slope = negative_slope
        self.dropout      = dropout

        # 每个头的维度
        # concat=False: 多头取均值，最终输出 out_channels，所以每个头输出 out_channels
        # concat=True:  多头拼接，最终输出 heads*head_dim，所以 head_dim = out_channels
        self.head_dim = out_channels if not concat else out_channels

        # W_l: 左节点（源节点）投影
        self.W_l = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        # W_r: 右节点（目标节点）投影（也用于最终特征）
        self.W_r = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        # 注意力向量 a: (heads, head_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, self.head_dim))

        if bias:
            out_dim = heads * self.head_dim if concat else out_channels
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l.weight, gain=1.414)
        nn.init.xavier_uniform_(self.W_r.weight, gain=1.414)
        nn.init.xavier_uniform_(self.att)

    def forward(
        self,
        x: torch.Tensor,           # (N, C_in)
        edge_index: torch.Tensor,  # (2, E)
        edge_weight=None,          # 兼容 GCNConv 接口，GAT 不使用
    ) -> torch.Tensor:
        """
        Args:
            x:          (N, C_in)  节点特征
            edge_index: (2, E)     边索引 COO 格式
        Returns:
            out:        (N, C_out) 输出节点特征
        """
        N = x.size(0)
        H = self.heads
        D = self.head_dim

        # Store original dtype for AMP compatibility
        original_dtype = x.dtype

        # ── Step1: 添加自环 ──────────────────────────────────────────
        edge_index = self._add_self_loops(edge_index, N)
        src, dst = edge_index[0], edge_index[1]   # (E,) 源/目标节点索引

        # ── Step2: 线性投影 ──────────────────────────────────────────
        x_l = self.W_l(x).view(N, H, D)   # (N, H, D) 左投影
        x_r = self.W_r(x).view(N, H, D)   # (N, H, D) 右投影

        # ── Step3: 计算注意力分数（GATv2 动态注意力）────────────────
        # e_ij = a^T * LeakyReLU(x_l[i] + x_r[j])
        x_l_src = x_l[src]   # (E, H, D) 每条边的源节点特征
        x_r_dst = x_r[dst]   # (E, H, D) 每条边的目标节点特征

        e = F.leaky_relu(x_l_src + x_r_dst, self.negative_slope)  # (E, H, D)
        e = (e * self.att).sum(dim=-1)                              # (E, H)

        # ── Step4: Softmax 归一化（按目标节点分组）─────────────────
        alpha = self._softmax_by_node(e, dst, N)   # (E, H)

        # Dropout
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout)

        # ── Step5: 加权聚合 ──────────────────────────────────────────
        # out[i] = Σ_{j∈N(i)} α_ij * x_r[j]
        x_r_src = x_r[src]                                     # (E, H, D)
        weighted = x_r_src * alpha.unsqueeze(-1)               # (E, H, D)

        out = torch.zeros(N, H, D, device=x.device, dtype=x.dtype)
        dst_expand = dst.view(-1, 1, 1).expand_as(weighted)
        out.scatter_add_(0, dst_expand, weighted)              # (N, H, D)

        # ── Step6: 多头合并 ──────────────────────────────────────────
        if self.concat:
            out = out.view(N, H * D)    # (N, H*D)
        else:
            out = out.mean(dim=1)       # (N, D) 均值

        if self.bias is not None:
            out = out + self.bias

        return out   # (N, C_out)

    # ──────────────────────────────────────────────────────────────────
    def _add_self_loops(self, edge_index: torch.Tensor, N: int) -> torch.Tensor:
        """添加自环 i→i，与 GCNConv 保持一致"""
        self_loops = torch.arange(N, device=edge_index.device).unsqueeze(0).repeat(2, 1)
        return torch.cat([edge_index, self_loops], dim=1)

    def _softmax_by_node(
        self,
        e: torch.Tensor,    # (E, H)
        dst: torch.Tensor,  # (E,)
        N: int,
    ) -> torch.Tensor:
        """按目标节点分组做 softmax，替代 PyG 的 softmax"""
        # 数值稳定：减去每个目标节点的最大值
        e_max = torch.zeros(N, e.size(1), device=e.device, dtype=e.dtype)
        e_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce='amax', include_self=True)
        e_shifted = e - e_max[dst]

        exp_e = e_shifted.exp()   # (E, H)

        # 计算分母（每个目标节点的 exp 之和）
        denom = torch.zeros(N, e.size(1), device=e.device, dtype=e.dtype)
        denom.scatter_add_(0, dst.unsqueeze(1).expand_as(exp_e), exp_e)
        denom = denom.clamp(min=1e-16)

        alpha = exp_e / denom[dst]   # (E, H)
        return alpha

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})'


# =====================================================================
# SimpleGAT: 多层 GAT，接口与 SimpleGCN 完全一致
# =====================================================================

class SimpleGAT(nn.Module):
    """
    多层 GAT 网络，与 SimpleGCN 接口完全一致
    可以直接替换 BoTGCN.__init__ 中的 self.gcn = SimpleGCN(...)

    使用示例（在 bot_gcn_model.py 中替换）:
        # 原来:
        self.gcn = SimpleGCN(in_channels=2048, ..., num_layers=1)
        # 替换为:
        self.gcn = SimpleGAT(in_channels=2048, ..., num_layers=1)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        if num_layers == 1:
            # 单层：直接输出 out_channels，多头取均值
            self.conv1 = GATv2Conv(
                in_channels, out_channels,
                heads=heads, concat=False, dropout=dropout
            )
            self.conv2 = None
        else:
            # 双层：第一层和第二层都用均值
            self.conv1 = GATv2Conv(
                in_channels, hidden_channels,
                heads=heads, concat=False, dropout=dropout
            )
            self.conv2 = GATv2Conv(
                hidden_channels, out_channels,
                heads=heads, concat=False, dropout=dropout
            )

    def forward(
        self,
        x: torch.Tensor,           # (N, C_in)
        edge_index: torch.Tensor,  # (2, E)
    ) -> torch.Tensor:
        """
        与 SimpleGCN.forward 接口完全一致
        输入: (N, C_in), (2, E)
        输出: (N, C_out)
        """
        if self.num_layers == 1:
            x = self.conv1(x, edge_index)
        else:
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)

        return x

    def __repr__(self):
        return f'{self.__class__.__name__}(layers={self.num_layers}, heads={self.conv1.heads})'
