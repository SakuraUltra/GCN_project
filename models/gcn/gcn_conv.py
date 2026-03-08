"""
GCN (Graph Convolutional Network) 模块

实现 Kipf & Welling 的标准 GCN 层
论文: Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    """
    单层 GCN 卷积
    
    公式: H' = σ(D^(-1/2) A D^(-1/2) H W)
    
    其中:
    - A: 邻接矩阵（带自环）
    - D: 度矩阵
    - H: 输入节点特征 (N, C_in)
    - W: 可学习权重 (C_in, C_out)
    - σ: 激活函数
    """
    
    def __init__(self, in_channels, out_channels, bias=True):
        """
        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            bias: 是否使用偏置
        """
        super(GCNConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 可学习的权重矩阵
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_weight=None):
        assert x.dim() == 2 and edge_index.dim() == 2 and edge_index.size(0) == 2
        
        N = x.size(0)
        
        # 1. 添加自环和归一化
        edge_index, edge_weight = self.add_self_loops(edge_index, edge_weight, num_nodes=N)
        edge_index, edge_weight = self.normalize_adj(edge_index, edge_weight, num_nodes=N)
        
        # 2. 图传播（聚合邻居特征）
        out = self.propagate(x, edge_index, edge_weight)
        
        # 3. 特征变换
        out = torch.matmul(out, self.weight)
        
        # 4. 添加偏置
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def add_self_loops(self, edge_index, edge_weight, num_nodes):
        loop_index = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        
        if edge_weight is not None:
            loop_weight = torch.ones(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
            edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        else:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        return edge_index, edge_weight
    
    def normalize_adj(self, edge_index, edge_weight, num_nodes):
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
        deg.scatter_add_(0, row, edge_weight)
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        return edge_index, edge_weight
    
    def propagate(self, x, edge_index, edge_weight):
        """
        消息传递: 聚合邻居特征
        
        使用稀疏矩阵乘法优化
        公式: out = A_norm @ x
        
        Note: Sparse operations don't support AMP, so we force FP32
        """
        N = x.size(0)
        
        # Store original dtype and convert to FP32 for sparse ops
        original_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()
            edge_weight = edge_weight.float()
        
        row, col = edge_index[0], edge_index[1]
        
        # 构建稀疏邻接矩阵
        indices = edge_index
        values = edge_weight
        size = (N, N)
        
        # 使用 PyTorch 稀疏张量
        adj_sparse = torch.sparse_coo_tensor(indices, values, size)
        
        # 稀疏矩阵乘法: A @ x
        out = torch.sparse.mm(adj_sparse, x)
        
        # Convert back to original dtype
        if original_dtype == torch.float16:
            out = out.half()
        
        return out
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, dropout=0.5):
        super(SimpleGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 构建多层GCN
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            # 第一层
            self.convs.append(GCNConv(in_channels, hidden_channels))
            # 中间层
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            # 最后一层
            self.convs.append(GCNConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            # 最后一层不加激活和dropout
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


def test_gcn():
    """测试 GCN 层的正确性"""
    print("=" * 80)
    print("🧪 测试 GCN 层")
    print("=" * 80)
    
    # 创建简单图（5个节点）
    num_nodes = 5
    in_channels = 8
    out_channels = 4
    
    # 节点特征
    x = torch.randn(num_nodes, in_channels)
    
    # 边索引（环形图）
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 1, 2, 3, 4, 0],  # source
        [1, 2, 3, 4, 0, 0, 1, 2, 3, 4],  # target
    ], dtype=torch.long)
    
    print(f"\n输入:")
    print(f"  节点数: {num_nodes}")
    print(f"  输入特征: {x.shape}")
    print(f"  边数: {edge_index.size(1)}")
    
    # 测试单层 GCN
    print(f"\n测试单层 GCN (num_layers=1)...")
    gcn = SimpleGCN(in_channels, 16, out_channels, num_layers=1)
    out = gcn(x, edge_index)
    
    print(f"  输出特征: {out.shape}")
    print(f"  输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
    
    # 检查梯度流动
    print(f"\n检查梯度流动...")
    loss = out.sum()
    loss.backward()
    
    grad_norm = gcn.conv1.weight.grad.norm().item()
    has_nan = torch.isnan(out).any().item()
    
    print(f"  权重梯度范数: {grad_norm:.6f}")
    print(f"  包含 NaN: {has_nan}")
    
    if grad_norm > 0 and not has_nan:
        print(f"\n✅ 单层 GCN 测试通过!")
        print(f"  ✓ 前向传播正常")
        print(f"  ✓ 梯度流动正常")
        print(f"  ✓ 无 NaN 值")
    else:
        print(f"\n❌ GCN 测试失败!")
    
    print("=" * 80)


if __name__ == '__main__':
    test_gcn()
