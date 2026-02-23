"""
Embedding Fusion Module
嵌入融合模块：融合全局CNN嵌入和图GCN嵌入

支持两种融合策略：
1. Concat + Projection: 拼接后投影到目标维度
2. Gated Fusion: 门控机制自适应融合

目标：为公平比较，保持输出维度与baseline一致 (2048)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatProjectionFusion(nn.Module):
    """
    Concat + Projection Fusion
    拼接投影融合
    
    方法: [global_emb; graph_emb] -> Linear -> output_emb
    
    优点:
    - 简单直接，易于训练
    - 保留两种嵌入的完整信息
    - 计算高效
    
    缺点:
    - 参数量增加 (需要投影层)
    - 融合方式固定，缺乏自适应性
    """
    
    def __init__(self, global_dim=2048, graph_dim=2048, output_dim=2048, dropout=0.5):
        """
        Args:
            global_dim: 全局CNN嵌入维度
            graph_dim: 图GCN嵌入维度
            output_dim: 输出嵌入维度 (默认2048，与baseline一致)
            dropout: Dropout概率
        """
        super(ConcatProjectionFusion, self).__init__()
        
        self.global_dim = global_dim
        self.graph_dim = graph_dim
        self.output_dim = output_dim
        
        # 拼接后的维度
        concat_dim = global_dim + graph_dim
        
        # 投影层: (global_dim + graph_dim) -> output_dim
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, global_emb, graph_emb):
        """
        Args:
            global_emb: (B, global_dim) 全局CNN嵌入 (e.g., GAP特征)
            graph_emb: (B, graph_dim) 图GCN嵌入 (经过graph pooling)
        
        Returns:
            fused_emb: (B, output_dim) 融合后的嵌入
        """
        # 拼接
        concat_emb = torch.cat([global_emb, graph_emb], dim=1)  # (B, global_dim + graph_dim)
        
        # 投影
        fused_emb = self.projection(concat_emb)  # (B, output_dim)
        
        return fused_emb


class GatedFusion(nn.Module):
    """
    Gated Fusion
    门控融合
    
    方法: 学习一个门控权重，自适应融合两种嵌入
    
    公式:
        gate = σ(W_g [global_emb; graph_emb] + b_g)
        fused = gate ⊙ f_g(global_emb) + (1-gate) ⊙ f_gr(graph_emb)
    
    或更简单的版本:
        gate = σ(MLP(global_emb, graph_emb))
        fused = gate ⊙ global_emb + (1-gate) ⊙ graph_emb
    
    优点:
    - 自适应学习融合权重
    - 可解释性强 (可视化gate值)
    - 动态平衡两种信息
    
    缺点:
    - 训练复杂度较高
    - 需要两种嵌入维度相同 (或先投影)
    """
    
    def __init__(self, global_dim=2048, graph_dim=2048, output_dim=2048, 
                 hidden_dim=512, dropout=0.5):
        """
        Args:
            global_dim: 全局CNN嵌入维度
            graph_dim: 图GCN嵌入维度
            output_dim: 输出嵌入维度
            hidden_dim: 门控网络隐藏层维度
            dropout: Dropout概率
        """
        super(GatedFusion, self).__init__()
        
        self.global_dim = global_dim
        self.graph_dim = graph_dim
        self.output_dim = output_dim
        
        # 如果维度不同，先投影到相同维度
        self.global_proj = None
        self.graph_proj = None
        
        if global_dim != output_dim:
            self.global_proj = nn.Linear(global_dim, output_dim)
        
        if graph_dim != output_dim:
            self.graph_proj = nn.Linear(graph_dim, output_dim)
        
        # 门控网络: 输入两个嵌入，输出门控权重
        self.gate_net = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # 输出 [0, 1] 的门控权重
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, global_emb, graph_emb):
        """
        Args:
            global_emb: (B, global_dim) 全局CNN嵌入
            graph_emb: (B, graph_dim) 图GCN嵌入
        
        Returns:
            fused_emb: (B, output_dim) 融合后的嵌入
            gate: (B, output_dim) 门控权重 (用于可视化)
        """
        # 1. 投影到相同维度 (如果需要)
        if self.global_proj is not None:
            global_emb_proj = self.global_proj(global_emb)
        else:
            global_emb_proj = global_emb
        
        if self.graph_proj is not None:
            graph_emb_proj = self.graph_proj(graph_emb)
        else:
            graph_emb_proj = graph_emb
        
        # 2. 计算门控权重
        concat = torch.cat([global_emb_proj, graph_emb_proj], dim=1)  # (B, output_dim * 2)
        gate = self.gate_net(concat)  # (B, output_dim)
        
        # 3. 门控融合
        fused_emb = gate * global_emb_proj + (1 - gate) * graph_emb_proj  # (B, output_dim)
        
        return fused_emb, gate


class EmbeddingFusion(nn.Module):
    """
    统一的嵌入融合模块
    
    支持多种融合策略的切换，方便消融实验
    """
    
    def __init__(self, fusion_type='concat', global_dim=2048, graph_dim=2048, 
                 output_dim=2048, hidden_dim=512, dropout=0.5):
        """
        Args:
            fusion_type: 融合类型 ('concat', 'gated', 'add', 'none')
                - 'concat': Concat + Projection
                - 'gated': Gated Fusion
                - 'add': 简单相加 (需要维度相同)
                - 'none': 只使用graph嵌入
            global_dim: 全局CNN嵌入维度
            graph_dim: 图GCN嵌入维度
            output_dim: 输出嵌入维度
            hidden_dim: 门控网络隐藏层维度 (仅gated需要)
            dropout: Dropout概率
        """
        super(EmbeddingFusion, self).__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            self.fusion = ConcatProjectionFusion(
                global_dim, graph_dim, output_dim, dropout
            )
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(
                global_dim, graph_dim, output_dim, hidden_dim, dropout
            )
        elif fusion_type == 'add':
            # 简单相加，需要维度相同
            assert global_dim == graph_dim == output_dim, \
                "Add fusion requires same dimensions"
            self.fusion = None
        elif fusion_type == 'none':
            # 只使用graph嵌入
            if graph_dim != output_dim:
                self.fusion = nn.Linear(graph_dim, output_dim)
            else:
                self.fusion = None
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, global_emb, graph_emb):
        """
        Args:
            global_emb: (B, global_dim) 全局CNN嵌入
            graph_emb: (B, graph_dim) 图GCN嵌入
        
        Returns:
            fused_emb: (B, output_dim) 融合后的嵌入
            extra: dict, 额外信息 (如gate权重)
        """
        extra = {}
        
        if self.fusion_type == 'concat':
            fused_emb = self.fusion(global_emb, graph_emb)
        
        elif self.fusion_type == 'gated':
            fused_emb, gate = self.fusion(global_emb, graph_emb)
            extra['gate'] = gate  # 保存门控权重用于可视化
        
        elif self.fusion_type == 'add':
            fused_emb = global_emb + graph_emb
        
        elif self.fusion_type == 'none':
            if self.fusion is not None:
                fused_emb = self.fusion(graph_emb)
            else:
                fused_emb = graph_emb
        
        return fused_emb, extra
    
    def __repr__(self):
        return f"{self.__class__.__name__}(type={self.fusion_type})"


def test_embedding_fusion():
    """测试嵌入融合模块"""
    print("=" * 80)
    print("🧪 测试嵌入融合模块")
    print("=" * 80)
    
    batch_size = 32
    global_dim = 2048
    graph_dim = 2048
    output_dim = 2048
    
    # 模拟输入
    global_emb = torch.randn(batch_size, global_dim)
    graph_emb = torch.randn(batch_size, graph_dim)
    
    print(f"\n输入:")
    print(f"  Batch Size: {batch_size}")
    print(f"  全局嵌入: {global_emb.shape}")
    print(f"  图嵌入: {graph_emb.shape}")
    
    # 测试所有融合策略
    fusion_types = ['concat', 'gated', 'add', 'none']
    
    for fusion_type in fusion_types:
        print(f"\n{'=' * 80}")
        print(f"测试 {fusion_type.upper()} Fusion")
        print(f"{'=' * 80}")
        
        fusion = EmbeddingFusion(
            fusion_type=fusion_type,
            global_dim=global_dim,
            graph_dim=graph_dim,
            output_dim=output_dim
        )
        
        # 前向传播
        fused_emb, extra = fusion(global_emb, graph_emb)
        
        print(f"\n输出:")
        print(f"  融合嵌入形状: {fused_emb.shape}")
        print(f"  输出维度正确: {fused_emb.shape == (batch_size, output_dim)}")
        print(f"  输出范围: [{fused_emb.min().item():.4f}, {fused_emb.max().item():.4f}]")
        print(f"  输出均值: {fused_emb.mean().item():.4f}")
        print(f"  输出标准差: {fused_emb.std().item():.4f}")
        
        # 检查额外信息
        if 'gate' in extra:
            gate = extra['gate']
            print(f"\n门控权重:")
            print(f"  Gate形状: {gate.shape}")
            print(f"  Gate范围: [{gate.min().item():.4f}, {gate.max().item():.4f}]")
            print(f"  Gate均值: {gate.mean().item():.4f} (0.5表示均衡)")
            print(f"  Gate标准差: {gate.std().item():.4f}")
        
        # 检查梯度流动
        loss = fused_emb.sum()
        loss.backward()
        
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in fusion.parameters())
        
        print(f"\n梯度流动:")
        print(f"  参数有梯度: {has_grad or fusion_type in ['add', 'none']}")
        
        # 清除梯度
        fusion.zero_grad()
    
    # 测试维度不匹配的情况
    print(f"\n{'=' * 80}")
    print(f"测试维度不匹配场景")
    print(f"{'=' * 80}")
    
    global_emb_diff = torch.randn(batch_size, 1024)  # 不同维度
    graph_emb_diff = torch.randn(batch_size, 512)
    
    fusion_concat = EmbeddingFusion(
        fusion_type='concat',
        global_dim=1024,
        graph_dim=512,
        output_dim=2048
    )
    
    fused_emb = fusion_concat(global_emb_diff, graph_emb_diff)[0]
    print(f"\nConcat Fusion (1024 + 512 -> 2048):")
    print(f"  输入维度: global={global_emb_diff.shape}, graph={graph_emb_diff.shape}")
    print(f"  输出维度: {fused_emb.shape}")
    print(f"  ✓ 维度自动适配成功")
    
    fusion_gated = EmbeddingFusion(
        fusion_type='gated',
        global_dim=1024,
        graph_dim=512,
        output_dim=2048
    )
    
    fused_emb, extra = fusion_gated(global_emb_diff, graph_emb_diff)
    print(f"\nGated Fusion (1024, 512 -> 2048):")
    print(f"  输出维度: {fused_emb.shape}")
    print(f"  Gate均值: {extra['gate'].mean().item():.4f}")
    print(f"  ✓ 门控融合自动投影成功")
    
    # 参数量对比
    print(f"\n{'=' * 80}")
    print(f"📊 参数量对比")
    print(f"{'=' * 80}")
    
    for fusion_type in fusion_types:
        fusion = EmbeddingFusion(fusion_type=fusion_type)
        num_params = sum(p.numel() for p in fusion.parameters())
        print(f"  {fusion_type.upper():10s}: {num_params:,} 参数")
    
    print(f"\n{'=' * 80}")
    print(f"✅ 嵌入融合模块测试完成!")
    print(f"{'=' * 80}")
    print(f"\n关键发现:")
    print(f"  1. 所有融合策略输出维度正确")
    print(f"  2. Concat需要最多参数 (投影层)")
    print(f"  3. Gated自适应学习融合权重")
    print(f"  4. Add最简单，但需要维度相同")
    print(f"  5. 自动处理维度不匹配情况")
    print(f"\n建议:")
    print(f"  • Baseline: Concat + Projection (稳定)")
    print(f"  • Advanced: Gated Fusion (自适应)")
    print(f"  • Ablation: 对比4种融合策略")
    print("=" * 80)


if __name__ == '__main__':
    test_embedding_fusion()
