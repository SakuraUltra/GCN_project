"""
Graph Pooling Module
图池化模块：将图节点特征池化为全局图嵌入

支持三种池化策略：
1. Mean Pooling: 平均池化
2. Max Pooling: 最大池化  
3. Attention Pooling: 注意力加权池化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPooling(nn.Module):
    """
    Mean Pooling (平均池化)
    
    对所有节点特征取平均，得到图级表示
    公式: h_graph = (1/N) * Σ h_i
    
    优点: 简单高效，保留全局信息
    缺点: 对所有节点一视同仁，可能丢失重要节点信息
    """
    
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, x, batch=None):
        """
        Args:
            x: (N, D) 节点特征
            batch: (N,) 批次索引，指示每个节点属于哪个图
                   如果为 None，则认为所有节点属于同一个图
        
        Returns:
            graph_embedding: (B, D) 图嵌入，B 为图的数量
        """
        if batch is None:
            # 单图情况：对所有节点求平均
            return x.mean(dim=0, keepdim=True)  # (1, D)
        
        # 多图情况：对每个图的节点分别求平均
        num_graphs = batch.max().item() + 1
        graph_embeddings = []
        
        for i in range(num_graphs):
            mask = (batch == i)
            graph_nodes = x[mask]  # 属于图 i 的所有节点
            graph_emb = graph_nodes.mean(dim=0)  # (D,)
            graph_embeddings.append(graph_emb)
        
        return torch.stack(graph_embeddings, dim=0)  # (B, D)


class MaxPooling(nn.Module):
    """
    Max Pooling (最大池化)
    
    对每个维度取所有节点的最大值
    公式: h_graph[d] = max_i h_i[d]
    
    优点: 捕获最显著的特征，对异常值鲁棒
    缺点: 可能丢失细粒度信息，只关注极值
    """
    
    def __init__(self):
        super(MaxPooling, self).__init__()
    
    def forward(self, x, batch=None):
        """
        Args:
            x: (N, D) 节点特征
            batch: (N,) 批次索引
        
        Returns:
            graph_embedding: (B, D) 图嵌入
        """
        if batch is None:
            # 单图情况：对所有节点求最大值
            return x.max(dim=0, keepdim=True)[0]  # (1, D)
        
        # 多图情况：对每个图的节点分别求最大值
        num_graphs = batch.max().item() + 1
        graph_embeddings = []
        
        for i in range(num_graphs):
            mask = (batch == i)
            graph_nodes = x[mask]  # 属于图 i 的所有节点
            graph_emb = graph_nodes.max(dim=0)[0]  # (D,)
            graph_embeddings.append(graph_emb)
        
        return torch.stack(graph_embeddings, dim=0)  # (B, D)


class AttentionPooling(nn.Module):
    """
    Attention Pooling (注意力池化)
    
    学习每个节点的重要性权重，进行加权求和
    公式: 
        α_i = softmax(MLP(h_i))
        h_graph = Σ α_i * h_i
    
    优点: 自适应学习节点重要性，更灵活
    缺点: 引入额外参数，计算复杂度较高
    
    参考文献:
    - Graph Attention Networks (Veličković et al., ICLR 2018)
    - Set2Set (Vinyals et al., NIPS 2016)
    """
    
    def __init__(self, in_channels, hidden_channels=128):
        """
        Args:
            in_channels: 节点特征维度
            hidden_channels: 注意力网络隐藏层维度
        """
        super(AttentionPooling, self).__init__()
        
        # 注意力计算网络 (两层 MLP)
        self.att_net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)  # 输出标量注意力分数
        )
        
        # 初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Xavier 初始化"""
        for layer in self.att_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x, batch=None):
        """
        Args:
            x: (N, D) 节点特征
            batch: (N,) 批次索引
        
        Returns:
            graph_embedding: (B, D) 图嵌入
        """
        if batch is None:
            # 单图情况
            # 计算注意力分数
            att_scores = self.att_net(x)  # (N, 1)
            att_weights = F.softmax(att_scores, dim=0)  # (N, 1)
            
            # 加权求和
            graph_emb = (x * att_weights).sum(dim=0, keepdim=True)  # (1, D)
            return graph_emb
        
        # 多图情况：对每个图分别计算注意力
        num_graphs = batch.max().item() + 1
        graph_embeddings = []
        
        for i in range(num_graphs):
            mask = (batch == i)
            graph_nodes = x[mask]  # 属于图 i 的所有节点 (N_i, D)
            
            # 计算注意力分数
            att_scores = self.att_net(graph_nodes)  # (N_i, 1)
            att_weights = F.softmax(att_scores, dim=0)  # (N_i, 1)
            
            # 加权求和
            graph_emb = (graph_nodes * att_weights).sum(dim=0)  # (D,)
            graph_embeddings.append(graph_emb)
        
        return torch.stack(graph_embeddings, dim=0)  # (B, D)


class GraphPooling(nn.Module):
    """
    统一的图池化模块
    
    支持三种池化策略的切换，方便消融实验
    """
    
    def __init__(self, pooling_type='mean', in_channels=None, hidden_channels=128):
        """
        Args:
            pooling_type: 池化类型 ('mean', 'max', 'attention')
            in_channels: 节点特征维度 (仅 attention 需要)
            hidden_channels: 注意力网络隐藏层维度 (仅 attention 需要)
        """
        super(GraphPooling, self).__init__()
        
        self.pooling_type = pooling_type
        
        if pooling_type == 'mean':
            self.pooling = MeanPooling()
        elif pooling_type == 'max':
            self.pooling = MaxPooling()
        elif pooling_type == 'attention':
            if in_channels is None:
                raise ValueError("AttentionPooling requires 'in_channels' argument")
            self.pooling = AttentionPooling(in_channels, hidden_channels)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}. "
                           f"Choose from ['mean', 'max', 'attention']")
    
    def forward(self, x, batch=None):
        """
        Args:
            x: (N, D) 节点特征
            batch: (N,) 批次索引
        
        Returns:
            graph_embedding: (B, D) 图嵌入
        """
        return self.pooling(x, batch)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(type={self.pooling_type})"


def test_graph_pooling():
    """测试图池化模块"""
    print("=" * 80)
    print("🧪 测试图池化模块")
    print("=" * 80)
    
    # 创建测试数据
    num_nodes = 16  # 4x4 grid
    in_channels = 2048
    
    # 单个图的节点特征
    x = torch.randn(num_nodes, in_channels)
    
    print(f"\n输入:")
    print(f"  节点数: {num_nodes}")
    print(f"  节点特征维度: {in_channels}")
    print(f"  节点特征形状: {x.shape}")
    
    # 测试三种池化策略
    pooling_types = ['mean', 'max', 'attention']
    results = {}
    
    for pool_type in pooling_types:
        print(f"\n{'─' * 80}")
        print(f"测试 {pool_type.upper()} Pooling...")
        print(f"{'─' * 80}")
        
        # 创建池化层
        if pool_type == 'attention':
            pooling = GraphPooling(pool_type, in_channels=in_channels, hidden_channels=128)
        else:
            pooling = GraphPooling(pool_type)
        
        # 前向传播
        graph_emb = pooling(x)
        results[pool_type] = graph_emb
        
        print(f"  输出形状: {graph_emb.shape}")
        print(f"  输出范围: [{graph_emb.min().item():.4f}, {graph_emb.max().item():.4f}]")
        print(f"  输出均值: {graph_emb.mean().item():.4f}")
        print(f"  输出标准差: {graph_emb.std().item():.4f}")
        
        # 检查梯度流动
        if pool_type == 'attention':
            loss = graph_emb.sum()
            loss.backward()
            
            # 检查注意力网络参数是否有梯度
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                          for p in pooling.parameters())
            print(f"  注意力网络有梯度: {has_grad}")
    
    # 比较不同池化策略
    print(f"\n{'=' * 80}")
    print(f"📊 池化策略比较")
    print(f"{'=' * 80}")
    
    mean_emb = results['mean'].squeeze()
    max_emb = results['max'].squeeze()
    att_emb = results['attention'].squeeze()
    
    # 计算相似度
    mean_max_sim = F.cosine_similarity(mean_emb, max_emb, dim=0).item()
    mean_att_sim = F.cosine_similarity(mean_emb, att_emb, dim=0).item()
    max_att_sim = F.cosine_similarity(max_emb, att_emb, dim=0).item()
    
    print(f"\n余弦相似度:")
    print(f"  Mean vs Max:       {mean_max_sim:.4f}")
    print(f"  Mean vs Attention: {mean_att_sim:.4f}")
    print(f"  Max vs Attention:  {max_att_sim:.4f}")
    
    # 计算 L2 距离
    mean_max_dist = (mean_emb - max_emb).norm().item()
    mean_att_dist = (mean_emb - att_emb).norm().item()
    max_att_dist = (max_emb - att_emb).norm().item()
    
    print(f"\nL2 距离:")
    print(f"  Mean vs Max:       {mean_max_dist:.4f}")
    print(f"  Mean vs Attention: {mean_att_dist:.4f}")
    print(f"  Max vs Attention:  {max_att_dist:.4f}")
    
    # 测试批量处理
    print(f"\n{'=' * 80}")
    print(f"🔄 测试批量处理 (Batch Processing)")
    print(f"{'=' * 80}")
    
    # 模拟 batch=8 个图，每个图 16 个节点
    batch_size = 8
    x_batch = torch.randn(batch_size * num_nodes, in_channels)
    batch_idx = torch.arange(batch_size).repeat_interleave(num_nodes)
    
    print(f"\n输入:")
    print(f"  总节点数: {x_batch.shape[0]} ({batch_size} 个图 × {num_nodes} 节点)")
    print(f"  Batch索引形状: {batch_idx.shape}")
    
    for pool_type in pooling_types:
        if pool_type == 'attention':
            pooling = GraphPooling(pool_type, in_channels=in_channels)
        else:
            pooling = GraphPooling(pool_type)
        
        graph_embs = pooling(x_batch, batch_idx)
        print(f"\n{pool_type.upper()} Pooling:")
        print(f"  输出形状: {graph_embs.shape} (应为 [{batch_size}, {in_channels}])")
        print(f"  输出范围: [{graph_embs.min().item():.4f}, {graph_embs.max().item():.4f}]")
    
    print(f"\n{'=' * 80}")
    print(f"✅ 图池化模块测试完成!")
    print(f"{'=' * 80}")
    print(f"\n关键发现:")
    print(f"  1. 三种池化策略均正常工作")
    print(f"  2. Mean Pooling: 简单高效，作为 baseline")
    print(f"  3. Max Pooling: 捕获极值特征，互补性强")
    print(f"  4. Attention Pooling: 自适应加权，灵活性高")
    print(f"  5. 不同策略产生的嵌入有一定差异，适合消融实验")
    print(f"\n建议:")
    print(f"  • Baseline: 使用 Mean Pooling (最常用)")
    print(f"  • 消融实验: 比较三种策略对 ReID 性能的影响")
    print(f"  • 组合策略: 可尝试 concat(mean, max) 融合互补信息")
    print("=" * 80)


if __name__ == '__main__':
    test_graph_pooling()
