"""
Dynamic kNN Edge Builder for GCN
动态 kNN 图边构建器

用途：
- 基于节点特征相似度动态构建图边
- 支持余弦相似度和欧氏距离
- 可选梯度停止（early stopping）
- 用于消融实验对比固定网格邻接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KNNEdgeBuilder(nn.Module):
    """
    动态 kNN 图边构建器
    
    在每个 forward 中根据当前节点特征计算 top-k 最相似邻居
    """
    
    def __init__(self, k=4, metric='cosine', detach_features=False):
        """
        Args:
            k: 每个节点连接的最近邻数量
            metric: 相似度度量 ('cosine' 或 'euclidean')
            detach_features: 是否停止特征梯度（避免训练不稳定）
        """
        super(KNNEdgeBuilder, self).__init__()
        self.k = k
        self.metric = metric
        self.detach_features = detach_features
        
    def forward(self, node_features):
        """
        根据节点特征动态构建 kNN 图
        
        Args:
            node_features: (B, N, C) 批次节点特征
        
        Returns:
            edge_index_batch: List[(2, E)] 每个样本的边索引
            edge_weight_batch: List[(E,)] 每个样本的边权重（可选）
        """
        B, N, C = node_features.shape
        
        edge_index_batch = []
        edge_weight_batch = []
        
        for b in range(B):
            features = node_features[b]  # (N, C)
            
            # 可选：停止梯度
            if self.detach_features:
                features = features.detach()
            
            # 计算相似度矩阵
            if self.metric == 'cosine':
                # 余弦相似度
                features_norm = F.normalize(features, p=2, dim=1)
                similarity = torch.mm(features_norm, features_norm.t())  # (N, N)
            elif self.metric == 'euclidean':
                # 欧氏距离（负值，因为要找最小距离）
                dist = torch.cdist(features.unsqueeze(0), features.unsqueeze(0), p=2).squeeze(0)  # (N, N)
                similarity = -dist
            else:
                raise ValueError(f"Invalid metric: {self.metric}. Must be 'cosine' or 'euclidean'.")
            
            # 对角线设为负无穷（避免选择自己）
            similarity.fill_diagonal_(float('-inf'))
            
            # 为每个节点找 top-k 邻居
            k_neighbors = min(self.k, N - 1)  # 避免 k > N-1
            topk_values, topk_indices = similarity.topk(k_neighbors, dim=1)  # (N, k)
            
            # 构建边索引
            source_nodes = torch.arange(N, device=features.device).unsqueeze(1).expand(-1, k_neighbors).reshape(-1)
            target_nodes = topk_indices.reshape(-1)
            
            edge_index = torch.stack([source_nodes, target_nodes], dim=0)  # (2, N*k)
            edge_weight = topk_values.reshape(-1)  # (N*k)
            
            # 如果使用欧氏距离，转换为正值
            if self.metric == 'euclidean':
                edge_weight = -edge_weight
            
            edge_index_batch.append(edge_index)
            edge_weight_batch.append(edge_weight)
        
        return edge_index_batch, edge_weight_batch
    
    def extra_repr(self):
        return f"k={self.k}, metric={self.metric}, detach={self.detach_features}"


class HybridEdgeBuilder(nn.Module):
    """
    混合边构建器: 固定网格 + 动态 kNN
    
    用于消融实验：测试固定+动态边的组合效果
    """
    
    def __init__(self, grid_h, grid_w, adjacency_type='4', knn_k=4, knn_metric='cosine'):
        """
        Args:
            grid_h, grid_w: 网格尺寸
            adjacency_type: 固定邻接类型 ('4' or '8')
            knn_k: kNN 邻居数量
            knn_metric: kNN 相似度度量
        """
        super(HybridEdgeBuilder, self).__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.adjacency_type = adjacency_type
        
        # 固定网格边（预计算）
        self.register_buffer('grid_edge_index', self._build_grid_adjacency(grid_h, grid_w, adjacency_type))
        
        # 动态 kNN 边构建器
        self.knn_builder = KNNEdgeBuilder(k=knn_k, metric=knn_metric, detach_features=True)
        
    def _build_grid_adjacency(self, grid_h, grid_w, adjacency_type):
        """构建固定网格邻接（与 BoTGCN 相同逻辑）"""
        num_nodes = grid_h * grid_w
        edges = []
        
        if adjacency_type == '4':
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif adjacency_type == '8':
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            raise ValueError(f"Invalid adjacency_type: {adjacency_type}")
        
        for i in range(grid_h):
            for j in range(grid_w):
                node_id = i * grid_w + j
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_h and 0 <= nj < grid_w:
                        neighbor_id = ni * grid_w + nj
                        edges.append([node_id, neighbor_id])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def forward(self, node_features):
        """
        组合固定网格边和动态 kNN 边
        
        Args:
            node_features: (B, N, C)
        
        Returns:
            edge_index_batch: List[(2, E_total)]
        """
        B = node_features.size(0)
        
        # 动态 kNN 边
        knn_edge_batch, _ = self.knn_builder(node_features)
        
        # 组合固定边和动态边
        edge_index_batch = []
        for b in range(B):
            # 固定网格边
            grid_edges = self.grid_edge_index.to(node_features.device)
            
            # 动态 kNN 边
            knn_edges = knn_edge_batch[b]
            
            # 合并去重
            combined_edges = torch.cat([grid_edges, knn_edges], dim=1)  # (2, E_grid + E_knn)
            combined_edges = torch.unique(combined_edges, dim=1)  # 去重
            
            edge_index_batch.append(combined_edges)
        
        return edge_index_batch
    
    def extra_repr(self):
        return f"grid={self.grid_h}x{self.grid_w}, adjacency={self.adjacency_type}, knn_k={self.knn_builder.k}"
