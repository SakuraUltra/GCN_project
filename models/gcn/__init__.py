"""
Graph Convolutional Network Modules
图卷积网络模块

包含:
- GCNConv: 单层 GCN (Kipf & Welling, ICLR 2017)
- SimpleGCN: 多层 GCN 包装器
- GraphPooling: 图到嵌入的池化策略 (mean/max/attention)
- KNNEdgeBuilder: 动态 kNN 图边构建器
- HybridEdgeBuilder: 固定网格 + 动态 kNN 混合边构建器
"""

from .gcn_conv import GCNConv, SimpleGCN
from .graph_pooling import (
    GraphPooling,
    MeanPooling,
    MaxPooling,
    AttentionPooling
)
from .knn_edge_builder import KNNEdgeBuilder, HybridEdgeBuilder

__all__ = [
    'GCNConv',
    'SimpleGCN',
    'GraphPooling',
    'MeanPooling',
    'MaxPooling',
    'AttentionPooling',
    'KNNEdgeBuilder',
    'HybridEdgeBuilder'
]
