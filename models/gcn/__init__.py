"""
Graph Convolutional Network Modules
图卷积网络模块

包含:
- GCNConv: 单层 GCN (Kipf & Welling, ICLR 2017)
- SimpleGCN: 多层 GCN 包装器
- GraphPooling: 图到嵌入的池化策略 (mean/max/attention)
"""

from .gcn_conv import GCNConv, SimpleGCN
from .graph_pooling import (
    GraphPooling,
    MeanPooling,
    MaxPooling,
    AttentionPooling
)

__all__ = [
    'GCNConv',
    'SimpleGCN',
    'GraphPooling',
    'MeanPooling',
    'MaxPooling',
    'AttentionPooling'
]
