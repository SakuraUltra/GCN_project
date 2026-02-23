"""
Embedding Fusion Modules
嵌入融合模块

支持融合策略:
- ConcatProjectionFusion: 拼接后投影
- GatedFusion: 门控自适应融合
- EmbeddingFusion: 统一接口 (支持 concat/gated/add/none)
"""

from .embedding_fusion import (
    ConcatProjectionFusion,
    GatedFusion,
    EmbeddingFusion
)

__all__ = [
    'ConcatProjectionFusion',
    'GatedFusion',
    'EmbeddingFusion'
]
