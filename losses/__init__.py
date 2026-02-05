"""
Loss Functions for Vehicle Re-identification
所有损失函数的统一接口
"""

from .id_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .combined_loss import BoTLoss

__all__ = [
    'CrossEntropyLabelSmooth',
    'TripletLoss',
    'BoTLoss',
]
