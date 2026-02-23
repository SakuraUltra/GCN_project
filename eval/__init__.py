"""
Evaluation Module
评估相关的所有组件
"""

from .evaluator import ReIDEvaluator, compute_mAP_cmc

__all__ = [
    'ReIDEvaluator',
    'compute_mAP_cmc',
]
