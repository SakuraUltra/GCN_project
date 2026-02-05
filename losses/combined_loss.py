"""
Combined Loss for BoT-Baseline
结合ID Loss和Triplet Loss的混合损失函数
"""

import torch
import torch.nn as nn
from .id_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss


class BoTLoss(nn.Module):
    """
    BoT (Bag of Tricks) Combined Loss
    
    结合ID分类损失和Triplet度量学习损失
    这是车辆重识别SOTA方法的标准损失函数
    
    Args:
        num_classes (int): 身份类别数量
        epsilon (float): 标签平滑参数
        margin (float): Triplet Loss边界值
        
    Reference:
        "Bag of Tricks and A Strong Baseline for Deep Person Re-identification" - CVPRW 2019
    """
    
    def __init__(self, num_classes, epsilon=0.1, margin=0.3):
        super(BoTLoss, self).__init__()
        self.id_loss = CrossEntropyLabelSmooth(num_classes, epsilon)
        self.triplet_loss = TripletLoss(margin)
        
    def forward(self, score, feat, target):
        """
        Args:
            score (torch.Tensor): 分类器输出的logits, shape (N, num_classes)
            feat (torch.Tensor): 全局特征向量, shape (N, D)
            target (torch.Tensor): 身份标签, shape (N,)
            
        Returns:
            tuple: (总损失, ID损失, Triplet损失)
        """
        id_loss = self.id_loss(score, target)
        triplet_loss = self.triplet_loss(feat, target)
        
        # 默认1:1权重
        total_loss = id_loss + triplet_loss
        
        return total_loss, id_loss, triplet_loss
