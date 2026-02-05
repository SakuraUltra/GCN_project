"""
Identity Loss with Label Smoothing
用于车辆重识别的ID分类损失
"""

import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing Regularizer
    
    标签平滑正则化可以提高模型泛化能力，防止过拟合
    
    Args:
        num_classes (int): 类别数量
        epsilon (float): 平滑参数，默认0.1
        
    Reference:
        "Rethinking the Inception Architecture for Computer Vision" - CVPR 2016
    """
    
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): 模型输出的logits, shape (N, C)
            targets (torch.Tensor): 目标标签, shape (N,)
            
        Returns:
            torch.Tensor: 损失值
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        targets = targets.to(log_probs.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
