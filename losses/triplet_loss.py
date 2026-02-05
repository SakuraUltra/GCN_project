"""
Triplet Loss with Hard Mining
用于学习判别性特征表示的度量学习损失
"""

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """
    Triplet Loss with Hard Positive/Negative Mining
    
    通过最小化锚点与正样本的距离，同时最大化锚点与负样本的距离来学习特征
    使用Hard Mining策略选择最难的正负样本对
    
    Args:
        margin (float): 边界值，默认0.3
        
    Reference:
        "In Defense of the Triplet Loss for Person Re-Identification" - arXiv 2017
    """
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): 特征向量, shape (N, D)
            targets (torch.Tensor): 身份标签, shape (N,)
            
        Returns:
            torch.Tensor: 损失值
        """
        n = inputs.size(0)
        
        # AMP fix: conversions to float32 for stability and type matching
        inputs = inputs.float()
        
        # 计算成对距离矩阵
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # 为每个锚点找到最难的正样本和负样本
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        
        for i in range(n):
            # 最难正样本：同一身份中距离最远的样本
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            # 最难负样本：不同身份中距离最近的样本
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # 计算ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss
