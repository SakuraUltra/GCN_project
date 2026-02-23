"""
Learning Rate Schedulers
学习率调度策略
"""

import math
from torch.optim.lr_scheduler import LambdaLR


def create_warmup_cosine_scheduler(optimizer, warmup_epochs, max_epochs, eta_min_ratio=0.01):
    """
    创建Warmup + Cosine Annealing调度器
    
    这是SOTA方法的标准学习率策略：
    1. Warmup阶段：线性增长（避免训练初期梯度爆炸）
    2. Cosine Annealing：平滑衰减（保证收敛稳定）
    
    Args:
        optimizer: PyTorch优化器
        warmup_epochs (int): Warmup阶段的epoch数
        max_epochs (int): 总训练epoch数
        eta_min_ratio (float): 最小学习率相对于初始学习率的比例，默认0.01
    
    Returns:
        torch.optim.lr_scheduler.LambdaLR: 学习率调度器
        
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=0.00035)
        >>> scheduler = create_warmup_cosine_scheduler(optimizer, warmup_epochs=10, max_epochs=120)
        >>> for epoch in range(120):
        >>>     train_one_epoch(...)
        >>>     scheduler.step()
    
    Reference:
        "Bag of Tricks and A Strong Baseline for Deep Person Re-identification" - CVPRW 2019
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup phase: 线性增长 from 1% to 100%
            return max(0.01, epoch / warmup_epochs)
        else:
            # Cosine annealing phase
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            cosine_decay = (1 + math.cos(math.pi * progress)) / 2
            return eta_min_ratio + (1 - eta_min_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def create_multistep_scheduler(optimizer, milestones, gamma=0.1):
    """
    创建多步衰减调度器
    
    Args:
        optimizer: PyTorch优化器
        milestones (list): 衰减的epoch列表，如[40, 70]
        gamma (float): 衰减系数，默认0.1
    
    Returns:
        torch.optim.lr_scheduler.MultiStepLR
    """
    from torch.optim.lr_scheduler import MultiStepLR
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
