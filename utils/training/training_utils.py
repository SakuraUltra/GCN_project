"""
Training utilities for Vehicle Re-Identification
"""

import torch
import torch.nn as nn
import numpy as np
import os
import logging
import random


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger for training"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    """Save model checkpoint"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(save_dir, 'model_best.pth')
        torch.save(state, best_filepath)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load optimizer state if provided
    start_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return start_epoch, best_acc


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with warmup"""
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0/3,
                 warmup_iters=500, warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}".format(milestones)
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def bisect_right(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted"""
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def extract_features(model, data_loader, device):
    """Extract features from model"""
    model.eval()
    features = []
    pids = []
    camids = []
    
    with torch.no_grad():
        for batch_idx, (imgs, pid, camid) in enumerate(data_loader):
            imgs = imgs.to(device)
            
            # Forward pass
            feat = model(imgs)
            if isinstance(feat, tuple):
                feat = feat[0]  # Take the feature vector
            
            features.append(feat.cpu())
            pids.extend(pid)
            camids.extend(camid)
    
    features = torch.cat(features, dim=0)
    pids = np.array(pids)
    camids = np.array(camids)
    
    return features.numpy(), pids, camids


def create_optimizer(model, optimizer_name='adam', lr=0.0003, weight_decay=5e-4):
    """Create optimizer"""
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params, momentum=0.9)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer