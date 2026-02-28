"""
Data Augmentation Strategies for Vehicle Re-identification
包含多种遮挡增强策略: Random Erasing, Cutout, GridMask等
"""

import torch
import random
import math
import numpy as np
from PIL import Image


class RandomErasing:
    """
    Random Erasing Data Augmentation
    随机擦除数据增强，模拟遮挡情况
    
    Reference:
        "Random Erasing Data Augmentation" - AAAI 2020
    
    Args:
        probability (float): 应用增强的概率
        sl (float): 擦除区域最小面积比例
        sh (float): 擦除区域最大面积比例
        r1 (float): 擦除区域最小宽高比
        r2 (float): 擦除区域最大宽高比
        mean (tuple): 用于填充的均值（如果mode='pixel'）
        mode (str): 'random' 随机像素填充, 'pixel' 固定像素填充, 'black' 黑色填充
    """
    
    def __init__(
        self,
        probability=0.5,
        sl=0.02,
        sh=0.4,
        r1=0.3,
        r2=1/0.3,
        mean=(0.4914, 0.4822, 0.4465),
        mode='random'
    ):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
        self.mean = mean
        self.mode = mode
        
    def __call__(self, img):
        """
        Args:
            img (Tensor): 输入图像 (C, H, W)
        
        Returns:
            Tensor: 增强后的图像
        """
        if random.uniform(0, 1) >= self.probability:
            return img
        
        for _ in range(100):
            area = img.size()[1] * img.size()[2]
            
            # 随机目标面积
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, self.r2)
            
            # 计算擦除区域的高度和宽度
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                
                if self.mode == 'random':
                    # 随机像素填充
                    img[0, x1:x1+h, y1:y1+w] = torch.rand(h, w)
                    img[1, x1:x1+h, y1:y1+w] = torch.rand(h, w)
                    img[2, x1:x1+h, y1:y1+w] = torch.rand(h, w)
                elif self.mode == 'pixel':
                    # 均值像素填充
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                elif self.mode == 'black':
                    # 黑色填充
                    img[:, x1:x1+h, y1:y1+w] = 0
                
                return img
        
        return img


class Cutout:
    """
    Cutout Data Augmentation
    在图像中心随机位置切除固定大小的方块
    
    Reference:
        "Improved Regularization of Convolutional Neural Networks with Cutout" - arXiv 2017
    
    Args:
        n_holes (int): 切除方块的数量
        length (int): 方块边长（像素）
        probability (float): 应用增强的概率
        fill_value (float or tuple): 填充值，'mean'使用图像均值，'random'随机值，或指定值
    """
    
    def __init__(self, n_holes=1, length=64, probability=0.5, fill_value='mean'):
        self.n_holes = n_holes
        self.length = length
        self.probability = probability
        self.fill_value = fill_value
        
    def __call__(self, img):
        """
        Args:
            img (Tensor): 输入图像 (C, H, W)
        
        Returns:
            Tensor: 增强后的图像
        """
        if random.uniform(0, 1) >= self.probability:
            return img
        
        h = img.size(1)
        w = img.size(2)
        
        for _ in range(self.n_holes):
            # 随机中心点
            y = random.randint(0, h)
            x = random.randint(0, w)
            
            # 计算切除区域
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            
            # 填充
            if self.fill_value == 'mean':
                img[:, y1:y2, x1:x2] = img.mean()
            elif self.fill_value == 'random':
                img[:, y1:y2, x1:x2] = torch.rand_like(img[:, y1:y2, x1:x2])
            elif self.fill_value == 'black':
                img[:, y1:y2, x1:x2] = 0
            else:
                img[:, y1:y2, x1:x2] = self.fill_value
        
        return img


class GridMask:
    """
    GridMask Data Augmentation
    在图像上应用网格状遮挡
    
    Reference:
        "GridMask Data Augmentation" - arXiv 2020
    
    Args:
        d_range (tuple): 网格间距范围 (min, max)
        ratio (float): 遮挡比例 (0-1)
        probability (float): 应用增强的概率
        mode (str): 遮挡模式 'random'随机填充, 'black'黑色
        rotate (bool): 是否随机旋转网格
    """
    
    def __init__(
        self,
        d_range=(40, 100),
        ratio=0.6,
        probability=0.5,
        mode='black',
        rotate=True
    ):
        self.d_min, self.d_max = d_range
        self.ratio = ratio
        self.probability = probability
        self.mode = mode
        self.rotate = rotate
        
    def __call__(self, img):
        """
        Args:
            img (Tensor): 输入图像 (C, H, W)
        
        Returns:
            Tensor: 增强后的图像
        """
        if random.uniform(0, 1) >= self.probability:
            return img
        
        h, w = img.size(1), img.size(2)
        
        # 随机网格间距
        d = random.randint(self.d_min, self.d_max)
        l = int(d * self.ratio)
        
        # 创建mask
        mask = torch.ones_like(img)
        
        # 随机起始位置
        st_h = random.randint(0, d)
        st_w = random.randint(0, d)
        
        # 生成网格mask
        for i in range(st_h, h, d):
            mask[:, i:min(i+l, h), :] = 0
        
        for j in range(st_w, w, d):
            mask[:, :, j:min(j+l, w)] = 0
        
        # 应用mask
        if self.mode == 'black':
            img = img * mask
        elif self.mode == 'random':
            img = img * mask + torch.rand_like(img) * (1 - mask)
        
        return img


class PartDropout:
    """
    Part-based Dropout for Vehicle Re-ID
    随机丢弃图像的某些部分（如车辆的上半部分或下半部分）
    
    Args:
        probability (float): 应用增强的概率
        parts (int): 将图像划分为几个部分（水平划分）
        drop_parts (int): 随机丢弃几个部分
        mode (str): 'random', 'black', 'mean'
    """
    
    def __init__(self, probability=0.3, parts=4, drop_parts=1, mode='black'):
        self.probability = probability
        self.parts = parts
        self.drop_parts = drop_parts
        self.mode = mode
        
    def __call__(self, img):
        """
        Args:
            img (Tensor): 输入图像 (C, H, W)
        
        Returns:
            Tensor: 增强后的图像
        """
        if random.uniform(0, 1) >= self.probability:
            return img
        
        h = img.size(1)
        part_h = h // self.parts
        
        # 随机选择要丢弃的部分
        drop_indices = random.sample(range(self.parts), self.drop_parts)
        
        for idx in drop_indices:
            y1 = idx * part_h
            y2 = (idx + 1) * part_h if idx < self.parts - 1 else h
            
            if self.mode == 'black':
                img[:, y1:y2, :] = 0
            elif self.mode == 'mean':
                img[:, y1:y2, :] = img.mean()
            elif self.mode == 'random':
                img[:, y1:y2, :] = torch.rand_like(img[:, y1:y2, :])
        
        return img


class OcclusionAugmentation:
    """
    综合遮挡增强策略
    随机选择一种遮挡方法应用
    
    Args:
        strategy (str): 'random_erasing', 'cutout', 'gridmask', 'part_dropout', 'mixed'
        probability (float): 应用增强的概率
        **kwargs: 传递给具体增强方法的参数
    """
    
    def __init__(self, strategy='random_erasing', probability=0.5, **kwargs):
        self.strategy = strategy
        self.probability = probability
        
        # 初始化所有可用的增强方法
        self.augmentations = {
            'random_erasing': RandomErasing(probability=probability, **kwargs.get('re_args', {})),
            'cutout': Cutout(probability=probability, **kwargs.get('cutout_args', {})),
            'gridmask': GridMask(probability=probability, **kwargs.get('gridmask_args', {})),
            'part_dropout': PartDropout(probability=probability, **kwargs.get('part_args', {}))
        }
        
    def __call__(self, img):
        """
        Args:
            img (Tensor): 输入图像 (C, H, W)
        
        Returns:
            Tensor: 增强后的图像
        """
        if random.uniform(0, 1) >= self.probability:
            return img
        
        if self.strategy == 'mixed':
            # 随机选择一种增强方法
            method = random.choice(list(self.augmentations.values()))
            return method(img)
        elif self.strategy in self.augmentations:
            return self.augmentations[self.strategy](img)
        else:
            return img


def build_augmentation_config(aug_type='random_erasing', **kwargs):
    """
    构建数据增强配置
    
    Args:
        aug_type (str): 增强类型
        **kwargs: 增强参数
    
    Returns:
        增强对象
    
    Example:
        >>> aug = build_augmentation_config('random_erasing', probability=0.5)
        >>> aug = build_augmentation_config('cutout', n_holes=2, length=64)
        >>> aug = build_augmentation_config('mixed', probability=0.7)
    """
    if aug_type == 'random_erasing':
        return RandomErasing(**kwargs)
    elif aug_type == 'cutout':
        return Cutout(**kwargs)
    elif aug_type == 'gridmask':
        return GridMask(**kwargs)
    elif aug_type == 'part_dropout':
        return PartDropout(**kwargs)
    elif aug_type == 'mixed':
        return OcclusionAugmentation(strategy='mixed', **kwargs)
    elif aug_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
