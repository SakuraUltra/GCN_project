"""
Reproducibility Utilities
确保实验可复现的工具函数
"""

import random
import numpy as np
import torch
import os


def set_random_seed(seed=42, deterministic=False):
    """
    设置所有随机种子，确保实验可复现
    
    Args:
        seed (int): 随机种子，默认42
        deterministic (bool): 是否使用确定性算法（会降低性能），默认False
    
    Example:
        >>> from utils import set_random_seed
        >>> set_random_seed(42)
        >>> # 现在所有随机操作都是可复现的
    
    Note:
        - deterministic=True时会使用确定性CUDA算法，但可能降低性能
        - 某些操作（如数据增强）可能仍有随机性
    """
    # Python内置random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    
    # 确定性算法（可选）
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
    else:
        # 非确定性但更快
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f'✓ Random seed set to {seed} (deterministic={deterministic})')


def get_reproducibility_info():
    """
    获取当前环境的可复现性信息
    
    Returns:
        dict: 包含PyTorch版本、CUDA版本、CuDNN设置等信息
    """
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
    
    return info
