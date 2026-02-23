"""
Training Engine with AMP Support
专为车辆重识别设计的训练引擎，支持Automatic Mixed Precision
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import time
import logging
from tqdm import tqdm


class AMPTrainer:
    """
    Automatic Mixed Precision (AMP) Trainer
    
    核心特性:
    1. Automatic Mixed Precision - 节省显存，加速训练
    2. 梯度累积支持 - 模拟大batch训练
    3. 动态损失缩放 - 防止梯度下溢
    4. 跨平台支持 - CUDA/MPS/CPU
    5. 训练进度监控 - 实时损失和准确率
    
    Args:
        model (nn.Module): 训练模型
        loss_fn (nn.Module): 损失函数（如BoTLoss）
        optimizer: PyTorch优化器
        scheduler: 学习率调度器（可选）
        use_amp (bool): 是否使用AMP，默认True
        grad_accumulation_steps (int): 梯度累积步数，默认1
        device: 训练设备，默认自动检测
    
    Example:
        >>> from losses import BoTLoss
        >>> from train import AMPTrainer, create_warmup_cosine_scheduler
        >>> 
        >>> criterion = BoTLoss(num_classes=576)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=0.00035)
        >>> scheduler = create_warmup_cosine_scheduler(optimizer, 10, 120)
        >>> 
        >>> trainer = AMPTrainer(model, criterion, optimizer, scheduler)
        >>> for epoch in range(120):
        >>>     stats = trainer.train_epoch(train_loader, epoch)
        >>>     print(f"Loss: {stats['avg_loss']:.4f}")
    """
    
    def __init__(self, model, loss_fn, optimizer, scheduler=None, use_amp=True, 
                 grad_accumulation_steps=1, device=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.grad_accumulation_steps = grad_accumulation_steps
        
        # 设备自动检测
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # AMP只支持CUDA
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = GradScaler()
        else:
            self.use_amp = False
            self.scaler = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'AMPTrainer initialized | Device: {self.device} | AMP: {self.use_amp}')
    
    def train_epoch(self, data_loader, epoch, print_freq=50):
        """
        单个epoch的训练流程
        
        Args:
            data_loader: 训练数据加载器
            epoch (int): 当前epoch编号
            print_freq (int): 日志打印频率，默认每50个batch
            
        Returns:
            dict: 训练统计信息
                - avg_loss: 平均总损失
                - avg_id_acc: 平均ID准确率
                - batch_time: 平均batch处理时间
                - data_time: 平均数据加载时间
        """
        self.model.train()
        
        losses = []
        id_accuracies = []
        batch_time = 0
        data_time = 0
        end = time.time()
        
        # 禁用tqdm动态刷新，使用简洁输出避免日志文件中出现多行进度条
        pbar = tqdm(data_loader, desc=f'Epoch {epoch}', dynamic_ncols=False, ncols=100, 
                    mininterval=2.0, file=None)  # mininterval=2秒更新一次，减少输出
        
        for batch_idx, (images, labels, pids, camids) in enumerate(pbar):
            data_time += time.time() - end
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播 + 损失计算
            if self.use_amp:
                with autocast():
                    global_feat, cls_score = self.model(images)
                    loss, id_loss, triplet_loss = self.loss_fn(cls_score, global_feat, labels)
                    id_acc = (cls_score.argmax(1) == labels).float().mean() * 100
            else:
                global_feat, cls_score = self.model(images)
                loss, id_loss, triplet_loss = self.loss_fn(cls_score, global_feat, labels)
                id_acc = (cls_score.argmax(1) == labels).float().mean() * 100
            
            # 反向传播
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss / self.grad_accumulation_steps).backward()
                if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                (loss / self.grad_accumulation_steps).backward()
                if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # 统计
            losses.append(loss.item())
            id_accuracies.append(id_acc.item())
            batch_time += time.time() - end
            end = time.time()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'ID': f'{id_loss.item():.4f}',
                'Tri': f'{triplet_loss.item():.4f}',
                'Acc': f'{id_acc.item():.2f}%'
            })
            
            # 定期打印详细信息
            if batch_idx % print_freq == 0:
                self.logger.info(
                    f'Epoch: [{epoch}][{batch_idx}/{len(data_loader)}] '
                    f'Time {batch_time/60:.2f}min ({batch_time/(batch_idx+1):.3f}s/batch) '
                    f'Data {data_time/(batch_idx+1):.3f}s '
                    f'Loss {loss.item():.6f} '
                    f'ID_Loss {id_loss.item():.6f} '
                    f'Triplet_Loss {triplet_loss.item():.6f} '
                    f'ID_Acc {id_acc.item():.2f}%'
                )
        
        # 学习率调度
        if self.scheduler:
            self.scheduler.step()
        
        # 返回epoch统计
        return {
            'avg_loss': sum(losses) / len(losses),
            'avg_id_acc': sum(id_accuracies) / len(id_accuracies),
            'batch_time': batch_time / len(data_loader),
            'data_time': data_time / len(data_loader)
        }
    
    def save_checkpoint(self, epoch, best_mAP, save_path):
        """
        保存训练检查点
        
        Args:
            epoch (int): 当前epoch
            best_mAP (float): 最佳mAP
            save_path (str): 保存路径
        """
        checkpoint = {
            'epoch': int(epoch),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mAP': float(best_mAP),
            'device': str(self.device),
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, save_path)
        self.logger.info(f'Checkpoint saved: {save_path}')
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载训练检查点
        
        Args:
            checkpoint_path (str): 检查点路径
            
        Returns:
            tuple: (epoch, best_mAP)
        """
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
        except Exception as e:
            self.logger.warning(f"Safe loading failed, using legacy mode: {e}")
            checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        epoch = checkpoint['epoch']
        best_mAP = checkpoint.get('best_mAP', 0.0)
        
        self.logger.info(f'Checkpoint loaded: {checkpoint_path}')
        return epoch, best_mAP
