"""
训练 BoT-GCN 模型

Usage:
    python scripts/training/train_bot_gcn.py --config configs/gcn_transformer_configs/bot_gcn_776.yaml
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from models.bot_baseline.bot_gcn_model import BoTGCN
from models.bot_baseline.veri_dataset import VeRiDataset, VehicleIDDataset, split_vehicleid_test
from utils.pk_sampler import PKSampler
from losses.combined_loss import BoTLoss
from train.scheduler import create_warmup_cosine_scheduler
from eval.evaluator import ReIDEvaluator
from utils.reproducibility import set_random_seed


def setup_logger(log_file):
    """设置标准logger"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('BoT-GCN-Training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # 文件handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    
    # 控制台handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train BoT-GCN Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (override config)')
    
    # 消融实验参数 (覆盖config)
    parser.add_argument('--use_gcn', type=lambda x: x.lower() == 'true', default=None,
                       help='Enable GCN branch')
    parser.add_argument('--grid_h', type=int, default=None,
                       help='Grid height')
    parser.add_argument('--grid_w', type=int, default=None,
                       help='Grid width')
    parser.add_argument('--pooling_type', type=str, default=None,
                       choices=['mean', 'max', 'attention'],
                       help='Graph pooling type')
    parser.add_argument('--fusion_type', type=str, default=None,
                       choices=['concat', 'gated', 'add', 'none'],
                       help='Embedding fusion type')
    parser.add_argument('--gcn_num_layers', type=int, default=None,
                       help='Number of GCN layers')
    
    # 其他
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate, no training')
    
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_dataloaders(config):
    data_root = config['DATA']['ROOT']
    batch_size = config['DATA']['BATCH_SIZE']
    num_workers = config['DATA']['NUM_WORKERS']
    use_pk_sampler = config['DATA'].get('USE_PK_SAMPLER', True)
    
    # 检测数据集类型
    is_vehicleid = 'VehicleID' in data_root or config['DATA'].get('DATASET_NAME', '').lower() == 'vehicleid'
    test_size = config['DATA'].get('TEST_SIZE', 'small') if is_vehicleid else None
    
    # 数据增强和预处理
    train_transforms = [
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # 添加数据增强 (如 Random Erasing)
    if 'AUGMENTATION' in config['DATA']:
        from utils.augmentations import build_augmentation_config
        aug_config = config['DATA']['AUGMENTATION']
        aug_type = aug_config.get('TYPE', 'random_erasing')
        aug_prob = aug_config.get('PROBABILITY', 0.5)
        aug_params = aug_config.get('PARAMS', {})
        
        aug = build_augmentation_config(aug_type, probability=aug_prob, **aug_params)
        if aug is not None:
            train_transforms.append(aug)
            print(f"✓ Added data augmentation: {aug_type} (probability={aug_prob}, params={aug_params})")
        else:
            print(f"⚠️ Failed to build augmentation: {aug_type}")
    else:
        print("⚠️ No AUGMENTATION config found, using basic transforms only")
    
    train_transform = T.Compose(train_transforms)
    
    test_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 训练集
    if is_vehicleid:
        train_dataset = VehicleIDDataset(
            root=data_root,
            mode='train',
            transform=train_transform
        )
    else:
        train_dataset = VeRiDataset(
            root=data_root,
            mode='train',
            transform=train_transform
        )
    
    if use_pk_sampler:
        P = config['DATA']['P']
        K = config['DATA']['K']
        train_sampler = PKSampler(
            dataset=train_dataset,
            p=P,
            k=K
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    # 测试集
    if is_vehicleid:
        # VehicleID需要先分割test set
        test_list_map = {
            'small': 'test_list_800.txt',
            'medium': 'test_list_1600.txt',
            'large': 'test_list_2400.txt'
        }
        test_list_name = test_list_map.get(test_size, 'test_list_800.txt')
        query_data, gallery_data = split_vehicleid_test(data_root, test_list_name)
        
        query_dataset = VehicleIDDataset(
            root=data_root,
            mode='query',
            test_data=query_data,
            transform=test_transform
        )
        
        gallery_dataset = VehicleIDDataset(
            root=data_root,
            mode='gallery',
            test_data=gallery_data,
            transform=test_transform
        )
    else:
        query_dataset = VeRiDataset(
            root=data_root,
            mode='query',
            transform=test_transform
        )
        
        gallery_dataset = VeRiDataset(
            root=data_root,
            mode='gallery',
            transform=test_transform
        )
    
    query_loader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, query_loader, gallery_loader, len(train_dataset.pids)


def build_model(config, num_classes, args):
    """构建模型"""
    model_config = config['MODEL']
    
    # 从 config 读取参数
    model_params = {
        'num_classes': num_classes,
        'last_stride': model_config.get('LAST_STRIDE', 1),
        'pretrain_path': model_config.get('PRETRAINED_PATH', ''),
        'use_gcn': model_config.get('USE_GCN', True),
        'grid_h': model_config.get('GRID_H', 4),
        'grid_w': model_config.get('GRID_W', 4),
        'gcn_hidden_dim': model_config.get('GCN_HIDDEN_DIM', 512),
        'gcn_num_layers': model_config.get('GCN_NUM_LAYERS', 1),
        'gcn_dropout': model_config.get('GCN_DROPOUT', 0.5),
        'pooling_type': model_config.get('POOLING_TYPE', 'mean'),
        'pooling_hidden_dim': model_config.get('POOLING_HIDDEN_DIM', 128),
        'fusion_type': model_config.get('FUSION_TYPE', 'concat'),
        'fusion_hidden_dim': model_config.get('FUSION_HIDDEN_DIM', 512),
        'fusion_dropout': model_config.get('FUSION_DROPOUT', 0.5),
        'neck': model_config.get('NECK', 'bnneck'),
    }
    
    # 命令行参数覆盖
    if args.use_gcn is not None:
        model_params['use_gcn'] = args.use_gcn
    if args.grid_h is not None:
        model_params['grid_h'] = args.grid_h
    if args.grid_w is not None:
        model_params['grid_w'] = args.grid_w
    if args.pooling_type is not None:
        model_params['pooling_type'] = args.pooling_type
    if args.fusion_type is not None:
        model_params['fusion_type'] = args.fusion_type
    if args.gcn_num_layers is not None:
        model_params['gcn_num_layers'] = args.gcn_num_layers
    
    model = BoTGCN(**model_params)
    
    return model


def build_optimizer(config, model):
    """构建优化器"""
    opt_config = config['OPTIMIZER']
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt_config['BASE_LR'],
        weight_decay=opt_config['WEIGHT_DECAY'],
        eps=opt_config['EPS']
    )
    
    return optimizer


def build_scheduler(config, optimizer):
    """构建学习率调度器"""
    sched_config = config['SCHEDULER']
    epochs = config['EPOCHS']
    warmup_epochs = sched_config['WARMUP_EPOCHS']
    
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=epochs,
        eta_min_ratio=1e-7 / config['OPTIMIZER']['BASE_LR']
    )
    
    return scheduler


def build_criterion(config, num_classes):
    """构建损失函数"""
    loss_config = config['LOSS']
    
    criterion = BoTLoss(
        num_classes=num_classes,
        epsilon=loss_config['LABEL_SMOOTH'],
        margin=loss_config['TRIPLET_MARGIN']
    )
    
    return criterion


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, 
                    scaler, device, epoch, config, logger):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    use_amp = config.get('USE_AMP', False)
    
    for batch_idx, (images, pids, camids, _) in enumerate(train_loader):
        images = images.to(device)
        pids = pids.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                logits, features = model(images)
                loss, id_loss, triplet_loss = criterion(logits, features, pids)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, features = model(images)
            loss, id_loss, triplet_loss = criterion(logits, features, pids)
            loss.backward()
            optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += pids.size(0)
        correct += predicted.eq(pids).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    id_acc = 100. * correct / total
    
    return avg_loss, id_acc


@torch.no_grad()
def evaluate(model, query_loader, gallery_loader, device):
    model.eval()
    evaluator = ReIDEvaluator(model=model, use_flip_test=False, use_rerank=False, device=device)
    return evaluator.evaluate(query_loader, gallery_loader, metric='cosine')


def main():
    args = parse_args()
    config = load_config(args.config)
    
    output_dir = args.output_dir if args.output_dir else config.get('OUTPUT_DIR', 'outputs/bot_gcn')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seed = config.get('SEED', 42)
    set_random_seed(seed)
    
    # 设置logger
    logger = setup_logger(output_dir / 'training.log')
    
    # 会话信息
    session_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    logger.info("=" * 80)
    logger.info("🚀 Starting BoT-GCN Training Session")
    logger.info("=" * 80)
    logger.info("📋 Training Configuration:")
    logger.info(f"   • Config File: {args.config}")
    logger.info(f"   • Total Epochs: {config['EPOCHS']}")
    logger.info(f"   • Output Directory: {output_dir}")
    logger.info(f"   • Log File: {output_dir / 'training.log'}")
    logger.info(f"   • Session Time: {session_time}")
    
    # Data Augmentation Configuration
    if 'AUGMENTATION' in config.get('DATA', {}):
        aug_config = config['DATA']['AUGMENTATION']
        aug_type = aug_config.get('TYPE', 'none')
        aug_prob = aug_config.get('PROBABILITY', 0.0)
        logger.info(f"   • Data Augmentation: {aug_type.upper()} (probability={aug_prob})")
        if aug_type == 'random_erasing' and 'PARAMS' in aug_config:
            params = aug_config['PARAMS']
            logger.info(f"     - Erasing Area: {params.get('sl', 0.02)*100:.1f}% ~ {params.get('sh', 0.4)*100:.1f}%")
            logger.info(f"     - Aspect Ratio: {params.get('r1', 0.3):.2f} ~ {params.get('r2', 3.33):.2f}")
            logger.info(f"     - Fill Mode: {params.get('mode', 'random')}")
    else:
        logger.info(f"   • Data Augmentation: NONE (baseline)")
    
    # 设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using device: cuda ({device_name})")
    else:
        logger.info(f"Using device: cpu")
    
    use_amp = config.get('USE_AMP', False)
    if use_amp:
        logger.info("Device acceleration: CUDA + AMP")
    else:
        logger.info("Device acceleration: CUDA")
    
    # 数据加载
    train_loader, query_loader, gallery_loader, num_classes = build_dataloaders(config)
    logger.info(f"Dataset loaded: {num_classes} identities")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Query batches: {len(query_loader)}")
    logger.info(f"Gallery batches: {len(gallery_loader)}")
    
    # 模型构建
    model = build_model(config, num_classes, args).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model built: {num_params:,} parameters ({num_trainable:,} trainable)")
    if model.use_gcn:
        logger.info(f"GCN Config: Grid {config['MODEL']['GRID_H']}×{config['MODEL']['GRID_W']}, "
                   f"Pooling={config['MODEL']['POOLING_TYPE']}, "
                   f"Fusion={config['MODEL']['FUSION_TYPE']}")
    
    # 优化器和调度器
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    criterion = build_criterion(config, num_classes)
    criterion = criterion.to(device)
    
    # AMP
    scaler = GradScaler() if use_amp else None
    
    # Resume
    start_epoch = 1
    best_mAP = 0.0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint.get('mAP', 0.0)
        logger.info(f"Resumed from epoch {start_epoch-1}, best mAP: {best_mAP:.4f}")
    
    # Eval only
    if args.eval_only:
        logger.info("=" * 80)
        logger.info("Evaluation Mode")
        logger.info("=" * 80)
        results = evaluate(model, query_loader, gallery_loader, device)
        logger.info(f"mAP: {results['mAP']:.4f}")
        logger.info(f"Rank-1: {results['rank1']:.4f}")
        logger.info(f"Rank-5: {results['rank5']:.4f}")
        logger.info(f"Rank-10: {results['rank10']:.4f}")
        return
    
    # 训练
    epochs = config['EPOCHS']
    eval_period = config['EVAL_PERIOD']
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, epochs + 1):
        # 训练
        avg_loss, id_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, config, logger
        )
        
        # 更新学习率
        scheduler.step()
        
        logger.info(f"Epoch {epoch} Training - Loss: {avg_loss:.4f}, ID_Acc: {id_acc:.2f}%")
        
        # 评估
        if epoch % eval_period == 0:
            logger.info(f"Evaluating at epoch {epoch}...")
            
            results = evaluate(model, query_loader, gallery_loader, device)
            
            mAP = results['mAP']
            rank1 = results['rank1']
            rank5 = results['rank5']
            rank10 = results['rank10']
            
            logger.info(f"Epoch {epoch} Evaluation - mAP: {mAP:.4f}, Rank-1: {rank1:.4f}, "
                       f"Rank-5: {rank5:.4f}, Rank-10: {rank10:.4f}")
            
            # 保存最佳模型
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mAP': mAP,
                    'rank1': rank1,
                    'config': config
                }, output_dir / 'best_model.pth')
                logger.info(f"New best model saved! mAP: {mAP:.4f}")
        
        # 定期保存checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'mAP': best_mAP,
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # 最终评估
    logger.info("Training completed! Running final evaluation...")
    
    checkpoint = torch.load(output_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    
    results = evaluate(model, query_loader, gallery_loader, device)
    
    logger.info("=" * 80)
    logger.info("🎯 FINAL EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info("📊 Performance Metrics:")
    logger.info(f"   • mAP (Mean Average Precision): {results['mAP']:.4f}")
    logger.info("   • CMC Curve Results:")
    logger.info(f"     - Rank-1 : {results['rank1']:.4f} ({results['rank1']*100:.2f}%)")
    logger.info(f"     - Rank-5 : {results['rank5']:.4f} ({results['rank5']*100:.2f}%)")
    logger.info(f"     - Rank-10: {results['rank10']:.4f} ({results['rank10']*100:.2f}%)")
    logger.info(f"   • Best Training mAP: {best_mAP:.4f}")
    logger.info("=" * 80)
    logger.info(f"✅ Training completed successfully after {epochs} epochs!")
    logger.info(f"📁 Results saved in: {output_dir}")
    logger.info(f"📝 Log file: {output_dir / 'training.log'}")
    logger.info(f"🕐 Session completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
