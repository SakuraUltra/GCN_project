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
    parser.add_argument('--adjacency_type', type=str, default=None,
                       choices=['4', '8', 'knn', 'hybrid'],
                       help='Adjacency type: 4/8-neighbor, knn, or hybrid')
    parser.add_argument('--knn_k', type=int, default=None,
                       help='kNN k value (for knn/hybrid adjacency)')
    parser.add_argument('--knn_metric', type=str, default=None,
                       choices=['cosine', 'euclidean'],
                       help='kNN similarity metric')
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
    
    # 构建数据增强 transform（支持从 config 读取尺寸）
    train_transform, test_transform = build_transforms(config)
    
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
    gcn_config = model_config.get('GCN', {})
    fusion_config = model_config.get('FUSION', {})
    
    # 兼容两种 BACKBONE 格式：字符串或字典
    backbone_config = model_config.get('BACKBONE', {})
    if isinstance(backbone_config, str):
        # 字符串格式：根据名称判断类型
        backbone_name = backbone_config.lower()
        if 'vit' in backbone_name or 'deit' in backbone_name:
            backbone_type = 'vit'
        else:
            backbone_type = 'resnet'
        backbone_config = {}
    else:
        # 字典格式：默认使用 ViT
        backbone_type = 'vit'
    
    model_params = {
        'num_classes': num_classes,
        'last_stride': model_config.get('LAST_STRIDE', 1),
        'pretrain_path': model_config.get('PRETRAINED_PATH', ''),
        'backbone_type': backbone_type,
        # ViT参数（如果使用ViT backbone）
        'vit_model_name': backbone_config.get('NAME', 'deit_small_patch16_224.fb_in1k'),
        'vit_pretrained': backbone_config.get('PRETRAINED', True),
        # VIT-25: 原生维度模式
        'vit_native_dim': backbone_config.get('NATIVE_DIM', False),
        'vit_proj_channels': backbone_config.get('OUT_CHANNELS', 2048),
        'vit_target_spatial': backbone_config.get('TARGET_SPATIAL', 8),
        # GCN参数（从 GCN 子配置读取）
        'use_gcn': gcn_config.get('USE_GCN', True),
        'grid_h': gcn_config.get('GRID_H', 4),
        'grid_w': gcn_config.get('GRID_W', 4),
        'adjacency_type': gcn_config.get('ADJACENCY_TYPE', '4'),
        'gcn_hidden_dim': gcn_config.get('HIDDEN_CHANNELS', 512),
        'gcn_out_dim': gcn_config.get('OUT_CHANNELS', None),  # VIT-25: 支持自定义输出维度
        'gcn_num_layers': gcn_config.get('NUM_LAYERS', 1),
        'gcn_dropout': gcn_config.get('DROPOUT', 0.5),
        # kNN 参数
        'knn_k': gcn_config.get('KNN_K', 4),
        'knn_metric': gcn_config.get('KNN_METRIC', 'cosine'),
        'knn_detach': gcn_config.get('KNN_DETACH', True),
        # GAT 参数 (VIT-26)
        'gnn_type': gcn_config.get('GNN_TYPE', 'gcn'),
        'gat_heads': gcn_config.get('HEADS', 4),
        'pooling_type': gcn_config.get('POOLING_TYPE', 'mean'),
        'pooling_hidden_dim': gcn_config.get('POOLING_HIDDEN_DIM', 128),
        # Fusion参数（从 FUSION 子配置读取）
        'fusion_type': fusion_config.get('TYPE', 'concat'),
        'fusion_hidden_dim': fusion_config.get('HIDDEN_DIM', 512),
        'fusion_dropout': fusion_config.get('DROPOUT', 0.5),
        'neck': model_config.get('NECK', 'bnneck'),
    }
    
    # 命令行参数覆盖
    if args.use_gcn is not None:
        model_params['use_gcn'] = args.use_gcn
    if args.grid_h is not None:
        model_params['grid_h'] = args.grid_h
    if args.grid_w is not None:
        model_params['grid_w'] = args.grid_w
    if args.adjacency_type is not None:
        model_params['adjacency_type'] = args.adjacency_type
    if args.knn_k is not None:
        model_params['knn_k'] = args.knn_k
    if args.knn_metric is not None:
        model_params['knn_metric'] = args.knn_metric
    if args.pooling_type is not None:
        model_params['pooling_type'] = args.pooling_type
    if args.fusion_type is not None:
        model_params['fusion_type'] = args.fusion_type
    if args.gcn_num_layers is not None:
        model_params['gcn_num_layers'] = args.gcn_num_layers
    
    model = BoTGCN(**model_params)
    
    return model


def build_transforms(config):
    """构建数据增强 transform，尺寸从 config 读取"""
    
    # 从 config 读取，兼容旧 config（没有 INPUT 字段时默认 256）
    input_cfg   = config.get('INPUT', {})
    train_size  = tuple(input_cfg.get('SIZE_TRAIN', [256, 256]))  # (224,224) or (256,256)
    test_size   = tuple(input_cfg.get('SIZE_TEST',  [256, 256]))
    pixel_mean  = input_cfg.get('PIXEL_MEAN', [0.485, 0.456, 0.406])
    pixel_std   = input_cfg.get('PIXEL_STD',  [0.229, 0.224, 0.225])

    # 数据增强配置
    aug_cfg     = config.get('AUGMENTATION', {})
    re_prob     = aug_cfg.get('RANDOM_ERASING_PROB', 0.5) if aug_cfg.get('RANDOM_ERASING', False) else 0.0

    train_transform = T.Compose([
        T.Resize(train_size),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop(train_size),
        T.ToTensor(),
        T.Normalize(mean=pixel_mean, std=pixel_std),
        *([T.RandomErasing(
            p=re_prob,
            scale=(0.02, 0.2),    # 论文 Re-ID 标准 sh=0.2
            ratio=(0.3, 3.3),     # 论文标准
            value='random'         # RE-R：随机像素填充
        )] if re_prob > 0 else []),
    ])

    test_transform = T.Compose([
        T.Resize(test_size),
        T.ToTensor(),
        T.Normalize(mean=pixel_mean, std=pixel_std),
    ])

    print(f"[Transform] train_size={train_size}  test_size={test_size}  RE_prob={re_prob}")
    return train_transform, test_transform


def build_optimizer(config, model):
    """构建优化器，支持 ViT 分组学习率"""

    # 兼容 config 结构：SOLVER 或 OPTIMIZER 字段
    solver     = config.get('SOLVER', config.get('OPTIMIZER', {}))
    base_lr    = solver['BASE_LR']
    weight_decay = solver.get('WEIGHT_DECAY', 1e-4)
    eps          = solver.get('EPS', 1e-8)
    lr_groups    = solver.get('LR_GROUPS', {})

    if lr_groups and hasattr(model, 'backbone') and hasattr(model.backbone, 'vit'):
        # ── ViT 分组学习率模式 ──────────────────────────────────────
        def _lr(key, default=1.0):
            return base_lr * lr_groups.get(key, default)

        # 收集各组参数（用 id 去重，防止重复）
        assigned_ids = set()

        def _collect(module, key, default=1.0):
            params = [p for p in module.parameters() if id(p) not in assigned_ids]
            assigned_ids.update(id(p) for p in params)
            return {'params': params, 'lr': _lr(key, default), 'name': key}

        param_groups = [
            _collect(model.backbone.vit,        'vit_backbone', 1.0),
        ]
        
        # 投影层仅在 native_dim=False 时存在
        if hasattr(model.backbone, 'patch_proj') and model.backbone.patch_proj is not None:
            param_groups.append(_collect(model.backbone.patch_proj,  'patch_proj',  10.0))
        if hasattr(model.backbone, 'cls_proj') and model.backbone.cls_proj is not None:
            param_groups.append(_collect(model.backbone.cls_proj,    'cls_proj',    10.0))
        
        # 只在使用 GCN 时添加 GCN 和 cls_fusion
        if model.use_gcn:
            param_groups.append(_collect(model.gcn, 'gcn', 10.0))
            param_groups.append(_collect(model.cls_fusion, 'cls_fusion', 10.0))

        # 剩余参数（bottleneck、classifier、embedding_fusion 等）
        remaining = [p for p in model.parameters() if id(p) not in assigned_ids]
        if remaining:
            param_groups.append({'params': remaining, 'lr': base_lr * 10.0, 'name': 'others'})

        # 过滤空组（避免 optimizer 报错）
        param_groups = [g for g in param_groups if len(g['params']) > 0]

        # 打印各组学习率
        print("[Optimizer] 分组学习率:")
        for g in param_groups:
            print(f"  {g['name']:20s}: lr={g['lr']:.2e}  params={len(g['params'])}")

        optimizer = optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            eps=eps,
        )

    else:
        # ── 原始单组模式（兼容 ResNet 版本）────────────────────────
        print(f"[Optimizer] 单组 lr={base_lr:.2e}")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
            eps=eps,
        )

    return optimizer


def build_scheduler(config, optimizer):
    """构建学习率调度器"""
    sched_config = config['SCHEDULER']
    epochs = config.get('TRAIN', {}).get('EPOCHS', config.get('EPOCHS', 120))
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
    
    output_dir = args.output_dir if args.output_dir else config.get('OUTPUT', {}).get('DIR', 'outputs/bot_gcn')
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
    logger.info(f"   • Total Epochs: {config.get('TRAIN', {}).get('EPOCHS', config.get('EPOCHS', 120))}")
    logger.info(f"   • Output Directory: {output_dir}")
    logger.info(f"   • Log File: {output_dir / 'training.log'}")
    logger.info(f"   • Session Time: {session_time}")
    
    # Data Augmentation Configuration
    # 支持两种格式: DATA.AUGMENTATION 或顶层 AUGMENTATION
    aug_config = config.get('DATA', {}).get('AUGMENTATION', config.get('AUGMENTATION', {}))
    
    if aug_config:
        # 新格式：AUGMENTATION.RANDOM_ERASING
        if 'RANDOM_ERASING' in aug_config:
            re_enabled = aug_config.get('RANDOM_ERASING', False)
            re_prob = aug_config.get('RANDOM_ERASING_PROB', 0.5) if re_enabled else 0.0
            if re_prob > 0:
                logger.info(f"   • Data Augmentation: RANDOM_ERASING (probability={re_prob})")
            else:
                logger.info(f"   • Data Augmentation: NONE (baseline)")
        # 旧格式：AUGMENTATION.TYPE
        elif 'TYPE' in aug_config:
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
    
    # 加载预训练权重（如果指定）
    pretrain_path = config['MODEL'].get('PRETRAINED_PATH', '')
    if pretrain_path and os.path.exists(pretrain_path):
        logger.info(f"Loading pretrained weights from: {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location='cpu', weights_only=False)
        
        # 提取模型权重（兼容新旧格式）
        if 'model_state_dict' in checkpoint:
            # 旧版 baseline 格式
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            # 新版格式
            state_dict = checkpoint['model']
        else:
            # 直接是 state_dict
            state_dict = checkpoint
        
        # 尝试加载权重（允许部分匹配）
        model_dict = model.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # 只加载backbone和neck的权重，跳过classifier
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
            elif k.startswith('module.'):
                # 处理 DataParallel 保存的权重
                new_k = k.replace('module.', '')
                if new_k in model_dict and model_dict[new_k].shape == v.shape:
                    pretrained_dict[new_k] = v
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model")
    elif pretrain_path:
        logger.warning(f"Pretrained path specified but not found: {pretrain_path}")
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model built: {num_params:,} parameters ({num_trainable:,} trainable)")
    if model.use_gcn:
        # 兼容新旧 config 结构
        model_cfg = config['MODEL']
        gcn_cfg = model_cfg.get('GCN', model_cfg)  # 新config用GCN子字段，旧config直接在MODEL下
        fusion_cfg = model_cfg.get('FUSION', model_cfg)
        
        grid_h = gcn_cfg.get('GRID_H', model_cfg.get('GRID_H', 4))
        grid_w = gcn_cfg.get('GRID_W', model_cfg.get('GRID_W', 4))
        pooling = gcn_cfg.get('POOLING_TYPE', model_cfg.get('POOLING_TYPE', 'mean'))
        fusion = fusion_cfg.get('TYPE', model_cfg.get('FUSION_TYPE', 'concat'))
        gnn_type = gcn_cfg.get('GNN_TYPE', 'gcn').upper()
        
        logger.info(f"GNN Config: Type={gnn_type}, Grid {grid_h}×{grid_w}, Pooling={pooling}, Fusion={fusion}")
    
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
        checkpoint = torch.load(args.resume, weights_only=False)
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
    epochs = config.get('TRAIN', {}).get('EPOCHS', config.get('EPOCHS', 120))
    eval_period = config.get('TRAIN', {}).get('EVAL_PERIOD', config.get('EVAL_PERIOD', 20))
    
    # 早停机制
    patience = config.get('EARLY_STOPPING', {}).get('PATIENCE', 30)
    min_delta = config.get('EARLY_STOPPING', {}).get('MIN_DELTA', 0.001)
    epochs_no_improve = 0
    
    logger.info("Starting training...")
    if patience < epochs:
        logger.info(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")
    
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
            if mAP > best_mAP + min_delta:
                best_mAP = mAP
                epochs_no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mAP': mAP,
                    'rank1': rank1,
                    'config': config
                }, output_dir / 'best_model.pth')
                logger.info(f"New best model saved! mAP: {mAP:.4f}")
            else:
                epochs_no_improve += eval_period
                logger.info(f"No improvement for {epochs_no_improve} epochs (best mAP: {best_mAP:.4f})")
                
                # 早停检查
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered! No improvement for {patience} epochs.")
                    logger.info(f"Best mAP: {best_mAP:.4f} at epoch {epoch - epochs_no_improve}")
                    break
        
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
