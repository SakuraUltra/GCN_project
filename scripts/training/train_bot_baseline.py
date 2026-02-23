"""
Clean BoT-Baseline Training Script - No conflicts, modular design
使用完全模块化的组件，避免代码重复
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bot_baseline.bot_model import build_bot_baseline
from models.bot_baseline.veri_dataset import create_data_loaders
from losses import BoTLoss
from train import AMPTrainer, create_warmup_cosine_scheduler
from eval import ReIDEvaluator
from utils.simple_logger import setup_logger


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Clean BoT-Baseline Training')
    parser.add_argument('--config', type=str, default='configs/baseline_configs/bot_baseline.yaml',
                        help='Path to config file')
    parser.add_argument('--data_root', type=str, default='776_DataSet',
                        help='Path to VeRi-776 dataset')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory (if not set, auto-create)')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--grid_h', type=int, default=0, help='Grid Pool Height (0 to disable)')
    parser.add_argument('--grid_w', type=int, default=0, help='Grid Pool Width (0 to disable)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine output directory
    if args.output_dir:
        # Use the output directory provided by shell script
        output_dir = args.output_dir
        print(f"Using provided output directory: {output_dir}")
    else:
        # Auto-create output directory (for manual runs)
        if 'VehicleID' in args.data_root:
            dataset_name = 'VehicleID'
        else:
            dataset_name = '776'
        
        # Create base output directory (1:1下采样实验)
        if args.grid_h > 0 and args.grid_w > 0:
            base_output_dir = os.path.join(f'./outputs/grid_scale_1_1/{args.grid_h}x{args.grid_w}', dataset_name)
        else:
            base_output_dir = os.path.join('./outputs/bot_baseline_1_1', dataset_name)
            
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Find next run index
        run_idx = 1
        while True:
            run_name = f'baseline_run_{run_idx:02d}'
            output_dir = os.path.join(base_output_dir, run_name)
            if not os.path.exists(output_dir):
                break
            run_idx += 1
            
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Setup logger - 统一使用独立日志文件
    epochs = config['EPOCHS']
    log_file = os.path.join(output_dir, f'training.log')
    
    logger = setup_logger('BoT-Training', log_file)
    logger.info('=' * 80)
    logger.info('🚀 Starting BoT-Baseline Training Session')
    logger.info('=' * 80)
    logger.info(f'📋 Configuration Summary:')
    logger.info(f'   • Config File: {args.config}')
    logger.info(f'   • Data Root: {args.data_root}')
    logger.info(f'   • Output Directory: {output_dir}')
    logger.info(f'   • Log File: {log_file}')
    logger.info(f'   • Session Time: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('')
    logger.info(f'🎯 Training Parameters:')
    logger.info(f'   • Total Epochs: {epochs}')
    logger.info(f'   • Batch Size: {config["DATA"]["BATCH_SIZE"]}')
    logger.info(f'   • Learning Rate: {config["OPTIMIZER"]["BASE_LR"]}')
    logger.info(f'   • Weight Decay: {config["OPTIMIZER"]["WEIGHT_DECAY"]}')
    logger.info(f'   • Optimizer: {config["OPTIMIZER"]["NAME"]}')
    logger.info(f'   • Scheduler: {config["SCHEDULER"]["NAME"]}')
    logger.info(f'   • Warmup Epochs: {config["SCHEDULER"]["WARMUP_EPOCHS"]}')
    logger.info(f'   • Eval Period: {config["EVAL_PERIOD"]} epochs')
    logger.info(f'   • Use AMP: {config["USE_AMP"]}')
    logger.info('')
    logger.info(f'📊 Data Configuration:')
    logger.info(f'   • PK Sampler: {config["DATA"]["USE_PK_SAMPLER"]}')
    if config["DATA"]["USE_PK_SAMPLER"]:
        logger.info(f'   • P (identities/batch): {config["DATA"]["P"]}')
        logger.info(f'   • K (images/identity): {config["DATA"]["K"]}')
        logger.info(f'   • Effective Batch: {config["DATA"]["P"]} × {config["DATA"]["K"]} = {config["DATA"]["P"] * config["DATA"]["K"]}')
    logger.info(f'   • Num Workers: {config["DATA"]["NUM_WORKERS"]}')
    logger.info('')
    logger.info(f'🎲 Loss Configuration:')
    logger.info(f'   • Label Smoothing: {config["LOSS"]["LABEL_SMOOTH"]}')
    logger.info(f'   • Triplet Margin: {config["LOSS"]["TRIPLET_MARGIN"]}')
    logger.info(f'   • ID Loss Weight: {config["LOSS"]["ID_LOSS_WEIGHT"]}')
    logger.info(f'   • Triplet Loss Weight: {config["LOSS"]["TRIPLET_LOSS_WEIGHT"]}')
    if args.grid_h > 0 and args.grid_w > 0:
        logger.info('')
        logger.info(f'🔬 Grid Pooling Configuration:')
        logger.info(f'   • Grid Size: {args.grid_h} × {args.grid_w}')
        logger.info(f'   • Total Nodes: {args.grid_h * args.grid_w}')
    
    # Setup device - 智能设备选择
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = f"CUDA ({torch.cuda.get_device_name()})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = "MPS (Apple Silicon GPU)"
    else:
        device = torch.device('cpu')
        device_name = "CPU"
    
    logger.info('=' * 80)
    logger.info(f'Using device: {device} ({device_name})')
    logger.info(f'Device acceleration: {"CUDA + AMP" if device.type == "cuda" else "Standard"}')
    logger.info('=' * 80)
    
    # Create data loaders - ✅ 适配YAML大写格式
    train_loader, query_loader, gallery_loader, num_classes = create_data_loaders(
        data_root=args.data_root,
        batch_size=config['DATA']['BATCH_SIZE'],
        num_workers=config['DATA']['NUM_WORKERS'],
        use_pk_sampler=config['DATA']['USE_PK_SAMPLER'],
        p=config['DATA']['P'],
        k=config['DATA']['K']
    )
    
    logger.info(f'Dataset loaded: {num_classes} identities')
    logger.info(f'Train batches: {len(train_loader)} (total ~{len(train_loader) * config["DATA"]["BATCH_SIZE"]} images)')
    logger.info(f'Query batches: {len(query_loader)}')  
    logger.info(f'Gallery batches: {len(gallery_loader)}')
    logger.info('=' * 80)
    
    # Build model - ✅ 适配YAML大写格式
    grid_size = (args.grid_h, args.grid_w) if (args.grid_h > 0 and args.grid_w > 0) else None
    
    logger.info(f'Building model: {config["MODEL"]["NAME"]}')
    logger.info(f'   • Backbone: {config["MODEL"]["BACKBONE"]}')
    logger.info(f'   • Pretrained: {config["MODEL"]["PRETRAINED"]}')
    logger.info(f'   • Feature Dim: {config["MODEL"]["FEATURE_DIM"]}')
    logger.info(f'   • Num Classes: {num_classes}')
    if grid_size:
        logger.info(f'   • Grid Pooling: {grid_size[0]}×{grid_size[1]}')
    
    model = build_bot_baseline(
        num_classes=num_classes,
        pretrain=config['MODEL']['PRETRAINED'],
        grid_size=grid_size
    )
    model = model.to(device)
    
    # Build loss function
    criterion = BoTLoss(
        num_classes=num_classes,
        epsilon=config['LOSS']['LABEL_SMOOTH'],
        margin=config['LOSS']['TRIPLET_MARGIN']
    ).to(device)
    
    # Build optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['OPTIMIZER']['BASE_LR'],
        weight_decay=config['OPTIMIZER']['WEIGHT_DECAY']
    )
    
    # Build scheduler - ✅ 使用PyTorch内置LambdaLR调度器
    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_epochs=config['SCHEDULER']['WARMUP_EPOCHS'],
        max_epochs=config['EPOCHS'],
        eta_min_ratio=0.01  # 最小学习率为初始学习率的1%
    )
    
    # Build trainer - Use modular AMPTrainer
    trainer = AMPTrainer(
        model=model,
        loss_fn=criterion,  # ✅ 关键！传递BoTLoss实例
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=config['USE_AMP'],
        device=device  # ✅ 传递设备信息
    )
    
    # Build evaluator - Use modular ReIDEvaluator  
    evaluator = ReIDEvaluator(
        model=model,
        use_flip_test=True,  # 简化配置，默认使用测试时增强
        device=device  # ✅ 传递设备信息
    )
    
    # Training loop
    best_mAP = 0.0
    start_epoch = 0
    
    if args.resume:
        start_epoch, best_mAP = trainer.load_checkpoint(args.resume)
        logger.info(f'Resumed from epoch {start_epoch}, best mAP: {best_mAP:.4f}')
    
    logger.info('=' * 80)
    logger.info('🚂 Starting Training Loop...')
    logger.info('=' * 80)
    for epoch in range(start_epoch + 1, config['EPOCHS'] + 1):
        
        # Training phase
        train_stats = trainer.train_epoch(train_loader, epoch)
        
        logger.info(f'Epoch {epoch} Training - '
                   f'Loss: {train_stats["avg_loss"]:.4f}, '
                   f'ID_Acc: {train_stats["avg_id_acc"]:.2f}%')
        
        # Evaluation phase
        if epoch % config['EVAL_PERIOD'] == 0:
            logger.info(f'Evaluating at epoch {epoch}...')
            
            eval_results = evaluator.evaluate(query_loader, gallery_loader, metric='cosine')
            
            current_mAP = eval_results['mAP']
            logger.info(f'Epoch {epoch} Evaluation - '
                       f'mAP: {current_mAP:.4f}, '
                       f'Rank-1: {eval_results["rank1"]:.4f}, '
                       f'Rank-5: {eval_results["rank5"]:.4f}, '
                       f'Rank-10: {eval_results["rank10"]:.4f}')
            
            # Save best model
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                best_model_path = os.path.join(output_dir, 'best_model.pth')
                trainer.save_checkpoint(epoch, best_mAP, best_model_path)
                logger.info(f'New best model saved! mAP: {best_mAP:.4f}')
        
        # Save regular checkpoint  
        if epoch % 20 == 0:  # 每20个epoch保存一次
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            trainer.save_checkpoint(epoch, best_mAP, checkpoint_path)
    
    # Final evaluation
    logger.info('Training completed! Running final evaluation...')
    if best_mAP > 0:
        best_model_path = os.path.join(output_dir, 'best_model.pth')
        trainer.load_checkpoint(best_model_path)
    
    final_results = evaluator.evaluate(query_loader, gallery_loader, metric='cosine')
    
    # Training completion summary
    logger.info('=' * 80)
    logger.info('🎯 FINAL EVALUATION RESULTS')
    logger.info('=' * 80)
    logger.info(f'📊 Performance Metrics:')
    logger.info(f'   • mAP (Mean Average Precision): {final_results["mAP"]:.4f}')
    logger.info(f'   • CMC Curve Results:')
    logger.info(f'     - Rank-1 : {final_results["rank1"]:.4f} ({final_results["rank1"]*100:.2f}%)')
    logger.info(f'     - Rank-5 : {final_results["rank5"]:.4f} ({final_results["rank5"]*100:.2f}%)')
    logger.info(f'     - Rank-10: {final_results["rank10"]:.4f} ({final_results["rank10"]*100:.2f}%)')
    logger.info(f'   • Best Training mAP: {best_mAP:.4f}')
    logger.info('=' * 80)
    logger.info(f'✅ Training completed successfully after {epochs} epochs!')
    logger.info(f'📁 Results saved in: {output_dir}')
    logger.info(f'📝 Log file: {log_file}')
    logger.info(f'🕐 Session completed: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('=' * 80)


if __name__ == '__main__':
    main()