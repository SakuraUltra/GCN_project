"""
Evaluation Entry Point
BoT-Baseline 评估主入口
"""

import os
import sys
import argparse
import yaml
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.bot_baseline.bot_model import build_bot_baseline
from models.bot_baseline.veri_dataset import create_data_loaders
from eval import ReIDEvaluator
from utils.simple_logger import setup_logger
from utils.reproducibility import set_random_seed


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='BoT-Baseline Evaluation for Vehicle Re-ID')
    parser.add_argument('--config', type=str, default='configs/baseline_configs/bot_baseline.yaml',
                        help='Path to config file')
    parser.add_argument('--data_root', type=str, default='data/776_DataSet',
                        help='Path to VeRi-776 dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                        help='Output directory for evaluation results')
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean'],
                        help='Distance metric for evaluation')
    parser.add_argument('--use_flip_test', action='store_true', default=True,
                        help='Use horizontal flip for test-time augmentation')
    parser.add_argument('--use_rerank', action='store_true',
                        help='Use re-ranking for post-processing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    set_random_seed(args.seed, deterministic=False)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(args.output_dir, 'evaluation.log')
    logger = setup_logger('BoT-Eval', log_file)
    logger.info('=' * 80)
    logger.info('🔍 BoT-Baseline Evaluation Session')
    logger.info('=' * 80)
    logger.info(f'📋 Configuration:')
    logger.info(f'   • Config: {args.config}')
    logger.info(f'   • Data Root: {args.data_root}')
    logger.info(f'   • Checkpoint: {args.checkpoint}')
    logger.info(f'   • Metric: {args.metric}')
    logger.info(f'   • Flip Test: {args.use_flip_test}')
    logger.info(f'   • Re-ranking: {args.use_rerank}')
    
    # 设备检测
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = f"CUDA ({torch.cuda.get_device_name()})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = "MPS (Apple Silicon)"
    else:
        device = torch.device('cpu')
        device_name = "CPU"
    
    logger.info(f'   • Device: {device} ({device_name})')
    
    # 创建数据加载器
    logger.info('Loading dataset...')
    _, query_loader, gallery_loader, num_classes = create_data_loaders(
        data_root=args.data_root,
        batch_size=config['DATA']['BATCH_SIZE'],
        num_workers=config['DATA']['NUM_WORKERS'],
        use_pk_sampler=False  # 评估时不使用PK采样
    )
    logger.info(f'Query: {len(query_loader.dataset)} images')
    logger.info(f'Gallery: {len(gallery_loader.dataset)} images')
    
    # 构建模型
    logger.info('Building model...')
    model = build_bot_baseline(
        num_classes=num_classes,
        pretrain=False  # 评估时不需要预训练
    ).to(device)
    
    # 加载检查点
    logger.info(f'Loading checkpoint from {args.checkpoint}...')
    try:
        checkpoint = torch.load(args.checkpoint, weights_only=True)
    except Exception as e:
        logger.warning(f"Safe loading failed, using legacy mode: {e}")
        checkpoint = torch.load(args.checkpoint, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f'✓ Checkpoint loaded (epoch {checkpoint.get("epoch", "unknown")})')
    
    # 构建评估器
    evaluator = ReIDEvaluator(
        model=model,
        use_flip_test=args.use_flip_test,
        use_rerank=args.use_rerank,
        device=device
    )
    
    # 执行评估
    logger.info('=' * 80)
    logger.info('Starting evaluation...')
    logger.info('=' * 80)
    
    results = evaluator.evaluate(query_loader, gallery_loader, metric=args.metric)
    
    # 输出结果
    logger.info('=' * 80)
    logger.info('🎯 EVALUATION RESULTS')
    logger.info('=' * 80)
    logger.info(f'mAP: {results["mAP"]:.4f}')
    logger.info(f'CMC Curve:')
    logger.info(f'  Rank-1 : {results["rank1"]:.4f} ({results["rank1"]*100:.2f}%)')
    logger.info(f'  Rank-5 : {results["rank5"]:.4f} ({results["rank5"]*100:.2f}%)')
    logger.info(f'  Rank-10: {results["rank10"]:.4f} ({results["rank10"]*100:.2f}%)')
    logger.info('=' * 80)
    logger.info(f'✅ Results saved in: {args.output_dir}')
    
    # 保存结果到文件
    results_file = os.path.join(args.output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write('=' * 50 + '\n')
        f.write('EVALUATION RESULTS\n')
        f.write('=' * 50 + '\n')
        f.write(f'Checkpoint: {args.checkpoint}\n')
        f.write(f'Metric: {args.metric}\n')
        f.write(f'Flip Test: {args.use_flip_test}\n')
        f.write(f'Re-ranking: {args.use_rerank}\n')
        f.write('-' * 50 + '\n')
        f.write(f'mAP: {results["mAP"]:.4f}\n')
        f.write(f'Rank-1: {results["rank1"]:.4f}\n')
        f.write(f'Rank-5: {results["rank5"]:.4f}\n')
        f.write(f'Rank-10: {results["rank10"]:.4f}\n')
        f.write('=' * 50 + '\n')
    
    logger.info(f'Results saved to: {results_file}')


if __name__ == '__main__':
    main()
