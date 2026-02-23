"""
特征提取脚本 - Step 1: 图准备阶段
提取CNN特征图用于图节点生成

功能:
1. 加载训练好的BoT-Baseline模型
2. 提取每张图像的特征图 (C, H, W) 和全局特征向量
3. 保存为.pt文件供后续GCN训练使用

交付物:
- train_features.pt: 训练集特征 {image_id: {'featmap': tensor(C,H,W), 'global_feat': tensor(C)}}
- query_features.pt: 查询集特征
- gallery_features.pt: 画廊集特征
"""

import os
import sys
import torch
import yaml
import argparse
from tqdm import tqdm
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bot_baseline.bot_model import build_bot_baseline
from models.bot_baseline.veri_dataset import create_data_loaders


def setup_logging(output_dir):
    """配置日志"""
    log_file = os.path.join(output_dir, 'feature_extraction.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def extract_features(model, data_loader, device, logger):
    """
    提取特征图和全局特征
    
    Returns:
        features_list: list of {
            'featmap': tensor(C, H, W),
            'global_feat': tensor(C),
            'pid': int,
            'camid': int,
            'index': int
        }
    """
    model.eval()
    features_list = []
    
    logger.info(f"开始提取特征，共 {len(data_loader)} 个batch...")
    
    global_index = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Extracting features")):
            # 兼容不同数据集的返回格式
            if len(batch_data) == 4:
                images, labels, pids, camids = batch_data
            else:
                images, pids, camids = batch_data
                labels = None
            
            images = images.to(device)
            
            # 提取特征（返回特征图）
            bn_feat, feat_map = model(images, return_featmap=True)
            
            # 转移到CPU并保存
            bn_feat = bn_feat.cpu()
            feat_map = feat_map.cpu()
            
            batch_size = images.size(0)
            for i in range(batch_size):
                features_list.append({
                    'featmap': feat_map[i],  # (C, H, W)
                    'global_feat': bn_feat[i],  # (C,)
                    'pid': pids[i].item(),
                    'camid': camids[i].item(),
                    'index': global_index
                })
                global_index += 1
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"已处理 {batch_idx + 1}/{len(data_loader)} 批次")
    
    logger.info(f"✅ 特征提取完成，共 {len(features_list)} 张图像")
    return features_list


def main():
    parser = argparse.ArgumentParser(description='提取CNN特征图用于GCN训练')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径 (best_model.pth)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='特征保存目录')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader workers')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info("=" * 80)
    logger.info("🔬 CNN特征提取 - 图准备阶段 Step 1")
    logger.info("=" * 80)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"数据集: {args.data_root}")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器（不使用PK Sampler，顺序加载）
    logger.info("创建数据加载器...")
    train_loader, query_loader, gallery_loader, num_classes = create_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_pk_sampler=False,  # 不使用PK采样，顺序加载所有数据
        p=16,
        k=4
    )
    
    logger.info(f"数据集类别数: {num_classes}")
    logger.info(f"训练集批次: {len(train_loader)}")
    logger.info(f"查询集批次: {len(query_loader)}")
    logger.info(f"画廊集批次: {len(gallery_loader)}")
    
    # 构建模型
    logger.info("构建模型...")
    model = build_bot_baseline(num_classes=num_classes, pretrain=False)
    model = model.to(device)
    
    # 加载checkpoint
    logger.info(f"加载checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✓ 加载epoch {checkpoint.get('epoch', 'unknown')}, mAP: {checkpoint.get('best_mAP', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
        logger.info("✓ 加载权重成功")
    
    model.eval()
    
    # 提取特征
    logger.info("\n" + "=" * 80)
    logger.info("开始提取训练集特征...")
    train_features = extract_features(model, train_loader, device, logger)
    
    logger.info("\n" + "=" * 80)
    logger.info("开始提取查询集特征...")
    query_features = extract_features(model, query_loader, device, logger)
    
    logger.info("\n" + "=" * 80)
    logger.info("开始提取画廊集特征...")
    gallery_features = extract_features(model, gallery_loader, device, logger)
    
    # 保存特征
    logger.info("\n" + "=" * 80)
    logger.info("保存特征到磁盘...")
    
    train_feat_path = os.path.join(args.output_dir, 'train_features.pt')
    query_feat_path = os.path.join(args.output_dir, 'query_features.pt')
    gallery_feat_path = os.path.join(args.output_dir, 'gallery_features.pt')
    
    torch.save(train_features, train_feat_path)
    torch.save(query_features, query_feat_path)
    torch.save(gallery_features, gallery_feat_path)
    
    logger.info(f"✅ 训练集特征: {train_feat_path} ({len(train_features)} 张图像)")
    logger.info(f"✅ 查询集特征: {query_feat_path} ({len(query_features)} 张图像)")
    logger.info(f"✅ 画廊集特征: {gallery_feat_path} ({len(gallery_features)} 张图像)")
    
    # 打印特征维度信息
    if len(train_features) > 0:
        sample_feat = train_features[0]
        logger.info("\n" + "=" * 80)
        logger.info("📊 特征维度信息:")
        logger.info(f"   • 特征图形状: {sample_feat['featmap'].shape}")
        logger.info(f"   • 全局特征形状: {sample_feat['global_feat'].shape}")
        logger.info(f"   • 示例 PID: {sample_feat['pid']}, CamID: {sample_feat['camid']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 特征提取完成！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
