#!/usr/bin/env python3
"""
Single Model Occlusion Test Script
单个模型在指定遮挡级别上的测试脚本

Usage:
    python scripts/testing/test_single_model.py \
        --model-path outputs/bot_gcn_776_v2/best_model.pth \
        --model-type resnet_gcn \
        --query-dir outputs/occlusion_tests_v2/query_00pct \
        --output-dir outputs/results/test
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import argparse
import json
from pathlib import Path
import glob
import re
from PIL import Image

from models.bot_baseline.bot_gcn_model import BoTGCN
from eval.evaluator import ReIDEvaluator


class SimpleVeRiDataset(Dataset):
    """简化的VeRi数据集，直接从指定目录加载"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = self._parse_data()
    
    def _parse_data(self):
        img_paths = glob.glob(os.path.join(self.image_dir, '*.jpg'))
        data = []
        for img_path in img_paths:
            filename = os.path.basename(img_path)
            pattern = r'(\d+)_c(\d+)_(\d+)_(\d+)\.jpg'
            match = re.match(pattern, filename)
            if match:
                vehicle_id = int(match.group(1))
                camera_id = int(match.group(2))
                data.append((img_path, vehicle_id, camera_id))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, pid, camid = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid


def parse_args():
    parser = argparse.ArgumentParser(description='Test single model on occlusion dataset')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, required=True, 
                        choices=['resnet_baseline', 'resnet_gcn', 'vit_baseline', 'vit_gcn', 'vit_native768'],
                        help='Model architecture type')
    parser.add_argument('--query-dir', type=str, required=True, help='Query images directory')
    parser.add_argument('--gallery-dir', type=str, default='data/dataset/776_DataSet/image_test',
                        help='Gallery images directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    return parser.parse_args()


def get_model_config(model_type):
    """根据模型类型返回配置"""
    configs = {
        'resnet_baseline': {'backbone': 'resnet', 'use_gcn': False},
        'resnet_gcn': {'backbone': 'resnet', 'use_gcn': True},
        'vit_baseline': {'backbone': 'vit', 'use_gcn': False},
        'vit_gcn': {'backbone': 'vit', 'use_gcn': True},
        'vit_native768': {'backbone': 'vit', 'use_gcn': True},
    }
    return configs[model_type]


def build_model_from_saved_config(cfg):
    """从 checkpoint 保存的 config 重建模型，保证架构完全一致"""
    model_cfg = cfg['MODEL']
    
    # 兼容两种 BACKBONE 格式
    backbone = model_cfg['BACKBONE']
    if isinstance(backbone, dict):
        # ViT 格式：{'NAME': 'vit_base_patch16_224', 'TYPE': 'vit', ...}
        backbone_cfg = backbone
        backbone_name = backbone_cfg.get('NAME', 'resnet50')
        backbone_type = backbone_cfg.get('TYPE', 'vit')
        
        # VIT-25: 处理 native 维度
        vit_native_dim = backbone_cfg.get('NATIVE_DIM', False)
        if vit_native_dim:
            # Native 模式：ViT-Base=768, ViT-Small=384
            if 'base' in backbone_name.lower():
                out_channels = 768
            elif 'small' in backbone_name.lower():
                out_channels = 384
            else:
                out_channels = 512  # 默认
        else:
            # Projected 模式：从配置读取或使用默认 2048
            out_channels = backbone_cfg.get('OUT_CHANNELS', 2048)
        
        vit_target_spatial = backbone_cfg.get('TARGET_SPATIAL', 8)
    else:
        # ResNet 格式：直接是字符串 'resnet50'
        backbone_name = backbone
        backbone_type = 'resnet'
        out_channels = 2048
        vit_target_spatial = 8
        vit_native_dim = False
    
    gcn_cfg = model_cfg.get('GCN', {})
    fusion_cfg = model_cfg.get('FUSION', {})
    
    # 提取关键参数
    num_classes = model_cfg.get('NUM_CLASSES', 576)
    
    # GCN 参数
    use_gcn = gcn_cfg.get('USE_GCN', True)
    grid_h = gcn_cfg.get('GRID_H', 4)
    grid_w = gcn_cfg.get('GRID_W', 4)
    gcn_hidden_dim = gcn_cfg.get('HIDDEN_CHANNELS', 512)
    gcn_out_dim = gcn_cfg.get('OUT_CHANNELS', None)
    gcn_num_layers = gcn_cfg.get('NUM_LAYERS', 1)
    gcn_dropout = gcn_cfg.get('DROPOUT', 0.5)
    
    # ViT 参数
    # 判断是否是 native 模式（已从 BACKBONE 配置读取）
    if backbone_type == 'vit':
        if vit_native_dim:
            vit_proj_channels = out_channels  # 不会用到，但设置一致
        else:
            vit_proj_channels = out_channels
    else:
        vit_native_dim = False
        vit_proj_channels = 2048
    
    # 融合参数
    fusion_type = fusion_cfg.get('TYPE', 'concat')
    
    # ResNet 不应该有 cls_fusion（cls_fusion 会根据 backbone_type 和 use_gcn 自动创建）
    
    # 创建模型
    model = BoTGCN(
        num_classes=num_classes,
        backbone_type=backbone_type,
        vit_model_name=backbone_name if backbone_type == 'vit' else "deit_small_patch16_224.fb_in1k",
        vit_pretrained=False,  # 不需要预训练，直接加载checkpoint
        vit_native_dim=vit_native_dim,
        vit_proj_channels=vit_proj_channels,
        vit_target_spatial=vit_target_spatial,
        use_gcn=use_gcn,
        grid_h=grid_h,
        grid_w=grid_w,
        gcn_hidden_dim=gcn_hidden_dim,
        gcn_out_dim=gcn_out_dim,
        gcn_num_layers=gcn_num_layers,
        gcn_dropout=gcn_dropout,
        fusion_type=fusion_type,
        neck=model_cfg.get('NECK', 'bnneck')
    )
    
    return model


def load_model(model_path, model_type, device):
    """加载模型 - 从checkpoint读取配置以确保架构匹配"""
    
    # 第一步：加载 checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 第二步：从 checkpoint 读取训练时的 config（而不是外部传入的 model_type）
    if 'config' in checkpoint:
        print(f"📋 从checkpoint读取config并重建模型...")
        saved_config = checkpoint['config']
        
        # 打印关键信息
        model_cfg = saved_config['MODEL']
        backbone = model_cfg['BACKBONE']
        
        # 兼容两种格式
        if isinstance(backbone, dict):
            backbone_name = backbone.get('NAME', 'N/A')
        else:
            backbone_name = backbone  # 字符串格式
        
        use_gcn = model_cfg.get('GCN', {}).get('USE_GCN', False)
        num_classes = model_cfg.get('NUM_CLASSES', 576)
        
        print(f"   Backbone: {backbone_name}")
        print(f"   Use GCN: {use_gcn}")
        print(f"   Num classes: {num_classes}")
        
        model = build_model_from_saved_config(saved_config)
    else:
        # 兜底：用旧逻辑（向后兼容没有config的旧checkpoint）
        print(f"⚠️  Checkpoint无config，使用默认配置")
        config = get_model_config(model_type)
        model = BoTGCN(
            num_classes=576,
            backbone_type=config['backbone'],
            use_gcn=config['use_gcn']
        )
    
    # 第三步：提取 state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 第四步：加载权重（strict=False 允许忽略 classifier 等层）
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # 报告加载状态
    loaded_keys = len(state_dict) - len(missing_keys)
    print(f"✅ 加载 {loaded_keys}/{len(state_dict)} 层")
    
    if missing_keys:
        # 通常是 classifier.weight/bias 等测试时不需要的层
        print(f"   ⚠️  缺失 {len(missing_keys)} 层 (通常是classifier)")
    if unexpected_keys:
        print(f"   ⚠️  额外 {len(unexpected_keys)} 层")
    
    return model


def create_dataloaders(query_dir, gallery_dir, batch_size, num_workers, model_config):
    """创建数据加载器"""
    # 根据backbone类型选择输入尺寸
    if model_config['backbone'] == 'vit':
        img_size = (224, 224)  # ViT需要224x224
    else:
        img_size = (256, 256)  # ResNet使用256x256
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    query_dataset = SimpleVeRiDataset(query_dir, transform=transform)
    gallery_dataset = SimpleVeRiDataset(gallery_dir, transform=transform)
    
    query_loader = DataLoader(query_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return query_loader, gallery_loader


def main():
    args = parse_args()
    
    print("="*60)
    print("Single Model Occlusion Test")
    print("="*60)
    print(f"Model Path: {args.model_path}")
    print(f"Model Type: {args.model_type}")
    print(f"Query Dir: {args.query_dir}")
    print(f"Gallery Dir: {args.gallery_dir}")
    print(f"Output Dir: {args.output_dir}")
    print("="*60)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print("\nLoading model...")
    model = load_model(args.model_path, args.model_type, device)
    
    # 获取模型配置
    model_config = get_model_config(args.model_type)
    
    # 创建数据加载器
    print("\nCreating dataloaders...")
    query_loader, gallery_loader = create_dataloaders(
        args.query_dir, args.gallery_dir, args.batch_size, args.num_workers, model_config
    )
    print(f"Query samples: {len(query_loader.dataset)}")
    print(f"Gallery samples: {len(gallery_loader.dataset)}")
    
    # 创建评估器并评估
    print("\nEvaluating...")
    evaluator = ReIDEvaluator(model, use_flip_test=True, device=device)
    results = evaluator.evaluate(query_loader, gallery_loader)
    
    # 打印结果
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"mAP:     {results['mAP']:.4f} ({results['mAP']*100:.2f}%)")
    print(f"Rank-1:  {results['rank1']:.4f} ({results['rank1']*100:.2f}%)")
    print(f"Rank-5:  {results['rank5']:.4f} ({results['rank5']*100:.2f}%)")
    print(f"Rank-10: {results['rank10']:.4f} ({results['rank10']*100:.2f}%)")
    print("="*60)
    
    # 保存结果
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'model_path': args.model_path,
            'model_type': args.model_type,
            'query_dir': args.query_dir,
            'gallery_dir': args.gallery_dir,
            'mAP': float(results['mAP']),
            'rank1': float(results['rank1']),
            'rank5': float(results['rank5']),
            'rank10': float(results['rank10']),
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")


if __name__ == '__main__':
    main()
