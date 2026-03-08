#!/usr/bin/env python3
"""
Batch Occlusion Robustness Evaluation for All Models
批量评估全部10个模型在11个遮挡级别上的鲁棒性

Usage:
    python scripts/testing/evaluate_all_models_occlusion.py [--gpu 0]

Output:
    outputs/occlusion_evaluation_all/
        ├── results_summary.csv       # 汇总表（10模型 × 11级别 = 110行）
        ├── model_comparison.png      # 对比曲线图
        └── individual_results/       # 每个模型的详细结果
            ├── resnet_baseline_nore.csv
            ├── resnet_gcn_nore.csv
            └── ...
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import project modules
from models.bot_baseline.bot_gcn_model import BoTGCN
from models.bot_baseline.veri_dataset import VeRiDataset
from eval.evaluator import ReIDEvaluator


# 定义10个模型配置
MODEL_CONFIGS = [
    # 旧模型（无RE）
    {
        'name': 'ResNet_Baseline_NoRE',
        'path': 'outputs/bot_baseline_1_1/776/baseline_run_01/best_model.pth',
        'backbone': 'resnet50_ibn_a',
        'use_gcn': False,
        'group': 'No_RE',
        'description': 'ResNet50-IBN-a Baseline (无RE)'
    },
    {
        'name': 'ResNet_GCN_NoRE',
        'path': 'outputs/bot_gcn_776_v2/best_model.pth',
        'backbone': 'resnet50_ibn_a',
        'use_gcn': True,
        'group': 'No_RE',
        'description': 'ResNet50-IBN-a + GCN (无RE)'
    },
    {
        'name': 'ViT_Baseline_NoRE',
        'path': 'outputs/bot_vitbase_baseline_nore_776/best_model.pth',
        'backbone': 'vit',
        'use_gcn': False,
        'group': 'No_RE',
        'description': 'ViT-Base Baseline (无RE)'
    },
    {
        'name': 'ViT_Native768_NoRE',
        'path': 'outputs/bot_vitbase_native768_nore_776/best_model.pth',
        'backbone': 'vit',
        'use_gcn': False,
        'group': 'No_RE',
        'description': 'ViT-Base Native 768 (无RE)'
    },
    
    # 旧模型（旧RE 0.02-0.33）
    {
        'name': 'ResNet_GCN_OldRE',
        'path': 'outputs/bot_gcn_776_Random/best_model.pth',
        'backbone': 'resnet50_ibn_a',
        'use_gcn': True,
        'group': 'Old_RE',
        'description': 'ResNet50-IBN-a + GCN (旧RE 0.02-0.33)'
    },
    {
        'name': 'ViT_Baseline_OldRE',
        'path': 'outputs/bot_vitbase_baseline_776/best_model.pth',
        'backbone': 'vit',
        'use_gcn': False,
        'group': 'Old_RE',
        'description': 'ViT-Base Baseline (旧RE 0.02-0.33)'
    },
    {
        'name': 'ViT_GCN_OldRE',
        'path': 'outputs/bot_vitbase_gcn_776/best_model.pth',
        'backbone': 'vit',
        'use_gcn': True,
        'group': 'Old_RE',
        'description': 'ViT-Base + GCN (旧RE 0.02-0.33)'
    },
    
    # 新模型（新RE 0.02-0.2）
    {
        'name': 'ResNet_GCN_NewRE',
        'path': 'outputs/new_re/resnet_gcn_re/best_model.pth',
        'backbone': 'resnet50_ibn_a',
        'use_gcn': True,
        'group': 'New_RE',
        'description': 'ResNet50-IBN-a + GCN (新RE 0.02-0.2)'
    },
    {
        'name': 'ViT_Baseline_NewRE',
        'path': 'outputs/new_re/vitbase_baseline_re/best_model.pth',
        'backbone': 'vit',
        'use_gcn': False,
        'group': 'New_RE',
        'description': 'ViT-Base Baseline (新RE 0.02-0.2)'
    },
    {
        'name': 'ViT_Native768_GCN_NewRE',
        'path': 'outputs/new_re/vitbase_native768_gcn_re/best_model.pth',
        'backbone': 'vit',
        'use_gcn': True,
        'group': 'New_RE',
        'description': 'ViT-Base Native 768 + GCN (新RE 0.02-0.2)'
    },
]

# 11个遮挡级别
OCCLUSION_LEVELS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]


def load_model(config, device):
    """加载模型权重"""
    print(f"\n{'='*60}")
    print(f"Loading: {config['description']}")
    print(f"Path: {config['path']}")
    
    # 创建模型
    model = BoTGCN(
        num_classes=576,  # VeRi-776有576个训练ID
        backbone_type=config['backbone'],
        use_gcn=config['use_gcn']
    )
    
    # 加载权重（PyTorch 2.6+ 需要 weights_only=False）
    checkpoint = torch.load(config['path'], map_location='cpu', weights_only=False)
    
    # 兼容不同的checkpoint格式
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 移除不匹配的层（如classifier）
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded {len(filtered_dict)}/{len(model_dict)} layers")
    return model


def create_dataloaders(occlusion_level, batch_size=64, num_workers=8):
    """创建指定遮挡级别的数据加载器"""
    
    # 数据预处理（测试时不做增强）
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Query目录（遮挡）
    query_dir = f"outputs/occlusion_tests_v2/query_{occlusion_level:02d}pct"
    
    # Gallery目录（正常）
    gallery_dir = "data/dataset/776_DataSet/image_test"
    
    # 创建数据集
    query_dataset = VeRiDataset(query_dir, transform=transform, mode='test')
    gallery_dataset = VeRiDataset(gallery_dir, transform=transform, mode='test')
    
    # 创建数据加载器
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return query_loader, gallery_loader


def evaluate_single_model(config, device, output_dir):
    """评估单个模型在所有遮挡级别上的表现"""
    
    print(f"\n{'='*80}")
    print(f"Evaluating Model: {config['name']}")
    print(f"{'='*80}")
    
    # 加载模型
    model = load_model(config, device)
    
    # 创建评估器
    evaluator = ReIDEvaluator(model, use_flip_test=True, device=device)
    
    # 存储结果
    results = []
    
    # 遍历所有遮挡级别
    for occ_level in OCCLUSION_LEVELS:
        print(f"\n--- Occlusion Level: {occ_level}% ---")
        
        # 创建数据加载器
        query_loader, gallery_loader = create_dataloaders(occ_level)
        
        # 提取特征并评估
        metrics = evaluator.evaluate(query_loader, gallery_loader)
        
        # 记录结果
        result = {
            'model_name': config['name'],
            'model_group': config['group'],
            'occlusion_level': occ_level,
            'mAP': metrics['mAP'],
            'rank1': metrics['rank1'],
            'rank5': metrics['rank5'],
            'rank10': metrics['rank10'],
        }
        results.append(result)
        
        print(f"Results: mAP={metrics['mAP']:.4f}, Rank-1={metrics['rank1']:.4f}")
    
    # 保存单个模型结果
    df = pd.DataFrame(results)
    individual_dir = output_dir / 'individual_results'
    individual_dir.mkdir(parents=True, exist_ok=True)
    csv_path = individual_dir / f"{config['name']}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved results to: {csv_path}")
    
    return results


def plot_comparison(all_results, output_dir):
    """绘制对比图"""
    
    df = pd.DataFrame(all_results)
    
    # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # 1. mAP vs Occlusion Level (按组分类)
    ax1 = axes[0, 0]
    for group in ['No_RE', 'Old_RE', 'New_RE']:
        group_data = df[df['model_group'] == group]
        for model in group_data['model_name'].unique():
            model_data = group_data[group_data['model_name'] == model].sort_values('occlusion_level')
            ax1.plot(model_data['occlusion_level'], model_data['mAP'], 
                    marker='o', label=model, linewidth=2)
    ax1.set_xlabel('Occlusion Level (%)', fontsize=12)
    ax1.set_ylabel('mAP', fontsize=12)
    ax1.set_title('mAP vs Occlusion Level (All Models)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Rank-1 vs Occlusion Level
    ax2 = axes[0, 1]
    for group in ['No_RE', 'Old_RE', 'New_RE']:
        group_data = df[df['model_group'] == group]
        for model in group_data['model_name'].unique():
            model_data = group_data[group_data['model_name'] == model].sort_values('occlusion_level')
            ax2.plot(model_data['occlusion_level'], model_data['rank1'], 
                    marker='s', label=model, linewidth=2)
    ax2.set_xlabel('Occlusion Level (%)', fontsize=12)
    ax2.set_ylabel('Rank-1 Accuracy', fontsize=12)
    ax2.set_title('Rank-1 vs Occlusion Level (All Models)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance Drop (相对0%遮挡的下降)
    ax3 = axes[1, 0]
    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model].sort_values('occlusion_level')
        baseline_mAP = model_data[model_data['occlusion_level'] == 0]['mAP'].values[0]
        drop = (model_data['mAP'] / baseline_mAP - 1) * 100  # 性能下降百分比
        ax3.plot(model_data['occlusion_level'], drop, marker='o', label=model, linewidth=2)
    ax3.set_xlabel('Occlusion Level (%)', fontsize=12)
    ax3.set_ylabel('mAP Drop (%)', fontsize=12)
    ax3.set_title('Relative Performance Drop (vs 0% Occlusion)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # 4. Heat map (模型 × 遮挡级别)
    ax4 = axes[1, 1]
    pivot = df.pivot(index='model_name', columns='occlusion_level', values='mAP')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd_r', ax=ax4, 
                cbar_kws={'label': 'mAP'}, linewidths=0.5)
    ax4.set_title('mAP Heatmap (Model × Occlusion)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Occlusion Level (%)')
    ax4.set_ylabel('Model')
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = output_dir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Batch Occlusion Evaluation')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('outputs') / 'occlusion_evaluation_all' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 验证所有模型权重存在
    print("\n" + "="*80)
    print("Validating Model Weights...")
    print("="*80)
    missing_models = []
    for config in MODEL_CONFIGS:
        if not Path(config['path']).exists():
            print(f"❌ Missing: {config['path']}")
            missing_models.append(config['name'])
        else:
            print(f"✅ Found: {config['name']}")
    
    if missing_models:
        print(f"\n❌ Error: {len(missing_models)} models not found!")
        return
    
    print(f"\n✅ All {len(MODEL_CONFIGS)} models validated!")
    
    # 验证遮挡测试集
    print("\n" + "="*80)
    print("Validating Occlusion Dataset...")
    print("="*80)
    missing_levels = []
    for level in OCCLUSION_LEVELS:
        query_dir = Path(f"outputs/occlusion_tests_v2/query_{level:02d}pct")
        if not query_dir.exists():
            print(f"❌ Missing: {query_dir}")
            missing_levels.append(level)
        else:
            num_images = len(list(query_dir.glob('*.jpg')))
            print(f"✅ Level {level:2d}%: {num_images} images")
    
    if missing_levels:
        print(f"\n❌ Error: {len(missing_levels)} occlusion levels not found!")
        return
    
    print(f"\n✅ All {len(OCCLUSION_LEVELS)} occlusion levels validated!")
    
    # 开始批量评估
    print("\n" + "="*80)
    print("Starting Batch Evaluation...")
    print(f"Total tasks: {len(MODEL_CONFIGS)} models × {len(OCCLUSION_LEVELS)} levels = {len(MODEL_CONFIGS) * len(OCCLUSION_LEVELS)} evaluations")
    print("="*80)
    
    all_results = []
    
    for i, config in enumerate(MODEL_CONFIGS, 1):
        print(f"\n[{i}/{len(MODEL_CONFIGS)}] Processing: {config['name']}")
        
        try:
            results = evaluate_single_model(config, device, output_dir)
            all_results.extend(results)
        except Exception as e:
            print(f"❌ Error evaluating {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存汇总结果
    print("\n" + "="*80)
    print("Saving Summary Results...")
    print("="*80)
    
    summary_df = pd.DataFrame(all_results)
    summary_path = output_dir / 'results_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved summary to: {summary_path}")
    
    # 绘制对比图
    print("\n" + "="*80)
    print("Generating Comparison Plots...")
    print("="*80)
    plot_comparison(all_results, output_dir)
    
    # 打印统计信息
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total models evaluated: {len(MODEL_CONFIGS)}")
    print(f"Total occlusion levels: {len(OCCLUSION_LEVELS)}")
    print(f"Total evaluations: {len(all_results)}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # 显示最佳性能（0%遮挡）
    zero_occ = summary_df[summary_df['occlusion_level'] == 0].sort_values('mAP', ascending=False)
    print("\nTop 5 Models (0% Occlusion):")
    for idx, row in zero_occ.head(5).iterrows():
        print(f"  {row['model_name']}: mAP={row['mAP']:.4f}, Rank-1={row['rank1']:.4f}")
    
    # 显示最鲁棒模型（30%遮挡）
    high_occ = summary_df[summary_df['occlusion_level'] == 30].sort_values('mAP', ascending=False)
    print("\nMost Robust Models (30% Occlusion):")
    for idx, row in high_occ.head(5).iterrows():
        print(f"  {row['model_name']}: mAP={row['mAP']:.4f}, Rank-1={row['rank1']:.4f}")
    
    print("\n✅ All evaluations completed successfully!")


if __name__ == '__main__':
    main()
