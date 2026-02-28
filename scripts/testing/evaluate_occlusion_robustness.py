#!/usr/bin/env python3
"""
Occlusion Robustness Evaluator
使用生成的遮挡测试集评估模型的遮挡鲁棒性
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models.bot_baseline.bot_model import BoTBaseline
from models.bot_baseline.bot_gcn_model import BoTGCN
from models.bot_baseline.veri_dataset import VeRiDataset, build_transforms
from eval.evaluator import ReIDEvaluator
from PIL import Image


class OcclusionDataset(torch.utils.data.Dataset):
    """加载遮挡测试集的数据集类"""
    
    def __init__(self, occlusion_dir, metadata_path=None, transform=None):
        self.occlusion_dir = Path(occlusion_dir)
        self.transform = transform
        
        # 构建数据列表
        self.data = []
        
        # 如果有metadata则使用，否则直接读取目录
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            for img_meta in metadata['images']:
                img_path = self.occlusion_dir / Path(img_meta['relative_path'])
                if img_path.exists():
                    filename = Path(img_meta['original_path']).stem
                    parts = filename.split('_')
                    if len(parts) >= 1:
                        pid = int(parts[0])
                        camid = 0
                        self.data.append((str(img_path), pid, camid))
        else:
            # 直接从目录读取所有jpg文件
            for img_path in sorted(self.occlusion_dir.glob("*.jpg")):
                # VeRi-776格式: 0002_c002_00030600_0.jpg -> ID=2
                filename = img_path.stem
                parts = filename.split('_')
                if len(parts) >= 1:
                    pid = int(parts[0])
                    camid = 0
                    self.data.append((str(img_path), pid, camid))
        
        print(f"Loaded {len(self.data)} images from {occlusion_dir}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, pid, camid = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, pid, camid


def evaluate_occlusion_robustness(
    model_path,
    occlusion_test_dir,
    gallery_dir,
    output_dir,
    device='cuda'
):
    """
    评估模型在不同遮挡程度下的性能
    
    Args:
        model_path (str): 模型checkpoint路径
        occlusion_test_dir (str): 遮挡测试集目录
        gallery_dir (str): Gallery目录（无遮挡）
        output_dir (str): 结果输出目录
        device (str): 设备
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载元数据
    metadata_path = Path(occlusion_test_dir) / 'occlusion_test_set_metadata.json'
    with open(metadata_path, 'r') as f:
        global_metadata = json.load(f)
    
    print("="*80)
    print("OCCLUSION ROBUSTNESS EVALUATION")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Test set: {occlusion_test_dir}")
    print(f"Configurations: {len(global_metadata['occlusion_configs'])}")
    print("="*80 + "\n")
    
    # 加载模型
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 处理不同的checkpoint格式
    if 'config' in checkpoint:
        num_classes = checkpoint['config'].get('MODEL', {}).get('NUM_CLASSES', 576)
    else:
        num_classes = 576  # VeRi-776默认值
    
    # 尝试检测模型类型
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 检查是否有GCN相关的键
    is_gcn_model = any('gcn' in key.lower() or 'graph' in key.lower() for key in state_dict.keys())
    
    if is_gcn_model:
        print("Detected GCN model")
        model = BoTGCN(num_classes=num_classes)
    else:
        print("Detected Baseline model")
        model = BoTBaseline(num_classes=num_classes)
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"✓ Model loaded from epoch {epoch}\n")
    
    # 准备transforms
    _, test_transform = build_transforms()
    
    # 加载gallery (无遮挡)
    print(f"Loading gallery from {gallery_dir}...")
    
    # 检查gallery_dir是否已经是image_test目录，如果是则获取父目录
    gallery_path = Path(gallery_dir)
    if gallery_path.name == 'image_test':
        veri_root = gallery_path.parent
    else:
        veri_root = gallery_path
    
    gallery_dataset = VeRiDataset(str(veri_root), mode='gallery', transform=test_transform)
    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset, batch_size=64, shuffle=False, num_workers=4
    )
    print(f"✓ Gallery: {len(gallery_dataset)} images\n")
    
    # 评估每个遮挡配置
    results = []
    evaluator = ReIDEvaluator(model=model, use_flip_test=False, device=device)
    
    for config in tqdm(global_metadata['occlusion_configs'], desc="Evaluating configs"):
        config_name = config['config_name']
        
        # 解析配置名称: occ_XX_type -> ratio_XX/type
        parts = config_name.split('_')
        ratio_str = parts[1]  # "00", "10", "20", "30"
        occ_type = '_'.join(parts[2:])  # "center", "top", etc.
        
        # 构建新的路径结构: ratio_XX/type
        config_dir = Path(occlusion_test_dir) / f'ratio_{ratio_str}' / occ_type
        config_metadata_path = config_dir / 'metadata.json'
        
        # 检查路径是否存在，如果不存在尝试旧格式
        if not config_dir.exists():
            config_dir = Path(occlusion_test_dir) / config_name
            config_metadata_path = config_dir / 'metadata.json'
        
        print(f"\n{'='*80}")
        print(f"Evaluating: {config_name} ({config_dir})")
        print(f"{'='*80}")
        
        # 加载query (遮挡)
        query_dataset = OcclusionDataset(
            config_dir, 
            config_metadata_path,
            transform=test_transform
        )
        query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=64, shuffle=False, num_workers=4
        )
        
        # 运行评估
        metrics = evaluator.evaluate(query_loader, gallery_loader, metric='cosine')
        
        # 解析配置
        ratio = int(ratio_str) / 100.0
        
        # 记录结果
        result = {
            'config_name': config_name,
            'occlusion_ratio': ratio,
            'occlusion_type': occ_type,
            'mAP': metrics['mAP'],
            'rank1': metrics['rank1'],
            'rank5': metrics['rank5'],
            'rank10': metrics['rank10']
        }
        results.append(result)
        
        print(f"Results: mAP={metrics['mAP']:.4f}, Rank-1={metrics['rank1']:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_csv = output_dir / 'occlusion_robustness_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Results saved to {results_csv}")
    
    # 生成可视化
    plot_occlusion_results(results_df, output_dir)
    
    # 生成报告
    generate_occlusion_report(results_df, output_dir)
    
    return results_df


def plot_occlusion_results(results_df, output_dir):
    """可视化遮挡评估结果"""
    output_dir = Path(output_dir)
    
    # 1. mAP vs Occlusion Ratio (按类型分组)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # mAP
    ax = axes[0, 0]
    for occ_type in results_df['occlusion_type'].unique():
        data = results_df[results_df['occlusion_type'] == occ_type]
        ax.plot(data['occlusion_ratio']*100, data['mAP']*100, 
                marker='o', label=occ_type, linewidth=2)
    ax.set_xlabel('Occlusion Ratio (%)', fontsize=12)
    ax.set_ylabel('mAP (%)', fontsize=12)
    ax.set_title('mAP vs Occlusion Ratio', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rank-1
    ax = axes[0, 1]
    for occ_type in results_df['occlusion_type'].unique():
        data = results_df[results_df['occlusion_type'] == occ_type]
        ax.plot(data['occlusion_ratio']*100, data['rank1']*100,
                marker='s', label=occ_type, linewidth=2)
    ax.set_xlabel('Occlusion Ratio (%)', fontsize=12)
    ax.set_ylabel('Rank-1 (%)', fontsize=12)
    ax.set_title('Rank-1 vs Occlusion Ratio', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Performance Drop
    ax = axes[1, 0]
    baseline = results_df[results_df['occlusion_ratio'] == 0]['mAP'].values[0]
    for occ_type in results_df['occlusion_type'].unique():
        data = results_df[results_df['occlusion_type'] == occ_type]
        drop = (baseline - data['mAP']) * 100
        ax.plot(data['occlusion_ratio']*100, drop,
                marker='^', label=occ_type, linewidth=2)
    ax.set_xlabel('Occlusion Ratio (%)', fontsize=12)
    ax.set_ylabel('mAP Drop (%)', fontsize=12)
    ax.set_title('Performance Drop', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Heatmap
    ax = axes[1, 1]
    pivot = results_df.pivot(index='occlusion_type', 
                             columns='occlusion_ratio', 
                             values='mAP')
    sns.heatmap(pivot*100, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, 
                cbar_kws={'label': 'mAP (%)'})
    ax.set_title('mAP Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Occlusion Ratio', fontsize=12)
    ax.set_ylabel('Occlusion Type', fontsize=12)
    
    plt.tight_layout()
    plot_path = output_dir / 'occlusion_robustness_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {plot_path}")
    plt.close()


def generate_occlusion_report(results_df, output_dir):
    """生成遮挡鲁棒性评估报告"""
    output_dir = Path(output_dir)
    report_path = output_dir / 'occlusion_robustness_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OCCLUSION ROBUSTNESS EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # 整体统计
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        baseline = results_df[results_df['occlusion_ratio'] == 0]['mAP'].values[0]
        f.write(f"Baseline mAP (0% occlusion): {baseline*100:.2f}%\n")
        f.write(f"Number of configurations tested: {len(results_df)}\n")
        f.write(f"Occlusion ratios: {sorted(results_df['occlusion_ratio'].unique())}\n")
        f.write(f"Occlusion types: {list(results_df['occlusion_type'].unique())}\n\n")
        
        # 按遮挡比例分析
        f.write("PERFORMANCE BY OCCLUSION RATIO\n")
        f.write("-"*80 + "\n")
        for ratio in sorted(results_df['occlusion_ratio'].unique()):
            data = results_df[results_df['occlusion_ratio'] == ratio]
            avg_map = data['mAP'].mean()
            avg_r1 = data['rank1'].mean()
            drop = (baseline - avg_map) * 100
            f.write(f"\nOcclusion {int(ratio*100)}%:\n")
            f.write(f"  Average mAP: {avg_map*100:.2f}% (drop: {drop:.2f}%)\n")
            f.write(f"  Average Rank-1: {avg_r1*100:.2f}%\n")
            f.write(f"  Best type: {data.loc[data['mAP'].idxmax(), 'occlusion_type']}\n")
            f.write(f"  Worst type: {data.loc[data['mAP'].idxmin(), 'occlusion_type']}\n")
        
        # 按遮挡类型分析
        f.write("\n" + "="*80 + "\n")
        f.write("PERFORMANCE BY OCCLUSION TYPE\n")
        f.write("-"*80 + "\n")
        for occ_type in results_df['occlusion_type'].unique():
            data = results_df[results_df['occlusion_type'] == occ_type]
            f.write(f"\n{occ_type.upper()}:\n")
            for _, row in data.iterrows():
                drop = (baseline - row['mAP']) * 100
                f.write(f"  {int(row['occlusion_ratio']*100):2d}%: ")
                f.write(f"mAP={row['mAP']*100:5.2f}%, ")
                f.write(f"Rank-1={row['rank1']*100:5.2f}%, ")
                f.write(f"drop={drop:5.2f}%\n")
        
        # 详细表格
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate occlusion robustness')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--occlusion-dir', type=str, required=True,
                        help='Occlusion test set directory')
    parser.add_argument('--gallery-dir', type=str, required=True,
                        help='Gallery directory (without occlusion)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    results_df = evaluate_occlusion_robustness(
        model_path=args.model,
        occlusion_test_dir=args.occlusion_dir,
        gallery_dir=args.gallery_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    print("\n" + "="*80)
    print("✅ Occlusion robustness evaluation completed!")
    print("="*80)


if __name__ == '__main__':
    main()
