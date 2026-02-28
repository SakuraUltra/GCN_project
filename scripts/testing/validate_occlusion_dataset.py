#!/usr/bin/env python3
"""
遮挡数据集验证脚本
全面检查遮挡测试集的正确性
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def validate_occlusion_ratio(img_path, expected_ratio, tolerance=0.02):
    """
    验证图像的实际遮挡比例是否符合预期
    
    Args:
        img_path: 图像路径
        expected_ratio: 期望的遮挡比例
        tolerance: 允许的误差范围
    
    Returns:
        (is_valid, actual_ratio, error)
    """
    img = Image.open(img_path)
    img_array = np.array(img)
    
    # 检测黑色像素
    # 注意：JPEG压缩会导致纯黑色(0,0,0)变为接近黑色的值(1-10)
    # 因此使用阈值检测而不是严格相等
    if len(img_array.shape) == 3:
        # 所有通道都<=10视为黑色
        black_pixels = np.all(img_array <= 10, axis=2)
    else:
        black_pixels = (img_array <= 10)
    
    total_pixels = img_array.shape[0] * img_array.shape[1]
    occluded_pixels = np.sum(black_pixels)
    actual_ratio = occluded_pixels / total_pixels
    error = abs(actual_ratio - expected_ratio)
    
    is_valid = error <= tolerance
    
    return is_valid, actual_ratio, error


def visualize_samples(occlusion_dir, output_path):
    """可视化每种遮挡类型的样本"""
    occlusion_dir = Path(occlusion_dir)
    
    # 选择一个测试图像
    test_img_name = "0002_c002_00030600_0.jpg"
    
    # 定义遮挡配置
    configs = [
        ('occ_00_center', '0% (Baseline)'),
        ('occ_10_top', '10% Top'),
        ('occ_10_bottom', '10% Bottom'),
        ('occ_10_left', '10% Left'),
        ('occ_10_right', '10% Right'),
        ('occ_10_center', '10% Center'),
        ('occ_10_grid', '10% Grid'),
        ('occ_20_bottom', '20% Bottom'),
        ('occ_30_bottom', '30% Bottom'),
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, (folder, title) in enumerate(configs):
        img_path = occlusion_dir / folder / test_img_name
        if img_path.exists():
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(title, fontsize=14, fontweight='bold')
            axes[idx].axis('off')
            
            # 计算实际遮挡比例
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                # 使用阈值检测（适应JPEG压缩）
                black_pixels = np.all(img_array <= 10, axis=2)
                total_pixels = img_array.shape[0] * img_array.shape[1]
                actual_ratio = np.sum(black_pixels) / total_pixels
                axes[idx].text(0.5, -0.05, f'Actual: {actual_ratio*100:.1f}%', 
                             ha='center', transform=axes[idx].transAxes,
                             fontsize=12, color='red')
        else:
            axes[idx].text(0.5, 0.5, 'Image not found', 
                         ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化结果已保存到: {output_path}")


def main():
    print("=" * 80)
    print("遮挡数据集验证")
    print("=" * 80)
    
    occlusion_dir = Path("outputs/occlusion_tests/veri776_query")
    metadata_path = occlusion_dir / "occlusion_test_set_metadata.json"
    
    # 加载元数据
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n数据集信息:")
    print(f"  总图像数: {metadata['total_images']}")
    print(f"  配置数: {len(metadata['occlusion_configs'])}")
    
    # 验证每个配置
    print(f"\n验证遮挡比例...")
    validation_results = []
    
    for config in tqdm(metadata['occlusion_configs'], desc="验证配置"):
        config_name = config['config_name']
        
        # 从配置名称解析比例和类型 (如 occ_10_bottom -> 10%, bottom)
        parts = config_name.split('_')
        expected_ratio = int(parts[1]) / 100.0
        occ_type = parts[2] if len(parts) > 2 else 'center'
        
        folder_path = occlusion_dir / config_name
        
        # 随机抽样50张图像验证（更准确的平均值）
        sample_images = list(folder_path.glob("*.jpg"))[:50]
        
        ratios = []
        for img_path in sample_images:
            is_valid, actual_ratio, error = validate_occlusion_ratio(
                img_path, expected_ratio, tolerance=0.03
            )
            ratios.append(actual_ratio)
        
        avg_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        error = abs(avg_ratio - expected_ratio)
        
        validation_results.append({
            'config': config_name,
            'expected': expected_ratio,
            'actual': avg_ratio,
            'std': std_ratio,
            'error': error,
            'status': '✓' if error <= 0.03 else '✗'
        })
    
    # 打印验证结果
    print(f"\n" + "=" * 80)
    print("验证结果汇总")
    print("=" * 80)
    print(f"{'配置':<20} {'期望比例':<12} {'实际比例':<12} {'误差':<10} {'状态':<5}")
    print("-" * 80)
    
    all_valid = True
    for result in validation_results:
        print(f"{result['config']:<20} {result['expected']*100:>6.1f}% "
              f"{result['actual']*100:>8.1f}±{result['std']*100:.1f}% "
              f"{result['error']*100:>7.1f}% {result['status']:>6}")
        if result['status'] == '✗':
            all_valid = False
    
    print("=" * 80)
    
    # 可视化样本
    print(f"\n生成可视化样本...")
    visualize_samples(occlusion_dir, "outputs/occlusion_validation.png")
    
    # 最终结论
    print(f"\n" + "=" * 80)
    if all_valid:
        print("✓ 所有配置的遮挡比例均符合预期！")
    else:
        print("✗ 部分配置的遮挡比例不符合预期，请检查！")
    print("=" * 80)
    
    # 保存验证报告
    report_path = occlusion_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"\n验证报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
