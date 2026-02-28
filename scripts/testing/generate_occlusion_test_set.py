#!/usr/bin/env python3
"""
Occlusion Test Set Generator
生成确定性遮挡测试集，用于评估模型对遮挡的鲁棒性

支持的遮挡类型:
1. 固定面积遮挡 (0%, 10%, 20%, 30%)
2. 不同位置遮挡 (上/中/下/左/右/中心)
3. 网格遮挡 (Grid)
4. 随机块遮挡 (Random blocks)

输出:
- 遮挡图像保存到指定目录
- 元数据JSON文件记录遮挡参数
- 可用于官方协议之外的额外评估
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
from collections import defaultdict


class OcclusionGenerator:
    """
    生成确定性遮挡的工具类
    
    Args:
        occlusion_ratio (float): 遮挡面积比例 (0-1)
        occlusion_type (str): 遮挡类型
        fill_value (tuple or str): 填充颜色，'black', 'white', 'mean', 或RGB元组
    """
    
    def __init__(self, occlusion_ratio=0.2, occlusion_type='center', fill_value='black'):
        self.occlusion_ratio = occlusion_ratio
        self.occlusion_type = occlusion_type
        self.fill_value = fill_value
        
    def apply_occlusion(self, img, metadata=None):
        """
        对图像应用遮挡
        
        Args:
            img (PIL.Image or np.ndarray): 输入图像
            metadata (dict): 可选的元数据字典，用于记录遮挡参数
            
        Returns:
            PIL.Image: 遮挡后的图像
            dict: 遮挡元数据
        """
        # 转换为numpy数组
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img.copy()
            
        h, w = img_array.shape[:2]
        
        # 计算遮挡区域
        occlusion_area = int(h * w * self.occlusion_ratio)
        
        # 确定填充值
        if self.fill_value == 'black':
            fill_color = (0, 0, 0)
        elif self.fill_value == 'white':
            fill_color = (255, 255, 255)
        elif self.fill_value == 'mean':
            fill_color = tuple(img_array.mean(axis=(0, 1)).astype(int))
        elif self.fill_value == 'gray':
            fill_color = (128, 128, 128)
        else:
            fill_color = self.fill_value
            
        # 应用不同类型的遮挡
        if self.occlusion_type == 'none':
            occluded_img = img_array
            mask_coords = None
            
        elif self.occlusion_type == 'center':
            # 中心遮挡 - 动态选择方形或矩形，选择误差更小的方案
            occluded_img = img_array.copy()
            
            # 方案1：方形遮挡
            side_length = int(np.sqrt(occlusion_area) + 0.5)
            square_area = side_length ** 2
            square_error = abs(square_area - occlusion_area)
            
            # 方案2：矩形遮挡（保持图像长宽比）
            aspect_ratio = w / h
            rect_h = int(np.sqrt(occlusion_area / aspect_ratio) + 0.5)
            rect_w = int(rect_h * aspect_ratio + 0.5)
            rect_area = rect_h * rect_w
            rect_error = abs(rect_area - occlusion_area)
            
            # 选择误差更小的方案
            if square_error <= rect_error:
                occ_h = occ_w = side_length
            else:
                occ_h, occ_w = rect_h, rect_w
            
            y1 = (h - occ_h) // 2
            x1 = (w - occ_w) // 2
            y2 = y1 + occ_h
            x2 = x1 + occ_w
            occluded_img[y1:y2, x1:x2] = fill_color
            mask_coords = {'y1': y1, 'y2': y2, 'x1': x1, 'x2': x2}
            
        elif self.occlusion_type == 'top':
            # 上部遮挡 - 遮挡面积 = occlusion_h * w = h * w * ratio
            # 因此 occlusion_h = h * ratio
            occlusion_h = int(h * self.occlusion_ratio)
            occluded_img = img_array.copy()
            occluded_img[:occlusion_h, :] = fill_color
            mask_coords = {'y1': 0, 'y2': occlusion_h, 'x1': 0, 'x2': w}
            
        elif self.occlusion_type == 'bottom':
            # 下部遮挡 - 遮挡面积 = occlusion_h * w = h * w * ratio
            # 因此 occlusion_h = h * ratio
            occlusion_h = int(h * self.occlusion_ratio)
            occluded_img = img_array.copy()
            occluded_img[-occlusion_h:, :] = fill_color
            mask_coords = {'y1': h - occlusion_h, 'y2': h, 'x1': 0, 'x2': w}
            
        elif self.occlusion_type == 'middle':
            # 中部水平遮挡 - 遮挡面积 = occlusion_h * w = h * w * ratio
            # 因此 occlusion_h = h * ratio
            occlusion_h = int(h * self.occlusion_ratio)
            y1 = (h - occlusion_h) // 2
            occluded_img = img_array.copy()
            occluded_img[y1:y1+occlusion_h, :] = fill_color
            mask_coords = {'y1': y1, 'y2': y1 + occlusion_h, 'x1': 0, 'x2': w}
            
        elif self.occlusion_type == 'left':
            # 左侧遮挡 - 遮挡面积 = h * occlusion_w = h * w * ratio
            # 因此 occlusion_w = w * ratio
            occlusion_w = int(w * self.occlusion_ratio)
            occluded_img = img_array.copy()
            occluded_img[:, :occlusion_w] = fill_color
            mask_coords = {'y1': 0, 'y2': h, 'x1': 0, 'x2': occlusion_w}
            
        elif self.occlusion_type == 'right':
            # 右侧遮挡 - 遮挡面积 = h * occlusion_w = h * w * ratio
            # 因此 occlusion_w = w * ratio
            occlusion_w = int(w * self.occlusion_ratio)
            occluded_img = img_array.copy()
            occluded_img[:, -occlusion_w:] = fill_color
            mask_coords = {'y1': 0, 'y2': h, 'x1': w - occlusion_w, 'x2': w}
            
        elif self.occlusion_type == 'grid':
            # 网格遮挡 - 棋盘状均匀分布的小块
            occluded_img = img_array.copy()
            
            # 目标遮挡面积
            target_area = h * w * self.occlusion_ratio
            
            # 策略：使用固定的块大小和间隔，生成网格状分布
            # 块大小根据图像尺寸自适应
            total_pixels = h * w
            if total_pixels < 50000:
                block_size = 8
            elif total_pixels < 150000:
                block_size = 12
            else:
                block_size = 16
            
            block_area = block_size ** 2
            
            # 计算需要的块数
            num_blocks_needed = max(int(target_area / block_area + 0.5), 1)
            
            # 计算网格间隔，使块均匀分布在整个图像上
            # 使用aspect ratio确定行列比例
            aspect_ratio = w / h
            grid_rows = max(int(np.sqrt(num_blocks_needed / aspect_ratio) + 0.5), 1)
            grid_cols = max(int(np.ceil(num_blocks_needed / grid_rows)), 1)
            
            # 计算块之间的间隔（包括块本身）
            row_step = h / (grid_rows + 1)
            col_step = w / (grid_cols + 1)
            
            mask_coords = []
            blocks_placed = 0
            
            for i in range(grid_rows):
                if blocks_placed >= num_blocks_needed:
                    break
                for j in range(grid_cols):
                    if blocks_placed >= num_blocks_needed:
                        break
                    
                    # 计算块的中心位置（均匀分布）
                    center_y = int((i + 1) * row_step)
                    center_x = int((j + 1) * col_step)
                    
                    # 计算块的边界
                    y1 = max(0, center_y - block_size // 2)
                    y2 = min(h, y1 + block_size)
                    x1 = max(0, center_x - block_size // 2)
                    x2 = min(w, x1 + block_size)
                    
                    occluded_img[y1:y2, x1:x2] = fill_color
                    mask_coords.append({'y1': y1, 'y2': y2, 'x1': x1, 'x2': x2})
                    blocks_placed += 1
                    
        elif self.occlusion_type == 'random_blocks':
            # 随机块遮挡
            occluded_img = img_array.copy()
            block_size = int(np.sqrt(occlusion_area / 5))  # 5个随机块
            mask_coords = []
            np.random.seed(42)  # 确定性随机
            for _ in range(5):
                y1 = np.random.randint(0, h - block_size)
                x1 = np.random.randint(0, w - block_size)
                y2 = y1 + block_size
                x2 = x1 + block_size
                occluded_img[y1:y2, x1:x2] = fill_color
                mask_coords.append({'y1': y1, 'y2': y2, 'x1': x1, 'x2': x2})
        else:
            raise ValueError(f"Unknown occlusion type: {self.occlusion_type}")
        
        # 构建元数据
        occlusion_metadata = {
            'occlusion_ratio': self.occlusion_ratio,
            'occlusion_type': self.occlusion_type,
            'fill_value': str(fill_color),
            'image_shape': (h, w),
            'mask_coords': mask_coords
        }
        
        # 转换回PIL Image
        occluded_img_pil = Image.fromarray(occluded_img.astype(np.uint8))
        
        return occluded_img_pil, occlusion_metadata


def generate_occlusion_test_set(
    source_dir,
    output_dir,
    occlusion_ratios=[0.0, 0.1, 0.2, 0.3],
    occlusion_types=['center', 'top', 'bottom', 'left', 'right', 'grid'],
    fill_value='black',
    dataset_name='veri776',
    split='query'
):
    """
    生成遮挡测试集
    
    Args:
        source_dir (str): 源图像目录
        output_dir (str): 输出目录
        occlusion_ratios (list): 遮挡比例列表
        occlusion_types (list): 遮挡类型列表
        fill_value (str): 填充值
        dataset_name (str): 数据集名称
        split (str): 数据集划分 (query/gallery)
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f for f in source_dir.rglob('*') 
        if f.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    # 全局元数据
    global_metadata = {
        'dataset_name': dataset_name,
        'split': split,
        'source_dir': str(source_dir),
        'output_dir': str(output_dir),
        'total_images': len(image_files),
        'occlusion_configs': [],
        'image_metadata': {}
    }
    
    # 对每种遮挡配置生成测试集
    for ratio in occlusion_ratios:
        for occ_type in occlusion_types:
            # 跳过0%遮挡的非none类型
            if ratio == 0.0 and occ_type != 'center':
                continue
                
            config_name = f"occ_{int(ratio*100):02d}_{occ_type}"
            config_dir = output_dir / config_name
            config_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*80}")
            print(f"Generating: {config_name} (ratio={ratio}, type={occ_type})")
            print(f"{'='*80}")
            
            # 创建遮挡生成器
            generator = OcclusionGenerator(
                occlusion_ratio=ratio,
                occlusion_type=occ_type if ratio > 0 else 'none',
                fill_value=fill_value
            )
            
            # 处理每张图像
            config_metadata = {
                'config_name': config_name,
                'occlusion_ratio': ratio,
                'occlusion_type': occ_type if ratio > 0 else 'none',
                'fill_value': fill_value,
                'images': []
            }
            
            for img_path in tqdm(image_files, desc=f"Processing {config_name}"):
                try:
                    # 读取图像
                    img = Image.open(img_path).convert('RGB')
                    
                    # 应用遮挡
                    occluded_img, occ_metadata = generator.apply_occlusion(img)
                    
                    # 保存遮挡图像
                    relative_path = img_path.relative_to(source_dir)
                    output_path = config_dir / relative_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    occluded_img.save(output_path)
                    
                    # 记录元数据
                    img_metadata = {
                        'original_path': str(img_path),
                        'relative_path': str(relative_path),
                        'output_path': str(output_path),
                        **occ_metadata
                    }
                    config_metadata['images'].append(img_metadata)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            # 保存配置级元数据
            config_json_path = config_dir / 'metadata.json'
            with open(config_json_path, 'w') as f:
                json.dump(config_metadata, f, indent=2)
            
            global_metadata['occlusion_configs'].append({
                'config_name': config_name,
                'num_images': len(config_metadata['images']),
                'metadata_path': str(config_json_path)
            })
            
            print(f"✓ Saved {len(config_metadata['images'])} images to {config_dir}")
    
    # 保存全局元数据
    global_json_path = output_dir / 'occlusion_test_set_metadata.json'
    with open(global_json_path, 'w') as f:
        json.dump(global_metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Occlusion test set generation completed!")
    print(f"{'='*80}")
    print(f"Total configurations: {len(global_metadata['occlusion_configs'])}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Global metadata: {global_json_path}")
    print(f"{'='*80}\n")
    
    return global_metadata


def create_summary_report(metadata_path):
    """创建遮挡测试集的摘要报告"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("\n" + "="*80)
    print("OCCLUSION TEST SET SUMMARY REPORT")
    print("="*80)
    print(f"Dataset: {metadata['dataset_name']}")
    print(f"Split: {metadata['split']}")
    print(f"Total images: {metadata['total_images']}")
    print(f"Number of configurations: {len(metadata['occlusion_configs'])}")
    print("\nConfigurations:")
    print("-" * 80)
    print(f"{'Config Name':<30} {'Num Images':<15} {'Status'}")
    print("-" * 80)
    
    for config in metadata['occlusion_configs']:
        status = "✓" if config['num_images'] > 0 else "✗"
        print(f"{config['config_name']:<30} {config['num_images']:<15} {status}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate occlusion test sets for robustness evaluation'
    )
    parser.add_argument('--source-dir', type=str, required=True,
                        help='Source image directory (e.g., data/dataset/776_DataSet/image_query)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for occlusion test sets')
    parser.add_argument('--ratios', type=float, nargs='+', 
                        default=[0.0, 0.1, 0.2, 0.3],
                        help='Occlusion ratios (default: 0.0 0.1 0.2 0.3)')
    parser.add_argument('--types', type=str, nargs='+',
                        default=['center', 'top', 'bottom', 'left', 'right', 'grid'],
                        help='Occlusion types')
    parser.add_argument('--fill', type=str, default='black',
                        choices=['black', 'white', 'gray', 'mean'],
                        help='Fill value for occluded regions')
    parser.add_argument('--dataset', type=str, default='veri776',
                        help='Dataset name')
    parser.add_argument('--split', type=str, default='query',
                        choices=['query', 'gallery'],
                        help='Dataset split')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary report after generation')
    
    args = parser.parse_args()
    
    # 生成遮挡测试集
    metadata = generate_occlusion_test_set(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        occlusion_ratios=args.ratios,
        occlusion_types=args.types,
        fill_value=args.fill,
        dataset_name=args.dataset,
        split=args.split
    )
    
    # 显示摘要报告
    if args.summary:
        metadata_path = Path(args.output_dir) / 'occlusion_test_set_metadata.json'
        create_summary_report(metadata_path)


if __name__ == '__main__':
    main()
