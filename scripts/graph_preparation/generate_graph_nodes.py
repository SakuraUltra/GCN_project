#!/usr/bin/env python3
"""
Step 2: 网格池化创建节点特征

将提取的 CNN 特征图 (C, H, W) 通过网格池化转换为图节点特征
- 将特征图划分为 P×Q 网格
- 每个网格单元通过平均池化生成一个节点特征 (C维)
- 最终得到 N=P×Q 个节点，每个节点有 C 维特征
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm


def grid_pooling(featmap, grid_h, grid_w):
    """
    网格池化: 将特征图划分为 grid_h × grid_w 网格，每个网格平均池化为一个节点
    
    Args:
        featmap: (C, H, W) - CNN 特征图
        grid_h: 网格高度 P
        grid_w: 网格宽度 Q
    
    Returns:
        nodes: (N, C) - 图节点特征，N = P × Q
    """
    C, H, W = featmap.shape
    
    # 计算每个网格单元的高度和宽度
    cell_h = H / grid_h
    cell_w = W / grid_w
    
    nodes = []
    for i in range(grid_h):
        for j in range(grid_w):
            # 计算当前网格的像素边界
            y_start = int(i * cell_h)
            y_end = int((i + 1) * cell_h)
            x_start = int(j * cell_w)
            x_end = int((j + 1) * cell_w)
            
            # 提取网格区域并平均池化
            cell_region = featmap[:, y_start:y_end, x_start:x_end]  # (C, cell_h, cell_w)
            node_feat = cell_region.mean(dim=[1, 2])  # (C,)
            
            nodes.append(node_feat)
    
    return torch.stack(nodes)  # (N, C) where N = grid_h × grid_w


def process_split(features, grid_h, grid_w, split_name):
    """
    处理一个数据集分割，为每张图像生成图节点
    
    Args:
        features: list of dicts - 包含 'featmap', 'global_feat', 'pid', 'camid', 'index'
        grid_h: 网格高度
        grid_w: 网格宽度
        split_name: 分割名称 (train/query/gallery)
    
    Returns:
        graph_data: list of dicts - 包含 'nodes', 'global_feat', 'pid', 'camid', 'index'
    """
    graph_data = []
    
    for item in tqdm(features, desc=f"生成 {split_name} 图节点"):
        featmap = item['featmap']  # (C, H, W)
        
        # 网格池化生成节点特征
        nodes = grid_pooling(featmap, grid_h, grid_w)  # (N, C)
        
        graph_data.append({
            'nodes': nodes,              # (N, C) 图节点特征
            'global_feat': item['global_feat'],  # (C,) 全局特征
            'pid': item['pid'],          # 车辆 ID
            'camid': item['camid'],      # 相机 ID
            'index': item['index']       # 图像索引
        })
    
    return graph_data


def main():
    parser = argparse.ArgumentParser(description='网格池化生成图节点特征')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['776', 'VehicleID'],
                        help='数据集名称: 776 或 VehicleID')
    parser.add_argument('--grid-h', type=int, required=True,
                        help='网格高度 P')
    parser.add_argument('--grid-w', type=int, required=True,
                        help='网格宽度 Q')
    parser.add_argument('--features-dir', type=str, default='outputs/features',
                        help='输入特征目录')
    parser.add_argument('--output-dir', type=str, default='outputs/graph_nodes',
                        help='输出图节点目录')
    
    args = parser.parse_args()
    
    # 设置数据集路径
    if args.dataset == '776':
        features_subdir = '776_baseline'
    else:  # VehicleID
        features_subdir = 'VehicleID_baseline'
    
    features_path = Path(args.features_dir) / features_subdir
    output_path = Path(args.output_dir) / f"{features_subdir}_grid_{args.grid_h}x{args.grid_w}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"🔧 网格池化节点生成")
    print("=" * 80)
    print(f"数据集:         {args.dataset}")
    print(f"网格配置:       {args.grid_h} × {args.grid_w} = {args.grid_h * args.grid_w} 个节点")
    print(f"输入特征目录:   {features_path}")
    print(f"输出节点目录:   {output_path}")
    print("=" * 80)
    
    # 处理每个数据分割
    splits = ['train', 'query', 'gallery']
    
    for split in splits:
        print(f"\n📦 处理 {split} 集...")
        
        # 加载特征
        features_file = features_path / f"{split}_features.pt"
        if not features_file.exists():
            print(f"⚠️  跳过: {features_file} 不存在")
            continue
        
        print(f"   加载: {features_file.name}")
        features = torch.load(features_file, map_location='cpu')
        print(f"   图像数量: {len(features):,}")
        
        # 检查特征图尺寸
        sample_featmap = features[0]['featmap']
        C, H, W = sample_featmap.shape
        print(f"   特征图尺寸: ({C}, {H}, {W})")
        
        # 验证网格配置是否合法
        if args.grid_h > H or args.grid_w > W:
            print(f"❌ 错误: 网格尺寸 ({args.grid_h}, {args.grid_w}) 超过特征图尺寸 ({H}, {W})")
            sys.exit(1)
        
        # 生成图节点
        graph_data = process_split(features, args.grid_h, args.grid_w, split)
        
        # 保存图节点数据
        output_file = output_path / f"{split}_graph_nodes.pt"
        torch.save(graph_data, output_file)
        
        # 验证节点形状
        sample = graph_data[0]
        nodes_shape = sample['nodes'].shape
        N, C_node = nodes_shape
        file_size_mb = output_file.stat().st_size / (1024 ** 2)
        
        print(f"   ✅ 保存: {output_file.name}")
        print(f"   节点形状: ({N}, {C_node})")
        print(f"   文件大小: {file_size_mb:.1f} MB")
        
        # 断言验证
        expected_nodes = args.grid_h * args.grid_w
        assert N == expected_nodes, f"节点数不匹配! 期望 {expected_nodes}, 实际 {N}"
        assert C_node == C, f"节点特征维度不匹配! 期望 {C}, 实际 {C_node}"
    
    print("\n" + "=" * 80)
    print("✅ 图节点生成完成!")
    print("=" * 80)
    
    # 打印最终摘要
    print("\n📊 生成摘要:")
    total_size_mb = 0
    for split in splits:
        output_file = output_path / f"{split}_graph_nodes.pt"
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 ** 2)
            total_size_mb += size_mb
            print(f"   {split:8s}: {output_file.name:30s} ({size_mb:6.1f} MB)")
    print(f"   总计: {total_size_mb:.1f} MB")


if __name__ == '__main__':
    main()
