#!/usr/bin/env python3
"""
验证 kNN 图的正确性
"""

import torch
from pathlib import Path


def verify_knn_file(file_path, expected_k, expected_num_nodes):
    """验证单个 kNN 图文件"""
    print(f"\n  📁 {file_path.name}")
    
    if not file_path.exists():
        print(f"    ❌ 文件不存在")
        return False
    
    file_size_mb = file_path.stat().st_size / (1024 ** 2)
    print(f"    大小: {file_size_mb:.1f} MB")
    
    try:
        # 加载数据
        data = torch.load(file_path, map_location='cpu')
        num_samples = len(data)
        print(f"    样本数: {num_samples:,}")
        
        # 验证前几个样本
        for i in [0, min(1, num_samples-1), num_samples-1]:
            sample = data[i]
            
            edge_index = sample['edge_index']
            edge_weight = sample['edge_weight']
            num_nodes = sample['num_nodes']
            
            # 检查节点数
            if num_nodes != expected_num_nodes:
                print(f"    ❌ 样本 {i}: 节点数错误 ({num_nodes} vs {expected_num_nodes})")
                return False
            
            # 检查边数
            num_edges = edge_index.size(1)
            expected_edges = expected_num_nodes * expected_k
            
            if num_edges != expected_edges:
                print(f"    ❌ 样本 {i}: 边数错误 ({num_edges} vs {expected_edges})")
                return False
            
            # 检查自环
            self_loops = (edge_index[0] == edge_index[1]).sum().item()
            if self_loops > 0:
                print(f"    ❌ 样本 {i}: 包含 {self_loops} 个自环")
                return False
            
            # 检查边权重
            if edge_weight.size(0) != num_edges:
                print(f"    ❌ 样本 {i}: 边权重数量不匹配")
                return False
        
        print(f"    ✅ 验证通过:")
        print(f"       节点数: {expected_num_nodes}")
        print(f"       每节点边数: {expected_k}")
        print(f"       总边数: {expected_num_nodes * expected_k}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ 验证失败: {e}")
        return False


def main():
    print("=" * 80)
    print("🔍 kNN 图验证")
    print("=" * 80)
    
    base_dir = Path('outputs/graph_nodes')
    
    configs = [
        ('776_baseline_grid_4x4_knn4', 4, 16),
        ('776_baseline_grid_8x8_knn8', 8, 64),
        ('VehicleID_baseline_grid_4x4_knn4', 4, 16),
        ('VehicleID_baseline_grid_8x8_knn8', 8, 64),
    ]
    
    results = {}
    
    for config_name, k, num_nodes in configs:
        config_dir = base_dir / config_name
        
        print(f"\n📦 {config_name}")
        
        if not config_dir.exists():
            print(f"  ⚠️  目录不存在")
            results[config_name] = 'SKIP'
            continue
        
        splits = ['train', 'query', 'gallery']
        all_passed = True
        
        for split in splits:
            file_path = config_dir / f"{split}_knn_graph.pt"
            passed = verify_knn_file(file_path, k, num_nodes)
            if not passed:
                all_passed = False
        
        results[config_name] = 'PASS' if all_passed else 'FAIL'
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("📊 验证摘要")
    print("=" * 80)
    for config_name, result in results.items():
        emoji = '✅' if result == 'PASS' else '⚠️' if result == 'SKIP' else '❌'
        print(f"{emoji} {config_name}: {result}")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
