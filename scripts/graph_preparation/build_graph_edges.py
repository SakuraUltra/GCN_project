#!/usr/bin/env python3
"""
Step 3: 图边构造 - 邻接矩阵生成

为网格结构的图节点构建邻接关系:
- 4-邻居: 上、下、左、右
- 8-邻居: 4-邻居 + 4个对角线邻居

输出:
- edge_index: [2, E] 边索引
- 验证: 无孤立节点
"""

import torch
import argparse
from pathlib import Path


def build_grid_adjacency_4neighbor(grid_h, grid_w):
    """
    构建4-邻居邻接矩阵（上下左右）
    
    Args:
        grid_h: 网格高度
        grid_w: 网格宽度
    
    Returns:
        edge_index: [2, E] tensor，每列是一条边 [source, target]
    """
    edges = []
    
    for i in range(grid_h):
        for j in range(grid_w):
            node_id = i * grid_w + j
            
            # 上邻居
            if i > 0:
                neighbor_id = (i - 1) * grid_w + j
                edges.append([node_id, neighbor_id])
            
            # 下邻居
            if i < grid_h - 1:
                neighbor_id = (i + 1) * grid_w + j
                edges.append([node_id, neighbor_id])
            
            # 左邻居
            if j > 0:
                neighbor_id = i * grid_w + (j - 1)
                edges.append([node_id, neighbor_id])
            
            # 右邻居
            if j < grid_w - 1:
                neighbor_id = i * grid_w + (j + 1)
                edges.append([node_id, neighbor_id])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()  # [2, E]
    return edge_index


def build_grid_adjacency_8neighbor(grid_h, grid_w):
    """
    构建8-邻居邻接矩阵（上下左右 + 4个对角线）
    
    Args:
        grid_h: 网格高度
        grid_w: 网格宽度
    
    Returns:
        edge_index: [2, E] tensor
    """
    edges = []
    
    # 8个方向的偏移量: 上、下、左、右、左上、右上、左下、右下
    directions = [
        (-1, 0),   # 上
        (1, 0),    # 下
        (0, -1),   # 左
        (0, 1),    # 右
        (-1, -1),  # 左上
        (-1, 1),   # 右上
        (1, -1),   # 左下
        (1, 1),    # 右下
    ]
    
    for i in range(grid_h):
        for j in range(grid_w):
            node_id = i * grid_w + j
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                # 检查边界
                if 0 <= ni < grid_h and 0 <= nj < grid_w:
                    neighbor_id = ni * grid_w + nj
                    edges.append([node_id, neighbor_id])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()  # [2, E]
    return edge_index


def verify_adjacency(edge_index, num_nodes):
    """
    验证邻接矩阵的正确性
    
    检查:
    1. 无孤立节点（每个节点至少有一条边）
    2. 边索引范围正确
    3. 无自环
    """
    print("\n🔍 验证邻接矩阵:")
    
    # 检查边索引范围
    max_node_id = edge_index.max().item()
    min_node_id = edge_index.min().item()
    
    if min_node_id < 0 or max_node_id >= num_nodes:
        print(f"  ❌ 边索引超出范围: [{min_node_id}, {max_node_id}], 节点数: {num_nodes}")
        return False
    print(f"  ✅ 边索引范围正确: [0, {num_nodes-1}]")
    
    # 检查自环
    self_loops = (edge_index[0] == edge_index[1]).sum().item()
    if self_loops > 0:
        print(f"  ⚠️  包含 {self_loops} 个自环")
    else:
        print(f"  ✅ 无自环")
    
    # 检查孤立节点
    connected_nodes = torch.unique(edge_index.flatten())
    num_connected = connected_nodes.numel()
    
    if num_connected < num_nodes:
        isolated = set(range(num_nodes)) - set(connected_nodes.tolist())
        print(f"  ❌ 发现 {num_nodes - num_connected} 个孤立节点: {isolated}")
        return False
    print(f"  ✅ 无孤立节点 (所有 {num_nodes} 个节点都有连接)")
    
    # 统计每个节点的度数
    node_degrees = torch.zeros(num_nodes, dtype=torch.long)
    for node in edge_index[0]:
        node_degrees[node] += 1
    
    min_degree = node_degrees.min().item()
    max_degree = node_degrees.max().item()
    avg_degree = node_degrees.float().mean().item()
    
    print(f"  📊 节点度数统计:")
    print(f"     最小度数: {min_degree}")
    print(f"     最大度数: {max_degree}")
    print(f"     平均度数: {avg_degree:.2f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='构建图邻接矩阵')
    parser.add_argument('--grid-h', type=int, required=True,
                        help='网格高度')
    parser.add_argument('--grid-w', type=int, required=True,
                        help='网格宽度')
    parser.add_argument('--neighbor-type', type=str, default='8',
                        choices=['4', '8'],
                        help='邻居类型: 4-邻居或8-邻居')
    parser.add_argument('--output-dir', type=str, default='outputs/graph_structures',
                        help='输出目录')
    
    args = parser.parse_args()
    
    num_nodes = args.grid_h * args.grid_w
    
    print("=" * 80)
    print(f"🔗 图边构造 - 邻接矩阵生成")
    print("=" * 80)
    print(f"网格大小:     {args.grid_h} × {args.grid_w}")
    print(f"节点总数:     {num_nodes}")
    print(f"邻居类型:     {args.neighbor_type}-邻居")
    print("=" * 80)
    
    # 构建邻接矩阵
    if args.neighbor_type == '4':
        edge_index = build_grid_adjacency_4neighbor(args.grid_h, args.grid_w)
    else:  # 8-neighbor
        edge_index = build_grid_adjacency_8neighbor(args.grid_h, args.grid_w)
    
    num_edges = edge_index.shape[1]
    print(f"\n✅ 邻接矩阵构建完成:")
    print(f"   边数: {num_edges}")
    print(f"   边索引形状: {list(edge_index.shape)}")
    
    # 验证邻接矩阵
    is_valid = verify_adjacency(edge_index, num_nodes)
    
    if not is_valid:
        print("\n❌ 邻接矩阵验证失败!")
        return
    
    # 保存邻接矩阵
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"adj_grid_{args.grid_h}x{args.grid_w}_{args.neighbor_type}neighbor.pt"
    
    graph_structure = {
        'edge_index': edge_index,
        'num_nodes': num_nodes,
        'grid_h': args.grid_h,
        'grid_w': args.grid_w,
        'neighbor_type': args.neighbor_type,
        'num_edges': num_edges
    }
    
    torch.save(graph_structure, output_file)
    
    file_size_kb = output_file.stat().st_size / 1024
    print(f"\n💾 已保存邻接矩阵:")
    print(f"   文件: {output_file}")
    print(f"   大小: {file_size_kb:.2f} KB")
    
    print("\n" + "=" * 80)
    print("✅ 图边构造完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
