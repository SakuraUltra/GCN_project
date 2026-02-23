#!/usr/bin/env python3
"""
Step 3 扩展: kNN 图边构建（动态邻接）

基于节点特征的相似度构建 kNN 图：
- 使用余弦相似度计算节点间距离
- 为每个节点连接 k 个最相似的邻居
- 支持早期停止梯度（detach特征）
- 模块化设计，方便消融实验

用途：
- 与固定邻接（4/8-neighbor）对比
- 学习自适应的图结构
"""

import os
import sys
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_knn_graph(node_features, k, use_cosine=True, detach_features=True):
    """
    构建 kNN 图
    
    Args:
        node_features: (N, C) 节点特征
        k: 每个节点连接的最近邻数量
        use_cosine: True=余弦相似度, False=欧氏距离
        detach_features: 是否停止梯度（早期停止）
    
    Returns:
        edge_index: (2, E) 边索引
        edge_weight: (E,) 边权重（相似度或距离）
    """
    N = node_features.size(0)
    
    # 早期停止梯度（如果需要）
    if detach_features:
        features = node_features.detach()
    else:
        features = node_features
    
    # 计算相似度矩阵
    if use_cosine:
        # 余弦相似度
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.mm(features_norm, features_norm.t())  # (N, N)
    else:
        # 欧氏距离（负值，因为要找最小距离）
        dist = torch.cdist(features, features, p=2)  # (N, N)
        similarity = -dist
    
    # 对角线设为负无穷（避免选择自己）
    similarity.fill_diagonal_(float('-inf'))
    
    # 为每个节点找 top-k 邻居
    topk_values, topk_indices = similarity.topk(k, dim=1)  # (N, k)
    
    # 构建边索引
    source_nodes = torch.arange(N).unsqueeze(1).expand(-1, k).reshape(-1)  # (N*k,)
    target_nodes = topk_indices.reshape(-1)  # (N*k,)
    
    edge_index = torch.stack([source_nodes, target_nodes], dim=0)  # (2, N*k)
    edge_weight = topk_values.reshape(-1)  # (N*k,)
    
    # 如果使用欧氏距离，转换为正值
    if not use_cosine:
        edge_weight = -edge_weight
    
    return edge_index, edge_weight


def verify_knn_graph(edge_index, edge_weight, num_nodes, k):
    """
    验证 kNN 图的正确性
    """
    print("\n🔍 验证 kNN 图:")
    
    num_edges = edge_index.size(1)
    expected_edges = num_nodes * k
    
    # 检查边数
    if num_edges != expected_edges:
        print(f"  ⚠️  边数不匹配: {num_edges} (期望 {expected_edges})")
    else:
        print(f"  ✅ 边数正确: {num_edges} (每个节点 {k} 条边)")
    
    # 检查自环
    self_loops = (edge_index[0] == edge_index[1]).sum().item()
    if self_loops > 0:
        print(f"  ❌ 包含 {self_loops} 个自环")
        return False
    else:
        print(f"  ✅ 无自环")
    
    # 检查孤立节点
    connected_nodes = torch.unique(edge_index[0])
    num_connected = connected_nodes.numel()
    
    if num_connected < num_nodes:
        print(f"  ❌ 发现 {num_nodes - num_connected} 个孤立节点（出度=0）")
        return False
    else:
        print(f"  ✅ 无孤立节点（所有节点都有出边）")
    
    # 统计入度
    in_degrees = torch.zeros(num_nodes, dtype=torch.long)
    for target in edge_index[1]:
        in_degrees[target] += 1
    
    min_in_degree = in_degrees.min().item()
    max_in_degree = in_degrees.max().item()
    avg_in_degree = in_degrees.float().mean().item()
    
    print(f"  📊 节点入度统计:")
    print(f"     最小入度: {min_in_degree}")
    print(f"     最大入度: {max_in_degree}")
    print(f"     平均入度: {avg_in_degree:.2f}")
    
    # 检查边权重
    if edge_weight is not None:
        min_weight = edge_weight.min().item()
        max_weight = edge_weight.max().item()
        avg_weight = edge_weight.mean().item()
        
        print(f"  📊 边权重统计:")
        print(f"     最小权重: {min_weight:.4f}")
        print(f"     最大权重: {max_weight:.4f}")
        print(f"     平均权重: {avg_weight:.4f}")
    
    return True


def process_graph_nodes_file(nodes_file, k, use_cosine, detach_features, output_dir, batch_size=100):
    """
    为单个节点文件构建 kNN 图（批量处理加速）
    """
    print(f"\n📦 处理: {nodes_file.name}")
    
    # 加载节点数据
    print(f"   加载节点特征...")
    data = torch.load(nodes_file, map_location='cpu')
    num_samples = len(data)
    print(f"   样本数: {num_samples:,}")
    
    # 批量处理
    knn_graphs = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"   批量构建 kNN 图 (batch_size={batch_size})...")
    
    for batch_idx in tqdm(range(num_batches), desc="   处理批次"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = data[start_idx:end_idx]
        
        # 批量处理
        for sample in batch_data:
            nodes = sample['nodes']  # (N, C)
            
            # 构建 kNN 图
            edge_index, edge_weight = build_knn_graph(
                nodes, k, use_cosine, detach_features
            )
            
            # 保存图结构
            knn_graphs.append({
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'num_nodes': nodes.size(0),
                'index': sample['index'],
                'pid': sample['pid'],
                'camid': sample['camid']
            })
    
    # 保存 kNN 图
    output_file = output_dir / nodes_file.name.replace('graph_nodes', 'knn_graph')
    torch.save(knn_graphs, output_file)
    
    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"   ✅ 已保存: {output_file.name} ({file_size_mb:.1f} MB)")
    
    # 验证第一个样本
    print(f"\n   验证第一个样本:")
    sample_graph = knn_graphs[0]
    verify_knn_graph(
        sample_graph['edge_index'],
        sample_graph['edge_weight'],
        sample_graph['num_nodes'],
        k
    )
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='构建 kNN 图边')
    parser.add_argument('--nodes-dir', type=str, required=True,
                        help='节点特征目录 (e.g., outputs/graph_nodes/776_baseline_grid_8x8)')
    parser.add_argument('--k', type=int, default=8,
                        help='每个节点的最近邻数量')
    parser.add_argument('--metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'],
                        help='相似度度量: cosine 或 euclidean')
    parser.add_argument('--detach', action='store_true',
                        help='停止特征梯度（早期停止）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（默认与 nodes-dir 相同）')
    
    args = parser.parse_args()
    
    nodes_dir = Path(args.nodes_dir)
    if not nodes_dir.exists():
        print(f"❌ 节点目录不存在: {nodes_dir}")
        return
    
    # 设置输出目录
    if args.output_dir is None:
        output_dir = nodes_dir.parent / f"{nodes_dir.name}_knn{args.k}"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_cosine = (args.metric == 'cosine')
    
    print("=" * 80)
    print(f"🔗 kNN 图边构建")
    print("=" * 80)
    print(f"节点目录:     {nodes_dir}")
    print(f"输出目录:     {output_dir}")
    print(f"k 值:         {args.k}")
    print(f"相似度度量:   {args.metric}")
    print(f"早期停止:     {args.detach}")
    print("=" * 80)
    
    # 处理所有节点文件
    node_files = list(nodes_dir.glob('*_graph_nodes.pt'))
    
    if not node_files:
        print(f"⚠️  未找到节点文件")
        return
    
    print(f"\n找到 {len(node_files)} 个节点文件")
    
    for node_file in node_files:
        process_graph_nodes_file(
            node_file, args.k, use_cosine, args.detach, output_dir
        )
    
    print("\n" + "=" * 80)
    print("✅ kNN 图边构建完成!")
    print("=" * 80)
    
    # 打印使用说明
    print("\n📝 使用说明:")
    print("   1. kNN 图保存在:", output_dir)
    print("   2. 消融实验时可以选择:")
    print("      - 仅使用固定邻接（4/8-neighbor）")
    print("      - 仅使用 kNN 图")
    print("      - 同时使用两者（混合邻接）")
    print("   3. 训练时可通过配置文件控制是否加载 kNN 边")


if __name__ == '__main__':
    main()
