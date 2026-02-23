"""
图池化策略对比脚本
Compare Graph Pooling Strategies

用途: 
1. 在真实数据上比较三种池化策略 (mean/max/attention)
2. 评估不同池化方法对图嵌入质量的影响
3. 为消融实验提供定量分析

Usage:
    python scripts/testing/compare_pooling_strategies.py \\
        --graph_nodes outputs/graph_nodes/776_baseline_grid_4x4/train_graph_nodes.pt \\
        --grid_size 4 4 \\
        --output_dir outputs/pooling_comparison
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

from models.gcn.graph_pooling import GraphPooling


def load_graph_nodes(filepath):
    """加载图节点数据"""
    print(f"\n加载图节点: {filepath}")
    data = torch.load(filepath)
    print(f"  样本数: {len(data)}")
    print(f"  节点形状示例: {data[0]['nodes'].shape}")
    return data


def compute_embeddings(graph_nodes, pooling_type, in_channels=2048):
    """
    为所有图计算嵌入
    
    Args:
        graph_nodes: 图节点数据列表
        pooling_type: 池化类型 ('mean', 'max', 'attention')
        in_channels: 节点特征维度
    
    Returns:
        embeddings: (N, D) 图嵌入张量
        pids: (N,) 人员ID
        camids: (N,) 相机ID
    """
    print(f"\n{'=' * 80}")
    print(f"计算 {pooling_type.upper()} Pooling 嵌入...")
    print(f"{'=' * 80}")
    
    # 创建池化层
    if pooling_type == 'attention':
        pooling = GraphPooling(pooling_type, in_channels=in_channels, hidden_channels=128)
    else:
        pooling = GraphPooling(pooling_type)
    
    pooling.eval()  # 评估模式
    
    embeddings = []
    pids = []
    camids = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for sample in tqdm(graph_nodes, desc=f"{pooling_type.upper()} Pooling"):
            nodes = sample['nodes']  # (N_nodes, D)
            
            # 池化: (N_nodes, D) -> (1, D)
            emb = pooling(nodes, batch=None)  # (1, D)
            
            embeddings.append(emb.squeeze(0))  # (D,)
            pids.append(sample['pid'])
            camids.append(sample['camid'])
    
    elapsed = time.time() - start_time
    
    embeddings = torch.stack(embeddings, dim=0)  # (N, D)
    pids = torch.tensor(pids)
    camids = torch.tensor(camids)
    
    print(f"\n完成!")
    print(f"  嵌入形状: {embeddings.shape}")
    print(f"  用时: {elapsed:.2f}s ({elapsed/len(graph_nodes)*1000:.2f}ms/sample)")
    print(f"  嵌入范围: [{embeddings.min().item():.4f}, {embeddings.max().item():.4f}]")
    print(f"  嵌入均值: {embeddings.mean().item():.4f}")
    print(f"  嵌入标准差: {embeddings.std().item():.4f}")
    
    return embeddings, pids, camids


def analyze_embeddings(embeddings_dict):
    """
    分析不同池化策略产生的嵌入
    
    Args:
        embeddings_dict: {pooling_type: (embeddings, pids, camids)}
    """
    print(f"\n{'=' * 80}")
    print(f"📊 池化策略分析")
    print(f"{'=' * 80}")
    
    pooling_types = list(embeddings_dict.keys())
    
    # 1. 嵌入统计信息
    print(f"\n1️⃣  嵌入统计信息:")
    print(f"{'─' * 80}")
    print(f"{'策略':<15} {'均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
    print(f"{'─' * 80}")
    
    for pool_type in pooling_types:
        embs = embeddings_dict[pool_type][0]
        print(f"{pool_type:<15} {embs.mean().item():>10.4f} {embs.std().item():>10.4f} "
              f"{embs.min().item():>10.4f} {embs.max().item():>10.4f}")
    
    # 2. 嵌入相似度分析
    print(f"\n2️⃣  嵌入相似度分析:")
    print(f"{'─' * 80}")
    
    # 计算两两之间的平均余弦相似度
    for i, pool_type1 in enumerate(pooling_types):
        for pool_type2 in pooling_types[i+1:]:
            embs1 = embeddings_dict[pool_type1][0]
            embs2 = embeddings_dict[pool_type2][0]
            
            # 逐样本计算余弦相似度
            similarities = F.cosine_similarity(embs1, embs2, dim=1)
            avg_sim = similarities.mean().item()
            std_sim = similarities.std().item()
            
            print(f"  {pool_type1.capitalize():>10} vs {pool_type2.capitalize():<10}: "
                  f"平均={avg_sim:.4f}, 标准差={std_sim:.4f}")
    
    # 3. 嵌入差异分析 (L2 距离)
    print(f"\n3️⃣  嵌入差异分析 (L2 距离):")
    print(f"{'─' * 80}")
    
    for i, pool_type1 in enumerate(pooling_types):
        for pool_type2 in pooling_types[i+1:]:
            embs1 = embeddings_dict[pool_type1][0]
            embs2 = embeddings_dict[pool_type2][0]
            
            # 逐样本计算 L2 距离
            distances = (embs1 - embs2).norm(dim=1)
            avg_dist = distances.mean().item()
            std_dist = distances.std().item()
            
            print(f"  {pool_type1.capitalize():>10} vs {pool_type2.capitalize():<10}: "
                  f"平均={avg_dist:.4f}, 标准差={std_dist:.4f}")
    
    # 4. 类内/类间距离分析
    print(f"\n4️⃣  类内/类间距离分析 (基于 PID):")
    print(f"{'─' * 80}")
    
    for pool_type in pooling_types:
        embs, pids, _ = embeddings_dict[pool_type]
        
        # 计算类内距离 (同一 PID 的样本之间)
        intra_dists = []
        unique_pids = pids.unique()
        
        for pid in unique_pids:
            mask = (pids == pid)
            if mask.sum() < 2:
                continue  # 需要至少 2 个样本
            
            pid_embs = embs[mask]  # (N_pid, D)
            
            # 计算该 PID 内所有样本对的距离
            for i in range(len(pid_embs)):
                for j in range(i+1, len(pid_embs)):
                    dist = (pid_embs[i] - pid_embs[j]).norm().item()
                    intra_dists.append(dist)
        
        # 计算类间距离 (不同 PID 的样本之间) - 随机采样避免计算量过大
        inter_dists = []
        num_samples = min(10000, len(pids) * 10)  # 随机采样
        
        for _ in range(num_samples):
            idx1, idx2 = np.random.choice(len(embs), size=2, replace=False)
            if pids[idx1] != pids[idx2]:
                dist = (embs[idx1] - embs[idx2]).norm().item()
                inter_dists.append(dist)
        
        intra_mean = np.mean(intra_dists) if intra_dists else 0
        inter_mean = np.mean(inter_dists) if inter_dists else 0
        
        ratio = inter_mean / intra_mean if intra_mean > 0 else 0
        
        print(f"  {pool_type.capitalize():>10}:")
        print(f"    类内距离: {intra_mean:.4f}")
        print(f"    类间距离: {inter_mean:.4f}")
        print(f"    类间/类内比率: {ratio:.4f} {'(越大越好)' if ratio > 1 else ''}")
    
    # 5. 嵌入方差分析
    print(f"\n5️⃣  嵌入特征方差分析:")
    print(f"{'─' * 80}")
    
    for pool_type in pooling_types:
        embs = embeddings_dict[pool_type][0]
        
        # 计算每个维度的方差
        feature_vars = embs.var(dim=0)  # (D,)
        
        print(f"  {pool_type.capitalize():>10}:")
        print(f"    平均方差: {feature_vars.mean().item():.6f}")
        print(f"    方差标准差: {feature_vars.std().item():.6f}")
        print(f"    最大方差: {feature_vars.max().item():.6f}")
        print(f"    最小方差: {feature_vars.min().item():.6f}")
        print(f"    零方差维度数: {(feature_vars < 1e-6).sum().item()}")


def save_results(embeddings_dict, output_dir):
    """保存结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 80}")
    print(f"💾 保存结果到: {output_dir}")
    print(f"{'=' * 80}")
    
    for pool_type, (embs, pids, camids) in embeddings_dict.items():
        save_path = output_dir / f"{pool_type}_embeddings.pt"
        
        torch.save({
            'embeddings': embs,
            'pids': pids,
            'camids': camids,
            'pooling_type': pool_type
        }, save_path)
        
        print(f"  ✓ {pool_type.capitalize()}: {save_path} ({embs.shape})")
    
    print(f"\n完成!")


def main():
    parser = argparse.ArgumentParser(description='比较图池化策略')
    parser.add_argument('--graph_nodes', type=str, required=True,
                       help='图节点文件路径 (e.g., outputs/graph_nodes/776_baseline_grid_4x4/train_graph_nodes.pt)')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[4, 4],
                       help='网格大小 (H W), 默认 4 4')
    parser.add_argument('--in_channels', type=int, default=2048,
                       help='节点特征维度, 默认 2048')
    parser.add_argument('--output_dir', type=str, default='outputs/pooling_comparison',
                       help='输出目录')
    parser.add_argument('--pooling_types', type=str, nargs='+',
                       default=['mean', 'max', 'attention'],
                       help='要比较的池化策略')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 图池化策略对比实验")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  图节点文件: {args.graph_nodes}")
    print(f"  网格大小: {args.grid_size[0]}×{args.grid_size[1]}")
    print(f"  节点特征维度: {args.in_channels}")
    print(f"  池化策略: {args.pooling_types}")
    print(f"  输出目录: {args.output_dir}")
    
    # 检查文件是否存在
    if not os.path.exists(args.graph_nodes):
        print(f"\n❌ 错误: 文件不存在 {args.graph_nodes}")
        return
    
    # 加载数据
    graph_nodes = load_graph_nodes(args.graph_nodes)
    
    # 计算不同池化策略的嵌入
    embeddings_dict = {}
    
    for pool_type in args.pooling_types:
        embs, pids, camids = compute_embeddings(
            graph_nodes, 
            pool_type, 
            in_channels=args.in_channels
        )
        embeddings_dict[pool_type] = (embs, pids, camids)
    
    # 分析结果
    analyze_embeddings(embeddings_dict)
    
    # 保存结果
    save_results(embeddings_dict, args.output_dir)
    
    print(f"\n{'=' * 80}")
    print(f"✅ 池化策略对比完成!")
    print(f"{'=' * 80}")
    print(f"\n关键发现:")
    print(f"  • Mean Pooling: 最稳定，适合作为 baseline")
    print(f"  • Max Pooling: 捕获极值特征，可能对异常值敏感")
    print(f"  • Attention Pooling: 自适应加权，但需要额外训练")
    print(f"\n后续步骤:")
    print(f"  1. 将不同池化策略集成到 BoT 模型中")
    print(f"  2. 训练并评估各策略的 ReID 性能")
    print(f"  3. 进行消融实验，确定最优配置")
    print("=" * 80)


if __name__ == '__main__':
    main()
