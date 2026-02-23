#!/usr/bin/env python3
"""
测试 GCN 在真实图节点数据上的表现
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.gcn.gcn_conv import GCNConv, SimpleGCN


def test_gcn_on_real_data():
    """在真实的图节点数据上测试 GCN"""
    print("=" * 80)
    print("🧪 测试 GCN 在真实图节点数据上的表现")
    print("=" * 80)
    
    # 加载一个样本的图节点
    print("\n📦 加载图节点数据...")
    nodes_file = 'outputs/graph_nodes/776_baseline_grid_4x4/query_graph_nodes.pt'
    data = torch.load(nodes_file, map_location='cpu')
    
    sample = data[0]
    nodes = sample['nodes']  # (N, C)
    
    print(f"  节点特征形状: {nodes.shape}")
    print(f"  节点数: {nodes.size(0)}")
    print(f"  特征维度: {nodes.size(1)}")
    
    # 加载固定邻接矩阵
    print("\n📦 加载固定邻接矩阵...")
    adj_file = 'outputs/graph_structures/adj_grid_4x4_4neighbor.pt'
    graph_structure = torch.load(adj_file, map_location='cpu')
    
    edge_index = graph_structure['edge_index']
    num_nodes = graph_structure['num_nodes']
    
    print(f"  边索引形状: {edge_index.shape}")
    print(f"  边数: {edge_index.size(1)}")
    print(f"  节点数: {num_nodes}")
    
    # 验证维度匹配
    assert nodes.size(0) == num_nodes, f"节点数不匹配: {nodes.size(0)} vs {num_nodes}"
    
    # 创建 GCN 层
    print("\n🔧 创建 GCN 层...")
    in_channels = nodes.size(1)
    out_channels = 256
    
    gcn = GCNConv(in_channels, out_channels)
    
    # 前向传播
    print(f"\n🚀 前向传播...")
    out = gcn(nodes, edge_index)
    
    print(f"  输出形状: {out.shape}")
    print(f"  输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"  输出均值: {out.mean().item():.4f}")
    print(f"  输出标准差: {out.std().item():.4f}")
    
    # 检查梯度
    print(f"\n🔍 检查梯度流动...")
    loss = out.sum()
    loss.backward()
    
    grad_norm = gcn.weight.grad.norm().item()
    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    
    print(f"  权重梯度范数: {grad_norm:.6f}")
    print(f"  包含 NaN: {has_nan}")
    print(f"  包含 Inf: {has_inf}")
    
    # 测试多层 GCN
    print(f"\n🔧 测试多层 GCN...")
    model = SimpleGCN(in_channels, 512, out_channels, dropout=0.5)
    model.train()
    
    out = model(nodes, edge_index)
    print(f"  多层 GCN 输出形状: {out.shape}")
    
    loss = out.sum()
    loss.backward()
    
    grad_norm1 = model.conv1.weight.grad.norm().item()
    grad_norm2 = model.conv2.weight.grad.norm().item()
    
    print(f"  第1层梯度范数: {grad_norm1:.6f}")
    print(f"  第2层梯度范数: {grad_norm2:.6f}")
    
    # 验证结果
    print("\n" + "=" * 80)
    if grad_norm > 0 and not has_nan and not has_inf and grad_norm1 > 0 and grad_norm2 > 0:
        print("✅ 测试通过!")
        print("  ✓ 前向传播正常")
        print("  ✓ 梯度流动正常")
        print("  ✓ 无 NaN/Inf 值")
        print("  ✓ 多层 GCN 工作正常")
    else:
        print("❌ 测试失败!")
        if grad_norm == 0:
            print("  ✗ 单层梯度为零")
        if has_nan:
            print("  ✗ 包含 NaN 值")
        if has_inf:
            print("  ✗ 包含 Inf 值")
        if grad_norm1 == 0 or grad_norm2 == 0:
            print("  ✗ 多层梯度为零")
    
    print("=" * 80)


if __name__ == '__main__':
    test_gcn_on_real_data()
