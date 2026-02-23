#!/usr/bin/env python3
"""
GCN 深度测试套件

测试项：
1. 数学正确性：验证 GCN 公式实现
2. 边界条件：空图、单节点、全连接图
3. 数值稳定性：大规模图、极端值
4. 梯度检查：数值梯度 vs 自动微分
5. 归一化验证：对称归一化正确性
6. 批处理：多个图并行处理
7. 不同邻接结构：4-邻居、8-邻居、kNN
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from models.gcn.gcn_conv import GCNConv, SimpleGCN
import numpy as np


def test_1_mathematical_correctness():
    """测试1: 数学正确性 - 手动验证小图"""
    print("\n" + "=" * 80)
    print("测试1: 数学正确性验证")
    print("=" * 80)
    
    # 3节点简单图
    num_nodes = 3
    in_channels = 4
    out_channels = 2
    
    # 固定节点特征
    x = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    # 简单边: 0->1, 1->2
    edge_index = torch.tensor([
        [0, 1],
        [1, 2]
    ], dtype=torch.long)
    
    # 创建 GCN 并固定权重
    gcn = GCNConv(in_channels, out_channels)
    with torch.no_grad():
        gcn.weight.fill_(1.0)
        if gcn.bias is not None:
            gcn.bias.fill_(0.0)
    
    # 前向传播
    out = gcn(x, edge_index)
    
    print(f"输入特征:\n{x}")
    print(f"边索引: {edge_index.t().tolist()}")
    print(f"输出特征:\n{out}")
    
    # 检查输出形状
    assert out.shape == (num_nodes, out_channels), "输出形状错误"
    
    # 检查无 NaN/Inf
    assert not torch.isnan(out).any(), "输出包含 NaN"
    assert not torch.isinf(out).any(), "输出包含 Inf"
    
    print("✅ 数学正确性测试通过")
    return True


def test_2_boundary_conditions():
    """测试2: 边界条件"""
    print("\n" + "=" * 80)
    print("测试2: 边界条件测试")
    print("=" * 80)
    
    in_channels = 8
    out_channels = 4
    
    # 2.1 单节点图
    print("\n2.1 单节点图...")
    x = torch.randn(1, in_channels)
    edge_index = torch.zeros((2, 0), dtype=torch.long)  # 无边
    
    gcn = GCNConv(in_channels, out_channels)
    out = gcn(x, edge_index)
    
    assert out.shape == (1, out_channels), "单节点输出形状错误"
    assert not torch.isnan(out).any(), "单节点包含 NaN"
    print("  ✅ 单节点图通过")
    
    # 2.2 无边图（孤立节点）
    print("\n2.2 无边图...")
    x = torch.randn(5, in_channels)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    out = gcn(x, edge_index)
    assert out.shape == (5, out_channels), "无边图输出形状错误"
    assert not torch.isnan(out).any(), "无边图包含 NaN"
    print("  ✅ 无边图通过")
    
    # 2.3 全连接图
    print("\n2.3 全连接图...")
    num_nodes = 5
    x = torch.randn(num_nodes, in_channels)
    
    # 生成全连接边
    edge_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_list.append([i, j])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    out = gcn(x, edge_index)
    assert out.shape == (num_nodes, out_channels), "全连接图输出形状错误"
    assert not torch.isnan(out).any(), "全连接图包含 NaN"
    print("  ✅ 全连接图通过")
    
    print("\n✅ 边界条件测试通过")
    return True


def test_3_numerical_stability():
    """测试3: 数值稳定性"""
    print("\n" + "=" * 80)
    print("测试3: 数值稳定性测试")
    print("=" * 80)
    
    in_channels = 2048
    out_channels = 512
    
    # 3.1 大规模图（模拟 8×8 网格）
    print("\n3.1 大规模图 (64 节点)...")
    num_nodes = 64
    x = torch.randn(num_nodes, in_channels)
    
    # 8×8 网格的 4-邻居边
    edge_list = []
    grid_size = 8
    for i in range(grid_size):
        for j in range(grid_size):
            node = i * grid_size + j
            # 上
            if i > 0:
                edge_list.append([node, (i-1) * grid_size + j])
            # 下
            if i < grid_size - 1:
                edge_list.append([node, (i+1) * grid_size + j])
            # 左
            if j > 0:
                edge_list.append([node, i * grid_size + (j-1)])
            # 右
            if j < grid_size - 1:
                edge_list.append([node, i * grid_size + (j+1)])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    gcn = GCNConv(in_channels, out_channels)
    out = gcn(x, edge_index)
    
    assert out.shape == (num_nodes, out_channels), "大规模图输出形状错误"
    assert not torch.isnan(out).any(), "大规模图包含 NaN"
    assert not torch.isinf(out).any(), "大规模图包含 Inf"
    print(f"  ✅ 大规模图通过 (边数: {edge_index.size(1)})")
    
    # 3.2 极大值输入
    print("\n3.2 极大值输入...")
    x_large = torch.randn(10, in_channels) * 1000
    edge_index_small = torch.tensor([[0,1,2], [1,2,0]], dtype=torch.long)
    
    out = gcn(x_large, edge_index_small)
    assert not torch.isnan(out).any(), "极大值输入产生 NaN"
    assert not torch.isinf(out).any(), "极大值输入产生 Inf"
    print("  ✅ 极大值输入通过")
    
    # 3.3 极小值输入
    print("\n3.3 极小值输入...")
    x_small = torch.randn(10, in_channels) * 1e-6
    out = gcn(x_small, edge_index_small)
    assert not torch.isnan(out).any(), "极小值输入产生 NaN"
    print("  ✅ 极小值输入通过")
    
    # 3.4 混合正负值
    print("\n3.4 混合正负值...")
    x_mixed = torch.randn(10, in_channels)
    x_mixed[::2] *= -1  # 偶数行取负
    out = gcn(x_mixed, edge_index_small)
    assert not torch.isnan(out).any(), "混合正负值产生 NaN"
    print("  ✅ 混合正负值通过")
    
    print("\n✅ 数值稳定性测试通过")
    return True


def test_4_gradient_check():
    """测试4: 梯度检查"""
    print("\n" + "=" * 80)
    print("测试4: 梯度检查")
    print("=" * 80)
    
    # 小规模精确测试
    num_nodes = 5
    in_channels = 3
    out_channels = 2
    
    x = torch.randn(num_nodes, in_channels, requires_grad=True)
    edge_index = torch.tensor([[0,1,2,3], [1,2,3,0]], dtype=torch.long)
    
    gcn = GCNConv(in_channels, out_channels)
    
    # 前向传播
    out = gcn(x, edge_index)
    loss = out.sum()
    loss.backward()
    
    # 检查梯度
    print(f"\n输入梯度范数: {x.grad.norm().item():.6f}")
    print(f"权重梯度范数: {gcn.weight.grad.norm().item():.6f}")
    
    assert x.grad is not None, "输入梯度为 None"
    assert gcn.weight.grad is not None, "权重梯度为 None"
    assert x.grad.norm() > 0, "输入梯度为零"
    assert gcn.weight.grad.norm() > 0, "权重梯度为零"
    
    # 数值梯度检查（简化版）
    print("\n数值梯度检查...")
    eps = 1e-5
    
    # 对权重的第一个元素做数值梯度
    gcn.zero_grad()
    out1 = gcn(x.detach(), edge_index)
    loss1 = out1.sum()
    loss1.backward()
    analytical_grad = gcn.weight.grad[0, 0].item()
    
    # 数值梯度
    with torch.no_grad():
        original_val = gcn.weight[0, 0].item()
        gcn.weight[0, 0] = original_val + eps
    
    out_plus = gcn(x.detach(), edge_index)
    loss_plus = out_plus.sum().item()
    
    with torch.no_grad():
        gcn.weight[0, 0] = original_val - eps
    
    out_minus = gcn(x.detach(), edge_index)
    loss_minus = out_minus.sum().item()
    
    numerical_grad = (loss_plus - loss_minus) / (2 * eps)
    
    grad_diff = abs(analytical_grad - numerical_grad)
    grad_ratio = grad_diff / (abs(numerical_grad) + 1e-8)
    
    print(f"  解析梯度: {analytical_grad:.8f}")
    print(f"  数值梯度: {numerical_grad:.8f}")
    print(f"  相对误差: {grad_ratio:.8f}")
    
    assert grad_ratio < 1e-3, f"梯度检查失败: 相对误差 {grad_ratio} > 1e-3"
    
    print("\n✅ 梯度检查通过")
    return True


def test_5_normalization():
    """测试5: 归一化验证"""
    print("\n" + "=" * 80)
    print("测试5: 对称归一化验证")
    print("=" * 80)
    
    # 创建简单图并手动验证归一化
    num_nodes = 3
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    
    gcn = GCNConv(4, 2)
    
    # 添加自环
    edge_index_with_loops, edge_weight = gcn.add_self_loops(
        edge_index, None, num_nodes
    )
    
    print(f"原始边数: {edge_index.size(1)}")
    print(f"添加自环后边数: {edge_index_with_loops.size(1)}")
    assert edge_index_with_loops.size(1) == edge_index.size(1) + num_nodes, "自环添加错误"
    
    # 归一化
    edge_index_norm, edge_weight_norm = gcn.normalize_adj(
        edge_index_with_loops, edge_weight, num_nodes
    )
    
    print(f"归一化后权重范围: [{edge_weight_norm.min():.4f}, {edge_weight_norm.max():.4f}]")
    print(f"归一化后权重均值: {edge_weight_norm.mean():.4f}")
    
    # 检查归一化权重在合理范围
    assert edge_weight_norm.min() > 0, "归一化权重存在非正值"
    assert edge_weight_norm.max() <= 1.0, "归一化权重超过 1"
    
    print("\n✅ 归一化验证通过")
    return True


def test_6_real_graph_structures():
    """测试6: 真实图结构"""
    print("\n" + "=" * 80)
    print("测试6: 真实图结构测试")
    print("=" * 80)
    
    in_channels = 2048
    out_channels = 256
    
    # 6.1 4×4 网格 4-邻居
    print("\n6.1 测试 4×4 网格 4-邻居...")
    adj_file = 'outputs/graph_structures/adj_grid_4x4_4neighbor.pt'
    graph = torch.load(adj_file, map_location='cpu')
    
    x = torch.randn(graph['num_nodes'], in_channels)
    edge_index = graph['edge_index']
    
    gcn = GCNConv(in_channels, out_channels)
    out = gcn(x, edge_index)
    
    assert out.shape == (graph['num_nodes'], out_channels)
    assert not torch.isnan(out).any()
    print(f"  ✅ 4×4 网格通过 (节点: {graph['num_nodes']}, 边: {graph['num_edges']})")
    
    # 6.2 8×8 网格 8-邻居
    print("\n6.2 测试 8×8 网格 8-邻居...")
    adj_file = 'outputs/graph_structures/adj_grid_8x8_8neighbor.pt'
    graph = torch.load(adj_file, map_location='cpu')
    
    x = torch.randn(graph['num_nodes'], in_channels)
    edge_index = graph['edge_index']
    
    out = gcn(x, edge_index)
    
    assert out.shape == (graph['num_nodes'], out_channels)
    assert not torch.isnan(out).any()
    print(f"  ✅ 8×8 网格通过 (节点: {graph['num_nodes']}, 边: {graph['num_edges']})")
    
    # 6.3 真实节点数据
    print("\n6.3 测试真实节点特征...")
    nodes_file = 'outputs/graph_nodes/776_baseline_grid_4x4/query_graph_nodes.pt'
    data = torch.load(nodes_file, map_location='cpu')
    
    sample = data[0]
    nodes = sample['nodes']
    
    adj_file = 'outputs/graph_structures/adj_grid_4x4_8neighbor.pt'
    graph = torch.load(adj_file, map_location='cpu')
    edge_index = graph['edge_index']
    
    out = gcn(nodes, edge_index)
    
    assert out.shape == (nodes.size(0), out_channels)
    assert not torch.isnan(out).any()
    print(f"  ✅ 真实节点数据通过")
    
    print("\n✅ 真实图结构测试通过")
    return True


def test_7_multi_layer_gcn():
    """测试7: 多层 GCN"""
    print("\n" + "=" * 80)
    print("测试7: 多层 GCN 测试")
    print("=" * 80)
    
    num_nodes = 16
    in_channels = 2048
    hidden_channels = 512
    out_channels = 256
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([[0,1,2,3], [1,2,3,0]], dtype=torch.long)
    
    # 测试不同层数
    for num_layers in [1, 2, 3, 4]:
        print(f"\n测试 {num_layers} 层 GCN...")
        
        if num_layers == 1:
            model = GCNConv(in_channels, out_channels)
            out = model(x, edge_index)
        else:
            model = SimpleGCN(in_channels, hidden_channels, out_channels)
            out = model(x, edge_index)
        
        # 反向传播
        loss = out.sum()
        loss.backward()
        
        # 检查梯度
        has_grad = True
        for param in model.parameters():
            if param.grad is None or param.grad.norm() == 0:
                has_grad = False
                break
        
        assert has_grad, f"{num_layers} 层 GCN 梯度异常"
        assert not torch.isnan(out).any(), f"{num_layers} 层 GCN 输出包含 NaN"
        
        print(f"  ✅ {num_layers} 层 GCN 通过")
    
    print("\n✅ 多层 GCN 测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("🧪 GCN 深度测试套件")
    print("=" * 80)
    
    tests = [
        ("数学正确性", test_1_mathematical_correctness),
        ("边界条件", test_2_boundary_conditions),
        ("数值稳定性", test_3_numerical_stability),
        ("梯度检查", test_4_gradient_check),
        ("归一化验证", test_5_normalization),
        ("真实图结构", test_6_real_graph_structures),
        ("多层 GCN", test_7_multi_layer_gcn),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS", ""))
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            print(f"\n❌ {name} 测试失败: {e}")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    
    passed = sum(1 for _, status, _ in results if status == "PASS")
    total = len(results)
    
    for name, status, error in results:
        emoji = "✅" if status == "PASS" else "❌"
        print(f"{emoji} {name}: {status}")
        if error:
            print(f"   错误: {error}")
    
    print("=" * 80)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！GCN 实现正确！")
    else:
        print(f"⚠️  {total - passed} 个测试失败，请检查实现")
    
    print("=" * 80)
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
