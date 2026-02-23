# 项目状态总结

## 已完成

### 1. 图卷积网络 (GCN) 实现
- ✅ GCN卷积层 (`models/gcn/gcn_conv.py`)
- ✅ 图池化策略 (`models/gcn/graph_pooling.py`)
  - Mean Pooling
  - Max Pooling  
  - Attention Pooling

### 2. 嵌入融合策略
- ✅ Concat + Projection Fusion
- ✅ Gated Fusion
- ✅ 统一接口 (`models/fusion/embedding_fusion.py`)

### 3. BoT-GCN 模型
- ✅ 完整架构 (`models/bot_baseline/bot_gcn_model.py`)
- ✅ ResNet50-IBN backbone
- ✅ Grid pooling (4×4 / 8×8)
- ✅ GCN 处理
- ✅ 图池化
- ✅ 嵌入融合
- ✅ **修复**: 测试时使用 `final_emb` 而非 `bn_feat`,与训练一致

### 4. 训练流程
- ✅ 数据加载 (VeRi-776)
- ✅ PK Sampler
- ✅ BoT Loss (ID + Triplet)
- ✅ Warmup + Cosine Scheduler
- ✅ 训练脚本 (`scripts/training/train_bot_gcn.py`)
- ✅ SLURM 任务提交

### 5. 训练结果
- ✅ 完成 120 epochs 训练
- 模型参数: 39.3M
- 配置: Grid 4×4, Mean Pooling, Concat Fusion
- 最佳 mAP: 18.06% (Epoch 120)
- **注**: VeRi-776 为 cross-id 任务,训练集576个ID,测试集200个不同ID

## 当前状态

### 训练模型
- 位置: `outputs/bot_gcn_776_baseline/best_model.pth`
- 配置: `configs/gcn_transformer_configs/bot_gcn_776.yaml`
- 日志: `logs/bot_gcn_776_31123450.{out,err}`

### 代码优化
- ✅ 清理调试日志
- ✅ 删除临时文件  
- ✅ 简化训练脚本注释
- ✅ 精简 GCN 代码
- ✅ 删除中间特征文件 (节省 ~210GB)

### 性能问题诊断
问题: mAP 18% 低于预期
分析:
1. VeRi-776 是 zero-shot/cross-id 任务,测试集ID完全不在训练集中
2. 训练时 triplet loss 使用 `final_emb`,测试时错误使用 `bn_feat`
3. 已修复特征不一致问题,需重新训练验证

## 下一步

### 优先级 1: 验证修复效果
- [ ] 用修复后的代码重新训练 BoT-GCN
- [ ] 评估新模型性能,验证 mAP 提升

### 优先级 2: 消融实验
- [ ] Grid size: 4×4 vs 8×8
- [ ] Pooling: mean vs max vs attention  
- [ ] Fusion: concat vs gated
- [ ] 共 9 个配置组合

### 优先级 3: 与 Baseline 对比
- [ ] 训练纯 BoT baseline (无 GCN)
- [ ] 对比 BoT vs BoT-GCN
- [ ] 分析 GCN 带来的提升

## 文件结构

```
GCN_project/
├── models/
│   ├── gcn/              # GCN实现
│   ├── fusion/           # 融合策略
│   └── bot_baseline/     # BoT模型
├── scripts/
│   ├── training/         # 训练脚本
│   ├── testing/          # 测试脚本
│   └── slurm_jobs/       # SLURM任务
├── configs/              # 配置文件
├── outputs/              # 输出
│   ├── bot_gcn_776_baseline/  # 已训练模型
│   └── graph_structures/      # 图结构
└── logs/                 # 训练日志
```

## 关键配置

### VeRi-776 数据集
- 训练: 37,778 images, 576 identities
- Query: 1,678 images, 200 identities  
- Gallery: 11,579 images, 200 identities
- **重要**: 训练集和测试集ID无重叠 (cross-id任务)

### 模型配置
- Backbone: ResNet50-IBN-a
- Grid: 4×4 = 16 nodes
- GCN: 1 layer, 512 hidden dim
- Pooling: Mean
- Fusion: Concat (4096 → 2048)
- BNNeck: BatchNorm1d

### 训练配置  
- Optimizer: AdamW (lr=3.5e-4, wd=5e-4)
- Scheduler: Warmup(10 ep) + Cosine
- Batch: 64 (P=16, K=4)
- Epochs: 120
- AMP: Disabled (sparse ops不支持FP16)

## 最后更新
2026-02-22 15:00 GMT
