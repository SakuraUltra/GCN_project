# VIT-25 Native Dimension 实验结果

## 📊 最终性能对比

### ✅ 已完成实验

| 实验 | mAP | Rank-1 | Rank-5 | Rank-10 | 参数量 | 特征维度 | Job ID |
|------|-----|--------|--------|---------|--------|----------|--------|
| **ViT-Base+GCN Native 768** | **72.39%** | **93.74%** | 97.44% | 98.69% | 89.20M | 768 | 31379904 |
| **ViT-Base+GCN Native 512** | **72.19%** | **94.76%** | **97.62%** | **98.57%** | 87.80M | 512 | 31380317 |

### 基线对比

| 模型 | mAP | Rank-1 | 参数量 | 维度 |
|------|-----|--------|--------|------|
| ViT-Base+GAT (projected) | 72.04% | 93.33% | 111.12M | 2048 |
| ViT-Base+GCN (projected) | 71.77% | 93.98% | 111.12M | 2048 |
| ResNet50+GCN | 75.24% | 93.68% | 39.3M | 2048 |

## 🎯 关键发现

### 1. Native 768-dim vs 512-dim
- **mAP**: 72.39% vs 72.19% (-0.20%)
- **Rank-1**: 93.74% vs 94.76% (+1.02%) ⭐
- **参数**: 89.20M vs 87.80M (-1.57%)

**结论**: 
- 512-dim 实现了更高的 Rank-1 准确率（94.76%，所有实验最佳）
- 参数更少（节省 1.4M）
- mAP 仅略微下降 0.2%
- **512-dim 是更优选择**：更高效、Rank-1 更好

### 2. Native vs Projected 对比
- Native 768: **72.39% mAP** vs Projected 2048: 71.77% (+0.62%)
- Native 512: **72.19% mAP** vs Projected 2048: 71.77% (+0.42%)
- 参数节省: **~21-23M** (19-21%)

**结论**: Native dimension 优于 projected，无信息损失

### 3. VIT-25 vs 基线最佳
- ViT-Base (Native 512): 72.19% mAP, **94.76% Rank-1**
- ResNet50+GCN: **75.24% mAP**, 93.68% Rank-1

**差距**: mAP -3.05%, Rank-1 +1.08%

## 📁 结果保存位置

### ⚠️ 重要说明
由于训练脚本 bug（已修复），两个实验都保存到了 `outputs/bot_gcn/`，768-dim 结果被 512-dim 覆盖。

**当前状态**:
- ✅ `outputs/bot_vitbase_native512_gcn_776/`: 512-dim 完整结果
- ⚠️ `outputs/bot_vitbase_native_gcn_776/`: 空目录（768-dim 结果丢失）
- ✅ 日志文件保存完整：
  - `logs/vitbase_native_gcn_h100_31379904.err` (768-dim)
  - `logs/vitbase_native_gcn_h100_31380317.err` (512-dim)

**Bug 修复**: 
- 修改 `scripts/training/train_bot_gcn.py` line 467
- 从 `config.get('OUTPUT_DIR')` → `config.get('OUTPUT', {}).get('DIR')`
- 现在可以正确读取配置文件中的 OUTPUT.DIR

### 512-dim 模型文件
```
outputs/bot_vitbase_native512_gcn_776/
├── best_model.pth (72.19% mAP)
├── checkpoint_epoch_20.pth
├── checkpoint_epoch_40.pth
├── checkpoint_epoch_60.pth
├── checkpoint_epoch_80.pth
├── checkpoint_epoch_100.pth
├── checkpoint_epoch_120.pth
└── training.log
```

## 🚀 后续建议

1. **使用 512-dim 作为标准配置**
   - 更高的 Rank-1 准确率
   - 更少的参数量
   - 训练和推理更快

2. **可选：重新训练 768-dim**
   - 如需保留完整权重文件
   - 使用修复后的脚本确保保存到正确目录

3. **性能优化方向**
   - 探索 384-dim（ViT-Small native）
   - 尝试不同的 GCN hidden dimensions
   - 调整学习率策略

---

**实验完成时间**: 2026-03-04  
**VIT-25 实现**: Native Dimension Support ✅
