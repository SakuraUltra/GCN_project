# 🎯 BoT-GCN 训练策略优化方案

**日期**: 2026-02-22  
**状态**: 调试完成，准备重新训练

---

## 📋 背景分析

### 当前问题
- ✅ **Bug已修复**: 训练使用 `final_emb`，测试之前使用 `bn_feat` → 已统一为 `final_emb`
- ❌ **性能低下**: mAP 18.06% (VeRi-776，120 epochs)
- ⚠️ **预期性能**: 25-35% mAP (Cross-ID Re-ID 任务)

### 调试结果
- ✅ 所有维度正确 (input → backbone → GCN → fusion → output)
- ✅ Feature map: 2048×16×16 ✓
- ✅ Grid pooling: 4×4=16 nodes ✓  
- ✅ GCN处理: 64 nodes (batch=4) ✓
- ✅ L2归一化: [1.0, 1.0, 1.0, 1.0] ✓
- ✅ 模型参数: 39.3M (vs 26.7M baseline) ✓

---

## 🚀 优化策略

### 1️⃣ **立即重新训练 Baseline** (最高优先级)
**目的**: 验证Bug修复后的性能提升

#### 配置
```yaml
MODEL:
  BACKBONE: resnet50_ibn_a
  USE_GCN: true
  GRID_H: 4
  GRID_W: 4
  GCN_NUM_LAYERS: 1
  GCN_HIDDEN_DIM: 512
  POOLING_TYPE: mean
  FUSION_TYPE: concat

OPTIMIZER:
  NAME: AdamW
  BASE_LR: 0.00035
  WEIGHT_DECAY: 0.0005

SCHEDULER:
  WARMUP_EPOCHS: 10
  
EPOCHS: 120
BATCH_SIZE: 64  # P=16, K=4
```

#### 训练命令
```bash
cd /users/sl3753/scratch/GCN_project

# 提交训练任务
sbatch scripts/slurm_jobs/train_bot_gcn_776.sh

# 或直接运行
python scripts/training/train_bot_gcn.py \
  --config configs/gcn_transformer_configs/bot_gcn_776.yaml \
  --output_dir outputs/bot_gcn_776_fixed
```

#### 预期结果
- **训练时间**: ~12-24小时 (120 epochs, H100/A100)
- **目标mAP**: 25-35% (VeRi-776 cross-ID)
- **提升幅度**: +7~17% (从18%基线)

---

### 2️⃣ **学习率优化** (如果性能仍不理想)

#### 当前设置
```python
BASE_LR: 0.00035  # 3.5e-4
WEIGHT_DECAY: 0.0005
```

#### 优化方案A: **降低学习率** (保守策略)
```yaml
OPTIMIZER:
  BASE_LR: 0.0001  # 1e-4 (降低3.5倍)
  WEIGHT_DECAY: 0.0005
  
SCHEDULER:
  WARMUP_EPOCHS: 20  # 增加warmup
  
EPOCHS: 150  # 延长训练
```

#### 优化方案B: **学习率衰减** (激进策略)
```yaml
OPTIMIZER:
  BASE_LR: 0.00035
  
SCHEDULER:
  NAME: MultiStepLR  # 替换CosineAnnealing
  MILESTONES: [40, 70, 100]  # 在这些epoch降低lr
  GAMMA: 0.1
  WARMUP_EPOCHS: 10
```

#### 优化方案C: **分层学习率** (精细策略)
```python
# 在 train_bot_gcn.py 中添加
param_groups = [
    {'params': model.backbone.parameters(), 'lr': base_lr * 0.1},  # Backbone慢
    {'params': model.gcn.parameters(), 'lr': base_lr},              # GCN标准
    {'params': model.fusion.parameters(), 'lr': base_lr},           # Fusion标准
    {'params': model.bottleneck.parameters(), 'lr': base_lr},       # BNNeck标准
    {'params': model.classifier.parameters(), 'lr': base_lr},       # Classifier标准
]
optimizer = optim.AdamW(param_groups, weight_decay=0.0005)
```

---

### 3️⃣ **数据增强优化**

#### 当前策略
```python
train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

#### 优化方案: **增强数据多样性**
```python
train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop((256, 256)),
    
    # 新增
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomGrayscale(p=0.1),
    T.RandomErasing(p=0.5, scale=(0.02, 0.15)),
    
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

### 4️⃣ **损失函数优化**

#### 当前设置
```yaml
LOSS:
  LABEL_SMOOTH: 0.1
  TRIPLET_MARGIN: 0.3
  ID_LOSS_WEIGHT: 1.0
  TRIPLET_LOSS_WEIGHT: 1.0
```

#### 优化方案A: **增加Triplet权重**
```yaml
LOSS:
  ID_LOSS_WEIGHT: 1.0
  TRIPLET_LOSS_WEIGHT: 2.0  # 从1.0 → 2.0
  TRIPLET_MARGIN: 0.5       # 从0.3 → 0.5 (更严格)
```

#### 优化方案B: **Center Loss** (需要修改代码)
```python
# 添加 Center Loss
from losses.center_loss import CenterLoss

center_loss = CenterLoss(num_classes=576, feat_dim=2048)

# 在训练循环中
id_loss, tri_loss = bot_loss(logits, feat, labels)
cen_loss = center_loss(feat, labels)

loss = id_loss + tri_loss + 0.0005 * cen_loss
```

---

### 5️⃣ **Backbone预训练优化**

#### 当前: ImageNet预训练
```yaml
MODEL:
  BACKBONE: resnet50_ibn_a
  PRETRAINED: true
  PRETRAINED_PATH: ""  # torchvision默认
```

#### 优化方案: **BoT Baseline预训练**
```yaml
MODEL:
  PRETRAINED_PATH: "outputs/bot_baseline_1_1/776/best_model.pth"
```

**实现**:
```python
# 在 train_bot_gcn.py 中添加加载逻辑
if config['MODEL']['PRETRAINED_PATH']:
    checkpoint = torch.load(config['MODEL']['PRETRAINED_PATH'])
    
    # 只加载backbone + bottleneck + classifier
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model'].items() 
                      if k in model_dict and 'gcn' not in k and 'fusion' not in k}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    logger.log(f"✅ Loaded BoT baseline weights from {pretrained_path}")
    logger.log(f"   Initialized {len(pretrained_dict)}/{len(model_dict)} parameters")
```

---

### 6️⃣ **GCN架构优化**

#### 当前配置
```yaml
GCN_NUM_LAYERS: 1
GCN_HIDDEN_DIM: 512
GCN_DROPOUT: 0.5
```

#### 优化方案A: **多层GCN**
```yaml
GCN_NUM_LAYERS: 2       # 1 → 2
GCN_HIDDEN_DIM: 512     # 保持
GCN_DROPOUT: 0.3        # 降低dropout
```

#### 优化方案B: **更大Hidden Dim**
```yaml
GCN_NUM_LAYERS: 1
GCN_HIDDEN_DIM: 1024    # 512 → 1024
GCN_DROPOUT: 0.5
```

---

### 7️⃣ **训练技巧**

#### Gradient Clipping
```python
# 在 train_one_epoch() 中添加
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()
```

#### EMA (Exponential Moving Average)
```python
from torch_ema import ExponentialMovingAverage

ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

# 训练循环
for epoch in range(epochs):
    train_one_epoch(...)
    ema.update()
    
    # 评估时使用EMA模型
    with ema.average_parameters():
        evaluate(model, ...)
```

#### Label Smoothing调整
```yaml
LOSS:
  LABEL_SMOOTH: 0.05  # 从0.1 → 0.05 (减少正则化)
```

---

## 📊 实验计划

### Phase 1: Baseline验证 (优先级: 🔥🔥🔥)
| 实验 | 配置 | 目标 | 时间 |
|------|------|------|------|
| **Exp1-1** | Bug修复后重训 | mAP 25-35% | 12-24h |

### Phase 2: 学习率优化 (如果Exp1-1 < 25%)
| 实验 | 配置 | 改动 | 时间 |
|------|------|------|------|
| **Exp2-1** | 降低LR | BASE_LR=1e-4 | 12-24h |
| **Exp2-2** | 分层LR | Backbone×0.1 | 12-24h |
| **Exp2-3** | MultiStep | Milestones=[40,70,100] | 12-24h |

### Phase 3: 损失函数优化
| 实验 | 配置 | 改动 | 时间 |
|------|------|------|------|
| **Exp3-1** | Triplet权重 | TRI_WEIGHT=2.0 | 12-24h |
| **Exp3-2** | Center Loss | +0.0005×CenterLoss | 12-24h |

### Phase 4: 架构优化
| 实验 | 配置 | 改动 | 时间 |
|------|------|------|------|
| **Exp4-1** | 多层GCN | NUM_LAYERS=2 | 12-24h |
| **Exp4-2** | 更大Hidden | HIDDEN_DIM=1024 | 12-24h |
| **Exp4-3** | BoT预训练 | 加载baseline权重 | 12-24h |

---

## 🎬 立即行动方案

### 步骤1: 重新训练Baseline (立即执行)
```bash
cd /users/sl3753/scratch/GCN_project

# 创建新的输出目录
mkdir -p outputs/bot_gcn_776_v2

# 提交训练任务
sbatch scripts/slurm_jobs/train_bot_gcn_776.sh
```

### 步骤2: 监控训练
```bash
# 查看日志
tail -f outputs/bot_gcn_776_v2/train.log

# 查看SLURM输出
tail -f logs/bot_gcn_776_*.out
```

### 步骤3: 中期评估 (40 epochs)
```bash
# 如果mAP < 20%, 立即停止并调整学习率
# 如果mAP 20-25%, 继续训练
# 如果mAP > 25%, 说明修复成功
```

### 步骤4: 最终评估 (120 epochs)
```bash
python eval.py \
  --config configs/gcn_transformer_configs/bot_gcn_776.yaml \
  --checkpoint outputs/bot_gcn_776_v2/best_model.pth
```

---

## 📈 性能预期

### VeRi-776 Cross-ID Re-ID
| 模型 | mAP | Rank-1 | Rank-5 |
|------|-----|--------|--------|
| BoT Baseline (旧) | 18% | ~40% | ~60% |
| **BoT-GCN (预期)** | **25-35%** | **50-65%** | **70-85%** |
| IBNT-Net (SOTA) | ~40% | ~70% | ~88% |

### 改进空间
- ✅ Bug修复: +7~17% mAP
- 🔄 学习率优化: +2~5% mAP
- 🔄 损失函数优化: +1~3% mAP
- 🔄 架构优化: +2~5% mAP
- **总计潜力**: +12~30% mAP

---

## 🛠️ 代码修改清单

### 必须修改 (已完成)
- [x] `models/bot_baseline/bot_gcn_model.py` - 修复eval模式特征提取

### 推荐修改
- [ ] `scripts/training/train_bot_gcn.py` - 添加分层学习率
- [ ] `scripts/training/train_bot_gcn.py` - 添加gradient clipping
- [ ] `scripts/training/train_bot_gcn.py` - 添加BoT预训练加载
- [ ] `models/bot_baseline/veri_dataset.py` - 增强数据增强
- [ ] `losses/combined_loss.py` - 调整Triplet权重
- [ ] `losses/center_loss.py` - 添加Center Loss (新文件)

---

## ⏱️ 时间估算

| 阶段 | 任务 | 时间 |
|------|------|------|
| **立即** | 提交Exp1-1 | 5分钟 |
| **12-24h** | 训练完成 | 等待 |
| **+1h** | 评估分析 | 主动 |
| **决策** | 是否需要Phase 2-4 | 5分钟 |
| **总计** | Phase 1 | ~1天 |
| **总计** | 全部实验 (如需) | ~1-2周 |

---

## 🎯 成功标准

### Baseline (Exp1-1)
- ✅ **合格**: mAP ≥ 25%
- ⭐ **良好**: mAP ≥ 30%
- 🏆 **优秀**: mAP ≥ 35%

### 后续优化
- 🎯 **目标**: 逼近IBNT-Net (mAP ~40%)
- 📊 **可发表**: mAP ≥ 35% (Cross-ID Re-ID任务)

---

**下一步**: 立即执行 **步骤1** - 重新训练Baseline
