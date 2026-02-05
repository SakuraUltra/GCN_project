# 项目结构重构说明

## 📁 新的目录结构

```
GCN_transformer/
├── data/                    # 数据集目录
│   ├── 776_DataSet/        # VeRi-776数据集
│   ├── dataset_splits/     # 数据集划分
│   └── transforms/         # 数据增强
│
├── models/                  # 模型架构（只包含网络结构）
│   ├── backbones/          # 骨干网络（ResNet-IBN等）
│   ├── bot_baseline/       # BoT-Baseline模型
│   ├── gcn/                # 图卷积网络
│   ├── transformer/        # Transformer模块
│   └── lora/               # LoRA适配器
│
├── losses/                  # ✨ 新增：损失函数模块
│   ├── __init__.py
│   ├── id_loss.py          # ID分类损失（带标签平滑）
│   ├── triplet_loss.py     # Triplet度量学习损失
│   └── combined_loss.py    # BoTLoss组合损失
│
├── train/                   # ✨ 新增：训练模块
│   ├── __init__.py
│   ├── trainer.py          # AMPTrainer训练引擎
│   └── scheduler.py        # 学习率调度器
│
├── eval/                    # ✨ 新增：评估模块
│   ├── __init__.py
│   └── evaluator.py        # ReIDEvaluator评估引擎
│
├── configs/                 # YAML配置文件
│   ├── baseline_configs/   # Baseline配置
│   ├── datasets/           # 数据集配置
│   └── gcn_transformer_configs/
│
├── utils/                   # 工具函数
│   ├── logger.py           # 日志工具
│   ├── metrics.py          # 评估指标
│   ├── reproducibility.py  # ✨ 新增：随机种子固定
│   └── ...
│
├── scripts/                 # 实验脚本（保留旧版兼容）
│   ├── train_bot_baseline.py
│   └── eval_cl.py
│
├── train.py                 # ✨ 新增：训练主入口
├── eval.py                  # ✨ 新增：评估主入口
└── README.md
```

## 🎯 重构完成的任务

### ✅ 已完成
1. **创建 `losses/` 目录** - 所有损失函数独立管理
   - `id_loss.py`: CrossEntropyLabelSmooth
   - `triplet_loss.py`: TripletLoss (Hard Mining)
   - `combined_loss.py`: BoTLoss

2. **创建 `train/` 目录** - 训练逻辑独立管理
   - `trainer.py`: AMPTrainer (支持混合精度训练)
   - `scheduler.py`: 学习率调度器

3. **创建 `eval/` 目录** - 评估逻辑独立管理
   - `evaluator.py`: ReIDEvaluator (支持mAP和CMC计算)

4. **创建标准入口点**
   - `train.py`: 训练主入口
   - `eval.py`: 评估主入口

5. **添加随机种子固定功能**
   - `utils/reproducibility.py`: set_random_seed()

6. **删除旧的 `engine/` 目录** - 代码结构更清晰

7. **更新所有import路径** - 确保代码可运行

## 🚀 使用方法

### 训练模型
```bash
# 使用默认配置训练
python train.py

# 使用自定义配置
python train.py --config configs/baseline_configs/bot_baseline.yaml \
                --data_root data/776_DataSet \
                --output_dir outputs/my_experiment \
                --seed 42

# 恢复训练
python train.py --resume outputs/bot_baseline/checkpoint_epoch_60.pth
```

### 评估模型
```bash
# 评估训练好的模型
python eval.py --checkpoint outputs/bot_baseline/best_model.pth \
               --data_root data/776_DataSet \
               --metric cosine

# 使用测试时增强和重排序
python eval.py --checkpoint outputs/bot_baseline/best_model.pth \
               --use_flip_test \
               --use_rerank
```

### 在代码中使用新模块
```python
# 导入损失函数
from losses import BoTLoss, TripletLoss, CrossEntropyLabelSmooth

# 导入训练器
from train import AMPTrainer, create_warmup_cosine_scheduler

# 导入评估器
from eval import ReIDEvaluator

# 设置随机种子
from utils.reproducibility import set_random_seed
set_random_seed(42)
```

## 📋 配置文件管理

所有超参数都从YAML配置文件加载，无硬编码：

```yaml
# configs/baseline_configs/bot_baseline.yaml
DATA:
  BATCH_SIZE: 64
  NUM_WORKERS: 4
  
OPTIMIZER:
  BASE_LR: 0.00035
  WEIGHT_DECAY: 0.0005
  
EPOCHS: 120
```

## 🔄 与旧代码的兼容性

- `scripts/train_bot_baseline.py` 已更新import路径，仍可使用
- 旧的 `engine/` 目录已删除
- 所有功能迁移到新的 `losses/`, `train/`, `eval/` 目录

## ✨ 新特性

1. **随机种子固定** - 确保实验可复现
   ```python
   set_random_seed(42, deterministic=True)
   ```

2. **标准化入口点** - 根目录的 `train.py` 和 `eval.py`

3. **模块化设计** - 每个模块职责单一，易于维护

4. **清晰的代码组织** - 符合深度学习项目标准结构

## 📊 目录对比

| 旧结构 | 新结构 | 说明 |
|--------|--------|------|
| `engine/trainer.py` | `train/trainer.py` | 训练引擎 |
| `engine/evaluator.py` | `eval/evaluator.py` | 评估引擎 |
| `models/bot_baseline/bot_model.py` (含Loss) | `losses/*.py` | Loss独立 |
| `scripts/train_*.py` | `train.py` | 标准入口 |
| 无 | `utils/reproducibility.py` | 种子固定 |

---

**重构完成时间**: 2026-01-28  
**Python版本**: 3.8+  
**PyTorch版本**: 1.10+
