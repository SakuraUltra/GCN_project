# Scripts 目录结构

清晰的功能分类组织，便于维护和使用。

## 📁 目录说明

### 1️⃣ `graph_preparation/` - 图准备阶段
数据预处理和图结构构建脚本

| 文件 | 功能 | 用途 |
|------|------|------|
| `extract_features.py` | 提取 CNN 特征图 | Step 1: 从训练好的模型提取 (2048, 8, 8) 特征 |
| `generate_graph_nodes.py` | 网格池化生成节点 | Step 2: 将特征图划分为 N×M 图节点 |
| `build_graph_edges.py` | 构建固定邻接矩阵 | Step 3a: 4-邻居/8-邻居固定边 |
| `build_knn_graph.py` | 构建 kNN 动态图 | Step 3b: 基于特征相似度的 kNN 边 |

**使用示例：**
```bash
# Step 1: 提取特征
python graph_preparation/extract_features.py --config configs/baseline_configs/bot_baseline.yaml --checkpoint outputs/bot_baseline_1_1/776/best_model.pth --data_root data/dataset/776_DataSet --output_dir outputs/features/776_baseline

# Step 2: 生成节点
python graph_preparation/generate_graph_nodes.py --dataset 776 --grid-h 8 --grid-w 8

# Step 3a: 构建固定边
python graph_preparation/build_graph_edges.py --grid-h 8 --grid-w 8 --neighbor-type 8

# Step 3b: 构建 kNN 边
python graph_preparation/build_knn_graph.py --nodes-dir outputs/graph_nodes/776_baseline_grid_8x8 --k 8
```

---

### 2️⃣ `training/` - 训练脚本
模型训练和监控

| 文件 | 功能 |
|------|------|
| `train_bot_baseline.py` | BoT-Baseline 训练 |
| `monitor_training.py` | 训练进度监控 |

**使用示例：**
```bash
python training/train_bot_baseline.py --config configs/baseline_configs/bot_baseline.yaml
```

---

### 3️⃣ `testing/` - 测试和验证
功能测试、数据验证脚本

| 文件 | 功能 |
|------|------|
| `test_gcn_comprehensive.py` | GCN 深度测试套件（7项测试） |
| `test_gcn_real_data.py` | GCN 真实数据测试 |
| `compare_pooling_strategies.py` | 图池化策略对比（mean/max/attention） |
| `verify_features.py` | 验证提取的特征文件 |
| `verify_knn_graphs.py` | 验证 kNN 图文件 |
| `verify_environment.py` | 验证环境配置 |
| `debug_gcn_shapes.py` | 调试 GCN 形状问题 |

**使用示例：**
```bash
# 深度测试 GCN
python testing/test_gcn_comprehensive.py

# 对比池化策略
python testing/compare_pooling_strategies.py \
  --graph_nodes outputs/graph_nodes/776_baseline_grid_4x4/query_graph_nodes.pt \
  --output_dir outputs/pooling_comparison/776_4x4_query

# 验证特征
python testing/verify_features.py

# 验证环境
python testing/verify_environment.py
```

---

### 4️⃣ `slurm_jobs/` - SLURM 作业脚本
集群任务提交脚本

| 文件 | 功能 |
|------|------|
| `submit_bot_baseline_776.sh` | 提交 VeRi-776 基线训练 |
| `submit_bot_baseline_vehicleid.sh` | 提交 VehicleID 基线训练 |
| `submit_extract_features.sh` | 提交特征提取任务 |
| `submit_generate_nodes.sh` | 提交节点生成任务 |
| `submit_build_knn.sh` | 提交 kNN 图构建任务 |
| `run_grid_ablation_776.sh` | 批量运行 VeRi-776 网格实验 |
| `run_grid_ablation_vehicleid.sh` | 批量运行 VehicleID 网格实验 |

**使用示例：**
```bash
# 提交特征提取
sbatch slurm_jobs/submit_extract_features.sh 776

# 提交节点生成
sbatch slurm_jobs/submit_generate_nodes.sh 776 8 8

# 批量网格实验
bash slurm_jobs/run_grid_ablation_776.sh
```

---

## 🔄 工作流程

```
阶段1: 基线训练
  └─> slurm_jobs/submit_bot_baseline_*.sh

阶段2: 图准备
  ├─> graph_preparation/extract_features.py       (Step 1)
  ├─> graph_preparation/generate_graph_nodes.py  (Step 2)
  └─> graph_preparation/build_graph_edges.py     (Step 3)
       └─> graph_preparation/build_knn_graph.py  (Step 3 可选)

阶段3: GCN 训练
  ├─> testing/test_gcn_*.py                      (验证)
  └─> training/train_gcn.py                      (待实现)
```

---

## 📝 注意事项

1. **路径引用**：所有脚本都假定从项目根目录运行
2. **SLURM 脚本**：需要根据集群配置调整分区和资源
3. **测试脚本**：建议在本地先运行测试，再提交大规模任务
4. **数据依赖**：确保上游步骤完成后再运行下游脚本
