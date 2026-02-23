# Viking Cluster 环境配置指南

## 系统信息
- **集群**: University of York Viking Cluster
- **GPU**: NVIDIA A40 (46GB) / H100 (80GB)
- **Python**: 3.11.3
- **PyTorch**: 2.6+
- **CUDA**: 12.6

---

## 快速开始

### 日常使用（环境已配置）

```bash
cd /users/sl3753/scratch/GCN_project
source venv_t4/bin/activate
```

### 重新配置环境（仅当需要时）

环境配置脚本已归档至 `scripts/setup/`：

```bash
# 查看可用脚本
ls scripts/setup/

# 如需重新安装（不推荐）
bash scripts/setup/setup_viking_env_t4_v2.sh
```

---

## GPU使用

### 查看可用GPU
```bash
# 查看当前节点GPU
nvidia-smi

# 查看集群GPU分区
sinfo -p gpu,gpuplus -o "%P %G %N"
```

### GPU分区

#### gpu 分区 (NVIDIA A40)
- **显存**: 46GB
- **节点**: gpu03-gpu16 (14个节点)
- **性能**: 适合大多数训练任务

#### gpuplus 分区 (NVIDIA H100)
- **显存**: 80GB
- **节点**: gpu21-gpu26 (6个节点)
- **性能**: 约比A40快1.7倍
- **注意**: 经常排队，优先级低于gpu分区

### 提交训练任务

```bash
# VeRi-776 (A40)
sbatch scripts/slurm_jobs/submit_train_bot_gcn_776.sh

# VehicleID (H100)
sbatch scripts/slurm_jobs/submit_train_bot_gcn_vehicleid.sh

# 查看任务状态
squeue -u sl3753

# 取消任务
scancel <JOB_ID>
```

---

## 数据集位置

```bash
data/
├── 776_DataSet/          # VeRi-776
│   ├── image_train/
│   ├── image_query/
│   ├── image_test/
│   └── *.xml
└── dataset/
    └── VehicleID_V1.0/   # VehicleID
        ├── image/
        └── *.txt
```

---

## 训练和评估

### 训练
```bash
source venv_t4/bin/activate

# VeRi-776
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/bot_gcn_776.yaml \
    --output_dir outputs/bot_gcn_776_v3

# VehicleID
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/bot_gcn_vehicleid.yaml \
    --output_dir outputs/bot_gcn_vehicleid_v3
```

### 监控训练
```bash
# 查看SLURM输出
tail -f logs/bot_gcn_*.out

# 查看训练日志
tail -f outputs/bot_gcn_*/training.log

# 监控GPU
ssh gpu10 "nvidia-smi"
```

### 评估
```bash
python scripts/testing/eval_bot_gcn.py \
    --checkpoint outputs/bot_gcn_776_v2/best_model.pth \
    --dataset veri776
```

---

## 已知问题和解决方案

### 1. PyTorch 2.6 权重加载错误
**问题**: `UnpicklingError: Weights only load failed`

**解决方案**: 已在所有训练脚本中修复
```python
checkpoint = torch.load(path, weights_only=False)
```

### 2. AMP + 稀疏矩阵不兼容
**问题**: `NotImplementedError: sparse CUDA operations not supporting FP16`

**解决方案**: GCN前向传播禁用AMP
```python
with torch.cuda.amp.autocast(enabled=False):
    nodes_gcn = self.gcn(nodes_flat.float(), edge_index_batch)
```

### 3. H100排队时间长
**解决方案**: 
- 同时提交A40和H100任务
- A40训练速度足够快（~6小时/120 epochs）

---

## 性能基准

### VeRi-776 (120 epochs)
| GPU | 训练时间 | 最佳mAP |
|-----|---------|---------|
| A40 | ~4-5小时 | 74.77% |

### VehicleID (120 epochs)
| GPU | 训练时间 | 最佳mAP |
|-----|---------|---------|
| A40 | ~6小时 | 90.10% |
| H100 | ~3.6小时 | 90.51% |

---

## 常用命令

```bash
# 激活环境
source venv_t4/bin/activate

# 查看任务
squeue -u sl3753

# 查看GPU使用
ssh gpu10 "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv"

# 生成收敛曲线
python plot_bot_gcn_convergence.py  # VeRi-776
python plot_vehicleid_convergence.py  # VehicleID
```

---

**最后更新**: 2026-02-23  
**集群**: Viking (University of York)  
**GPU**: A40 / H100

---

## 快速开始

### 日常使用（环境已配置）

```bash
cd /users/sl3753/scratch/GCN_project
source venv_t4/bin/activate
```

### 重新配置环境（仅当需要时）

环境配置脚本已归档至 `scripts/setup/`：

```bash
# 查看可用脚本
ls scripts/setup/

# 如需重新安装（不推荐）
bash scripts/setup/setup_viking_env_t4_v2.sh
```

---

## GPU使用

### 查看可用GPU
```bash
# 查看当前节点GPU
nvidia-smi

# 查看集群GPU分区
sinfo -p gpu,gpuplus -o "%P %G %N"
```

### GPU分区

#### gpu 分区 (NVIDIA A40)
- **显存**: 46GB
- **节点**: gpu03-gpu16 (14个节点)
- **性能**: 适合大多数训练任务

#### gpuplus 分区 (NVIDIA H100)
- **显存**: 80GB
- **节点**: gpu21-gpu26 (6个节点)
- **性能**: 约比A40快1.7倍
- **注意**: 经常排队，优先级低于gpu分区

### 提交训练任务

```bash
# VeRi-776 (A40)
sbatch scripts/slurm_jobs/submit_train_bot_gcn_776.sh

# VehicleID (H100)
sbatch scripts/slurm_jobs/submit_train_bot_gcn_vehicleid.sh

# 查看任务状态
squeue -u sl3753

# 取消任务
scancel <JOB_ID>
```

---

## 数据集位置

```bash
data/
├── 776_DataSet/          # VeRi-776
│   ├── image_train/
│   ├── image_query/
│   ├── image_test/
│   └── *.xml
└── dataset/
    └── VehicleID_V1.0/   # VehicleID
        ├── image/
        └── *.txt
```

---

## 训练和评估

### 训练
```bash
source venv_t4/bin/activate

# VeRi-776
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/bot_gcn_776.yaml \
    --output_dir outputs/bot_gcn_776_v3

# VehicleID
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/bot_gcn_vehicleid.yaml \
    --output_dir outputs/bot_gcn_vehicleid_v3
```

### 监控训练
```bash
# 查看SLURM输出
tail -f logs/bot_gcn_*.out

# 查看训练日志
tail -f outputs/bot_gcn_*/training.log

# 监控GPU
ssh gpu10 "nvidia-smi"
```

### 评估
```bash
python scripts/testing/eval_bot_gcn.py \
    --checkpoint outputs/bot_gcn_776_v2/best_model.pth \
    --dataset veri776
```

---

## 已知问题和解决方案

### 1. PyTorch 2.6 权重加载错误
**问题**: `UnpicklingError: Weights only load failed`

**解决方案**: 已在所有训练脚本中修复
```python
checkpoint = torch.load(path, weights_only=False)
```

### 2. AMP + 稀疏矩阵不兼容
**问题**: `NotImplementedError: sparse CUDA operations not supporting FP16`

**解决方案**: GCN前向传播禁用AMP
```python
with torch.cuda.amp.autocast(enabled=False):
    nodes_gcn = self.gcn(nodes_flat.float(), edge_index_batch)
```

### 3. H100排队时间长
**解决方案**: 
- 同时提交A40和H100任务
- A40训练速度足够快（~6小时/120 epochs）

---

## 性能基准

### VeRi-776 (120 epochs)
| GPU | 训练时间 | 最佳mAP |
|-----|---------|---------|
| A40 | ~4-5小时 | 74.77% |

### VehicleID (120 epochs)
| GPU | 训练时间 | 最佳mAP |
|-----|---------|---------|
| A40 | ~6小时 | 90.10% |
| H100 | ~3.6小时 | 90.51% |

---

## 常用命令

```bash
# 激活环境
source venv_t4/bin/activate

# 查看任务
squeue -u sl3753

# 查看GPU使用
ssh gpu10 "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv"

# 生成收敛曲线
python plot_bot_gcn_convergence.py  # VeRi-776
python plot_vehicleid_convergence.py  # VehicleID
```

---

**最后更新**: 2026-02-23  
**集群**: Viking (University of York)  
**GPU**: A40 / H100
