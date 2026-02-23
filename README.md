# GCN-Enhanced Vehicle Re-Identification

**Platform**: Viking Cluster (University of York)  
**Status**: Production-ready ✅

---

## 🎯 Project Overview

This project implements a Graph Convolutional Network (GCN) enhanced Bag-of-Tricks (BoT) baseline for vehicle re-identification, achieving significant performance improvements over the standard BoT baseline.

### Key Features
- ✅ **BoT-GCN Model**: ResNet50-IBN backbone + 4×4 Grid GCN + Fusion layers
- ✅ **Multi-Dataset Support**: VeRi-776 and VehicleID
- ✅ **GPU Optimization**: A40 and H100 support
- ✅ **Mixed Precision Training**: Automatic Mixed Precision (AMP) with sparse matrix compatibility

---

## 📊 Performance Results

### VeRi-776 Dataset
| Model | mAP | Rank-1 | Rank-5 | Rank-10 |
|-------|-----|--------|--------|---------|
| BoT Baseline | 64.49% | 88.97% | - | - |
| **BoT-GCN** | **74.77%** | **93.62%** | **98.22%** | **99.15%** |
| **Improvement** | **+15.94%** | **+5.23%** | - | - |

### VehicleID Dataset (Small Test Set)
| Model | mAP | Rank-1 | Rank-5 | Rank-10 |
|-------|-----|--------|--------|---------|
| BoT Baseline | 76.28% | 70.84% | - | - |
| **BoT-GCN (H100)** | **90.51%** | **85.70%** | **96.80%** | **98.45%** |
| **BoT-GCN (A40)** | **90.10%** | **85.21%** | **96.29%** | **98.23%** |
| **Improvement** | **+18.65%** | **+20.98%** | - | - |

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd /users/sl3753/scratch/GCN_project
source venv_t4/bin/activate
```

### 2. Train BoT-GCN
```bash
# VeRi-776
sbatch scripts/slurm_jobs/submit_train_bot_gcn_776.sh

# VehicleID
sbatch scripts/slurm_jobs/submit_train_bot_gcn_vehicleid.sh
```

### 3. Evaluate Model
```bash
python scripts/testing/eval_bot_gcn.py \
    --checkpoint outputs/bot_gcn_776_v2/best_model.pth \
    --dataset veri776
```

---

## 📁 Project Structure

```
GCN_project/
├── configs/              # YAML configurations
│   ├── baseline_configs/
│   ├── datasets/
│   └── gcn_transformer_configs/
├── models/               # Model architectures
│   ├── bot_baseline/    # BoT and BoT-GCN models
│   ├── gcn/             # GCN layers and pooling
│   ├── fusion/          # Embedding fusion strategies
│   └── backbones/       # ResNet-IBN
├── losses/              # Loss functions
├── train/               # Training engine
├── eval/                # Evaluation engine
├── scripts/             # Training and testing scripts
│   ├── training/
│   ├── testing/
│   ├── slurm_jobs/
│   └── setup/          # Environment setup scripts (archived)
├── outputs/             # Model checkpoints and results
└── data/                # Datasets (VeRi-776, VehicleID)
```

---

## 📖 Documentation

- **[VIKING_SETUP.md](VIKING_SETUP.md)** - Viking cluster environment setup
- **[docs/archive/](docs/archive/)** - Historical documentation and debugging notes

---

## 🔧 Environment

- **GPU**: NVIDIA A40 (46GB) / H100 (80GB)
- **Python**: 3.11.3
- **PyTorch**: 2.6+ (with weights_only fix)
- **CUDA**: 12.6

---

## 🏆 Key Achievements

1. **GCN Integration**: Successfully integrated graph structure into vehicle ReID
2. **Significant Gains**: +15.94% mAP on VeRi-776, +18.65% on VehicleID
3. **Multi-GPU Support**: Validated on both A40 and H100 GPUs
4. **Bug Fixes**: 
   - Fixed GCN forward pass order (propagate before transform)
   - Fixed pretrained weight loading (model_state_dict key)
   - Fixed AMP compatibility with sparse matrices
   - Fixed PyTorch 2.6 weights_only compatibility

---

## 📝 Citation

If you use this code, please cite:
```
BoT-GCN: Graph Convolutional Networks for Vehicle Re-Identification
Viking Cluster, University of York, 2026
```

---

**Last Updated**: February 23, 2026  
**Status**: Production-ready, all major bugs fixed