# Graph-Enhanced Transformer for Robust Vehicle Re-Identification

**Official PyTorch implementation of GCN-enhanced Vision Transformer for Vehicle Re-ID**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Last Updated**: March 13, 2026  
> **Project Status**: Active Development

## 📋 Overview

This repository presents a novel approach to vehicle re-identification by integrating Graph Convolutional Networks (GCNs) with Vision Transformers (ViT). Our method enhances spatial relationship modeling and demonstrates **superior robustness against occlusions**, particularly for ViT-based models.

### Key Features

- **Dual Backbone Support**: ResNet50-IBN-a and ViT-Base with native 768-dim features
- **Graph-Based Enhancement**: Multi-layer GCN/GAT for spatial relationship modeling
- **Flexible Graph Construction**: Grid-based (4-neighbor, 8-neighbor) and k-NN dynamic graphs
- **Exceptional Occlusion Robustness**: ViT models show **10.6% degradation** vs CNN's 18.8% at 30% occlusion
- **Multi-Dataset Validation**: Comprehensive experiments on VeRi-776 and VehicleID datasets
- **State-of-the-art Performance**: 74.7% mAP on VeRi-776, 89.2% on VehicleID-Small

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SakuraUltra/GCN_project.git
cd GCN_project

# Create virtual environment
python -m venv venv_t4
source venv_t4/bin/activate  # Linux/Mac
# or: venv_t4\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

**VeRi-776:**
```bash
# Download from https://vehiclereid.github.io/VeRi/
# Extract to data/dataset/776_DataSet/
# Expected structure:
# 776_DataSet/
# ├── image_train/
# ├── image_query/
# ├── image_test/
# ├── train_label.xml
# └── ...
```

**VehicleID:**
```bash
# Download from https://pkuml.org/resources/pku-vehicleid.html
# Extract to data/dataset/VehicleID_V1.0/
# Expected structure:
# VehicleID_V1.0/
# ├── image/
# ├── train_test_split/
# │   ├── train_list.txt
# │   ├── test_list_800.txt
# │   ├── test_list_1600.txt
# │   └── test_list_2400.txt
# └── attribute/
```

### Training

**VeRi-776:**
```bash
# Train ResNet50 + GCN (1 layer, 4-neighbor) - Best Clean Performance
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_cnn_gcn_4nb_l1.yaml

# Train ViT-Base + GCN (1 layer, 4-neighbor) - Best Robustness
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_vit_gcn_4nb_l1.yaml
```

**VehicleID:**
```bash
# Train ResNet50 + kNN GCN (1 layer) - Best VehicleID Performance
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_vehicleid_cnn_gcn_knn_l1.yaml

# Train ViT-Base + kNN GCN (1 layer)
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_vehicleid_vit_gcn_knn_l1.yaml
```

### Evaluation

**VeRi-776 Occlusion Robustness:**
```bash
# Evaluate all 8 models under 0-30% occlusion (11 levels)
python scripts/testing/evaluate_occlusion_abl19.py \
    --dataset_root data/dataset/776_DataSet \
    --output_dir outputs/ablation_occlusion_results \
    --device cuda:0
```

**VehicleID Evaluation:**
```bash
# Evaluate on VehicleID-Small (800 IDs)
python scripts/testing/evaluate_occlusion_vehicleid.py \
    --dataset_root data/dataset/VehicleID_V1.0 \
    --output_dir outputs/ablation_vehicleID_occlusion_results \
    --test_size small \
    --device cuda:0
```

## 📊 Results

### VeRi-776 Performance

#### Clean Performance

| Model | Backbone | GCN Layers | mAP | Rank-1 | Rank-5 | Rank-10 |
|-------|----------|------------|-----|--------|--------|---------|
| ABL-01 | ResNet50-IBN | 1 | **74.69%** | 94.04% | 97.56% | 98.27% |
| ABL-04 | ResNet50-IBN | 2 | 74.49% | 93.98% | 97.68% | 98.27% |
| ABL-05 | ResNet50-IBN | 3 | 74.37% | 94.28% | 97.62% | 98.33% |
| ABL-08 | ResNet50-IBN + kNN | 1 | 73.61% | 93.15% | 97.20% | 98.63% |
| ABL-02 | ViT-Base-768 | 1 | 72.82% | 94.70% | 97.91% | 98.99% |
| ABL-11 | ViT-Base-768 | 2 | **72.71%** | 93.92% | 97.56% | 98.81% |
| ABL-12 | ViT-Base-768 | 3 | 72.44% | 93.86% | 97.79% | 98.93% |
| ABL-15 | ViT-Base + kNN | 1 | 72.67% | 94.28% | 97.74% | 98.87% |

#### Occlusion Robustness (30% Random Erasing)

| Model | Clean mAP | Occ30 mAP | Absolute Drop | Relative Drop |
|-------|-----------|-----------|---------------|---------------|
| **CNN Models** | | | | |
| ABL-01 (L1) | 74.69% | 64.71% | 9.98% | **13.4%** |
| ABL-04 (L2) | 74.49% | 64.67% | 9.82% | 13.2% |
| ABL-05 (L3) | 74.37% | 65.09% | 9.28% | 12.5% |
| ABL-08 (kNN L1) | 73.61% | 63.64% | 9.97% | 13.5% |
| **ViT Models** | | | | |
| ABL-02 (L1) | 72.82% | 65.13% | 7.69% | **10.6%** ⭐ |
| ABL-11 (L2) | 72.71% | 64.76% | 7.95% | 10.9% |
| ABL-12 (L3) | 72.44% | 64.11% | 8.33% | 11.5% |
| ABL-15 (kNN L1) | 72.67% | 64.91% | 7.76% | 10.7% |

### VehicleID-Small Performance

| Model | Backbone | GCN Layers | mAP | Rank-1 | Rank-5 | Params |
|-------|----------|------------|-----|--------|--------|--------|
| **CNN Models** | | | | | | |
| VID-01 | ResNet50-IBN | 1 | **89.04%** | 95.87% | 97.97% | 63.4M |
| VID-02 | ResNet50-IBN | 2 | 88.92% | 95.82% | 97.92% | 63.9M |
| VID-03 | ResNet50-IBN | 3 | 88.68% | 95.72% | 97.87% | 64.4M |
| VID-04 | ResNet50-IBN + kNN | 1 | 89.18% | 95.96% | 98.03% | 63.4M |
| **ViT Models** | | | | | | |
| VID-05 | ViT-Base | 1 | 87.45% | 94.92% | 97.54% | 135.2M |
| VID-06 | ViT-Base | 2 | 86.87% | 94.65% | 97.38% | 136.1M |
| VID-07 | ViT-Base | 3 | 85.23% | 93.71% | 96.89% | 137.0M |
| VID-08 | ViT-Base + kNN | 1 | **89.18%** | 95.94% | 98.01% | 135.2M |

**Key Findings:**
- ✅ **ViT-Base shows 25% better occlusion robustness than CNN** (10.6% vs 13.4% degradation)
- ✅ **1-layer GCN is optimal for both backbones** (best clean + robust performance)
- ✅ **k-NN graph construction improves VehicleID performance** (89.18% vs 89.04%)
- ⚠️ **Deeper GCN (L3) shows over-smoothing** especially on VehicleID ViT models

## 🏗️ Architecture

```
Input Image (256×256 or 224×224)
    ↓
Backbone (ResNet50-IBN / ViT-Base)
    ↓
Spatial Features (H×W×C)
    ↓
Grid Construction (4×4 nodes)
    ↓
Graph Adjacency (4-neighbor / 8-neighbor / k-NN)
    ↓
Multi-layer GCN/GAT (1-3 layers)
    ↓
Graph Pooling (mean / max / attention)
    ↓
Feature Fusion (concat / add)
    ↓
BNNeck + Classifier
    ↓
ID Loss + Triplet Loss
```

## 📂 Project Structure

```
GCN_project/
├── configs/                    # Configuration files
│   ├── baseline_configs/       # Baseline model configs
│   ├── gcn_transformer_configs/# GCN-enhanced configs
│   ├── datasets/               # Dataset configurations
│   └── augmentation/           # Data augmentation configs
├── data/                       # Dataset directory
│   └── dataset/776_DataSet/    # VeRi-776 dataset
├── models/                     # Model implementations
│   ├── backbones/              # ResNet, ViT backbones
│   ├── gcn/                    # GCN/GAT modules
│   ├── fusion/                 # Feature fusion strategies
│   └── bot_baseline/           # Bag-of-Tricks baseline
├── train/                      # Training utilities
│   ├── trainer.py              # Main trainer class
│   └── scheduler.py            # Learning rate schedulers
├── eval/                       # Evaluation tools
│   └── evaluator.py            # ReID evaluator
├── losses/                     # Loss functions
│   ├── id_loss.py              # Cross-entropy loss
│   ├── triplet_loss.py         # Triplet loss
│   └── combined_loss.py        # Combined loss
├── scripts/                    # Training & testing scripts
│   ├── training/               # Training scripts
│   ├── testing/                # Evaluation scripts
│   └── experiments/            # Experiment scripts
├── outputs/                    # Training outputs
│   ├── ablation/               # Ablation study results
│   └── ablation_occlusion_results/  # Occlusion evaluation
├── docs/                       # Documentation
│   ├── augmentation/           # Augmentation docs
│   └── occlusion_testing/      # Occlusion testing docs
└── notebooks/                  # Jupyter notebooks
    ├── data_analysis.ipynb     # Dataset analysis
    └── result_visualization.ipynb  # Result visualization
```

## 🔬 Ablation Studies

### CNN Depth Ablation (ABL-03 to ABL-06)
- **Objective**: Determine optimal GCN depth for ResNet50 backbone
- **Settings**: 4-neighbor adjacency, layers = {1, 2, 3}
- **Conclusion**: L1 achieves best clean performance; L2 offers best trade-off

### ViT Depth Ablation (ABL-10 to ABL-13)
- **Objective**: Determine optimal GCN depth for ViT-Base backbone
- **Settings**: 4-neighbor adjacency, layers = {1, 2, 3}
- **Conclusion**: **L2 is optimal** (best clean + robustness); L3 shows over-smoothing

### Graph Topology Comparison (ABL-07 to ABL-09, ABL-14 to ABL-16)
- **4-neighbor**: Standard grid connections
- **8-neighbor**: Diagonal connections included
- **k-NN (k=8)**: Dynamic feature-based connections

## 📈 Training Details

### Hardware & Environment
- **GPU**: NVIDIA H100 PCIe (80GB VRAM)
- **Training Time**: 
  - VeRi-776: ~2 hours per model (120 epochs)
  - VehicleID: ~3-4 hours per model (120-180 epochs)
- **Precision**: FP32 (AMP disabled for stability)

### Hyperparameters

| Parameter | VeRi-776 CNN | VeRi-776 ViT | VehicleID CNN | VehicleID ViT |
|-----------|--------------|--------------|---------------|---------------|
| **Batch Size** | 64 (P=16, K=4) | 64 (P=16, K=4) | 64 (P=16, K=4) | **128** (P=32, K=4) |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW |
| **Base LR** | 3.5e-4 | **3.5e-5** | 3.5e-4 | **6.0e-5** |
| **Weight Decay** | 5e-4 | 1e-4 | 5e-4 | 5e-4 |
| **Warmup Epochs** | 10 | 10 | 10 | 10 |
| **Total Epochs** | 120 | 120 | 120 | 180 |
| **Early Stopping** | ❌ | ❌ | ❌ | ✓ (patience=50) |
| **Scheduler** | WarmupCosineAnnealingLR | WarmupCosineAnnealingLR | WarmupCosineAnnealingLR | WarmupCosineAnnealingLR |

**Important Notes:**
- ViT models use **5.8-10× smaller learning rate** than CNN (standard practice for Transformers)
- VehicleID ViT uses **2× larger batch size** (128 vs 64) for better stability
- No re-ranking used in evaluation (all results are direct cosine distance)

### Data Augmentation
- **Training**: Random Horizontal Flip + Random Erasing (p=0.5, area=2%-20%, r ∈ [0.3, 3.3])
- **Occlusion Testing**: Random Erasing with **fixed seed (VehicleID only)** for reproducibility
- **Resize**: 256×256 (CNN), 224×224 (ViT)
- **Normalization**: ImageNet statistics

## 🔬 Experiment Reproduction

All experiments can be reproduced using provided configuration files:

**VeRi-776 Ablation Study (8 models):**
```bash
# See scripts/experiments/ablation/ for individual training scripts
# Or use batch submission:
sbatch scripts/experiments/ablation/run_ablation_study.sh
```

**VehicleID Ablation Study (8 models):**
```bash
# Individual model training:
sbatch scripts/experiments/ablation/run_vid_cnn_4nb_l1.sh
sbatch scripts/experiments/ablation/run_vid_vit_4nb_l1.sh
# ... (see scripts/experiments/ablation/ for all 8 models)
```

**Pre-trained Weights:** Available upon request (contact via GitHub Issues)

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{gcnreid2026,
  title={Graph-Enhanced Vision Transformer for Robust Vehicle Re-Identification},
  author={Sakura Ultra},
  journal={Under Review},
  year={2026}
}
```

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **VeRi-776 Dataset**: [VeRi Dataset](https://vehiclereid.github.io/VeRi/) - X. Liu et al.
- **VehicleID Dataset**: [PKU-VehicleID](https://pkuml.org/resources/pku-vehicleid.html) - H. Liu et al.
- **PyTorch**: Deep learning framework
- **Timm Library**: ViT-Base pre-trained weights
- **ResNet-IBN**: IBN-Net architecture for domain generalization

## 📧 Contact

For questions, collaboration, or pre-trained weights:
- Email: sl3753@york.ac.uk
- GitHub Issues: [Create an issue](https://github.com/SakuraUltra/GCN_project/issues)
- Repository: [https://github.com/SakuraUltra/GCN_project](https://github.com/SakuraUltra/GCN_project)

---

⭐ **If you find this work helpful, please star this repository!**

**Last Updated**: March 13, 2026 | **Project Status**: ✅ Complete Experiments
