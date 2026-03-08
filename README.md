# Graph-Enhanced Transformer for Robust Vehicle Re-Identification

**Official PyTorch implementation of GCN-enhanced Vision Transformer for Vehicle Re-ID**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This repository presents a novel approach to vehicle re-identification by integrating Graph Convolutional Networks (GCNs) with Vision Transformers (ViT). Our method enhances spatial relationship modeling and demonstrates superior robustness against occlusions.

### Key Features

- **Dual Backbone Support**: ResNet50-IBN-a and ViT-Base with native 768-dim features
- **Graph-Based Enhancement**: Multi-layer GCN/GAT for spatial relationship modeling
- **Flexible Graph Construction**: Grid-based (4-neighbor, 8-neighbor) and k-NN dynamic graphs
- **Occlusion Robustness**: Comprehensive evaluation under Random Erasing (0-30%)
- **State-of-the-art Performance**: 74%+ mAP on VeRi-776 dataset

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/GCN_project.git
cd GCN_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train ResNet50 + GCN (1 layer, 4-neighbor)
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_cnn_gcn_4nb_l1.yaml

# Train ViT-Base + GCN (2 layers, 4-neighbor)
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_vit_gcn_4nb_l2.yaml
```

### Evaluation

```bash
# Evaluate occlusion robustness
python scripts/testing/evaluate_occlusion_abl19.py \
    --dataset_root data/dataset/776_DataSet \
    --output_dir outputs/ablation_occlusion_results
```

## 📊 Results

### Clean Performance (VeRi-776)

| Model | Backbone | GCN Layers | mAP | Rank-1 | Rank-5 |
|-------|----------|------------|-----|--------|--------|
| ABL-01 | ResNet50-IBN | 1 | 74.69% | 94.04% | 97.56% |
| ABL-04 | ResNet50-IBN | 2 | 74.49% | 93.98% | 97.68% |
| ABL-05 | ResNet50-IBN | 3 | 74.37% | 94.28% | 97.62% |
| ABL-02 | ViT-Base-768 | 1 | 72.24% | 93.74% | 97.85% |
| ABL-11 | ViT-Base-768 | 2 | 72.71% | 94.10% | 97.62% |
| ABL-12 | ViT-Base-768 | 3 | 72.44% | 93.86% | 97.79% |

### Occlusion Robustness (30% Random Erasing)

| Model | Clean mAP | Occ30 mAP | Drop@30 | Relative Drop |
|-------|-----------|-----------|---------|---------------|
| ABL-01 (CNN L1) | 74.69% | 60.61% | 14.08% | 18.8% |
| ABL-04 (CNN L2) | 74.49% | 60.32% | 14.17% | 19.0% |
| ABL-05 (CNN L3) | 74.37% | 59.88% | 14.49% | 19.5% |
| ABL-02 (ViT L1) | 72.24% | 64.48% | 7.75% | 10.7% |
| ABL-11 (ViT L2) | 72.71% | 65.01% | 7.69% | 10.6% ⭐ |
| ABL-12 (ViT L3) | 72.44% | 62.85% | 9.59% | 13.2% |

**Key Findings:**
- ✅ **ViT-Base demonstrates superior occlusion robustness** (10.6% drop vs 18.8% for ResNet50)
- ✅ **2-layer GCN achieves optimal balance** between performance and robustness
- ⚠️ **3-layer GCN shows over-smoothing effects** in both backbones

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

### Hardware
- GPU: NVIDIA H100 PCIe (80GB)
- Training Time: ~40 minutes per model (120 epochs)
- Batch Size: 64 (P=16, K=4)

### Hyperparameters
- **Optimizer**: AdamW
- **Learning Rate**: 3.5e-5 (ViT), 1e-4 (ResNet50)
- **LR Schedule**: Warmup (10 epochs) + Cosine Annealing
- **Weight Decay**: 1e-4
- **Label Smoothing**: 0.1
- **Triplet Margin**: 0.3

### Data Augmentation
- Random Horizontal Flip
- Random Erasing (p=0.5, area=2%-20%)
- Color Jitter (optional)
- Resize + Normalize

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{yourname2026gcnreid,
  title={Graph-Enhanced Transformer for Robust Vehicle Re-Identification},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **VeRi-776 Dataset**: [VeRi Dataset](https://vehiclereid.github.io/VeRi/)
- **PyTorch Geometric**: Graph neural network library
- **Timm Library**: Vision Transformer implementations
- **Bag-of-Tricks**: ReID training best practices

## 📧 Contact

For questions or collaboration opportunities:
- Email: your.email@university.edu
- GitHub Issues: [Create an issue](https://github.com/YourUsername/GCN_project/issues)

---

⭐ **Star this repository if you find it helpful!**
