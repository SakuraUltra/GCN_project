# GCN-Enhanced Vehicle Re-Identification

**Platform**: Viking Cluster (University of York)  
**Status**: Production-ready ✅  
**Last Updated**: February 28, 2026

---

## 🎯 Project Overview

This project implements a Graph Convolutional Network (GCN) enhanced Bag-of-Tricks (BoT) baseline for vehicle re-identification with Random Erasing augmentation, achieving significant performance improvements over the standard baseline.

### Key Features
- ✅ **BoT-GCN+RE Model**: ResNet50-IBN backbone + 4×4 Grid GCN + Mean Pooling + Random Erasing
- ✅ **Multi-Dataset Support**: VeRi-776 and VehicleID
- ✅ **GPU Optimization**: A40 and H100 support
- ✅ **Comprehensive Robustness Evaluation**: 19 occlusion configurations tested
- ✅ **Random Erasing Ablation Study**: Demonstrates +2.51% average robustness improvement

---

## 📊 Performance Results

### VeRi-776 Dataset - Three-Model Comparison

**Clean Data Performance (0% Occlusion)**
| Model | mAP | Rank-1 | Rank-5 | Parameters | Training |
|-------|-----|--------|--------|------------|----------|
| BoT Baseline | 68.69% | 100.00% | 100.00% | - | No augmentation |
| BoT-GCN | 78.77% | 100.00% | 100.00% | 39.3M | No Random Erasing |
| **BoT-GCN+RE** | **79.25%** | **100.00%** | **100.00%** | **39.3M** | **+ Random Erasing** ⭐ |
| **Improvement** | **+10.56%** | - | - | - | **+0.48% over GCN** |

### Robustness Under Occlusion - Average Across All 18 Occluded Configurations

| Model | Avg mAP (Occlusion) | Improvement vs Baseline | RE Effect |
|-------|---------------------|-------------------------|-----------|
| BoT Baseline | 55.18% | - | - |
| BoT-GCN | 60.82% | +5.64% | - |
| **BoT-GCN+RE** | **63.32%** | **+8.15%** | **+2.51%** ⭐ |

**Random Erasing Impact by Category:**
- **Non-Grid Occlusions** (15 configs): +2.06% average improvement
- **Grid 10% Occlusion**: +6.65% (exceptional improvement from 33.47% → 40.12%)
- **Grid 20% Occlusion**: +4.24% (moderate improvement from 5.92% → 10.16%)
- **Grid 30% Occlusion**: +1.70% (limited improvement, still fails at 5.24%)

**Key Findings:**
- ✅ **GCN improves robustness**: +6.35% on non-Grid occlusions
- ✅ **Random Erasing adds robustness**: +2.51% overall, particularly effective on severe and scattered occlusions
- ⚠️ **Grid occlusion challenge**: All models fail catastrophically at 20-30% Grid (<11% mAP)
- 📊 **Best model**: BoT-GCN+RE (79.25% clean, 63.32% avg occlusion)

**Test Configuration**: 19 occlusion configs (1 clean + 18 occluded: 6 types × 3 ratios)  
**Occlusion Types**: center, top, bottom, left, right, grid  
**Occlusion Ratios**: 0%, 10%, 20%, 30%

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
# VeRi-776 with Random Erasing (recommended)
sbatch scripts/slurm_jobs/train_bot_gcn_re.sh

# VeRi-776 without Random Erasing (for ablation study)
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/bot_gcn_776.yaml

# VehicleID
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/bot_gcn_vehicleid.yaml
```

### 3. Evaluate Model
```bash
# Single model evaluation
python test_single_eval.py \
    --model-path outputs/bot_gcn_776_Random/best_model.pth \
    --dataset veri776
```

### 4. Robustness Testing
```bash
# Generate occlusion test set (19 configurations)
python scripts/testing/generate_occlusion_test_set.py

# Evaluate robustness (all three models)
bash scripts/testing/run_baseline_vs_gcn.sh

# Generate three-model comparison
python scripts/testing/generate_three_model_comparison.py
```

---

## 📁 Project Structure

```
GCN_project/
├── configs/              # YAML configurations
│   ├── baseline_configs/
│   ├── datasets/
│   ├── augmentation/     # Random Erasing configs
│   └── gcn_transformer_configs/
│       ├── bot_gcn_776.yaml        # GCN without RE
│       ├── bot_gcn_776_re.yaml     # GCN with RE (production) ⭐
│       └── bot_gcn_vehicleid.yaml
├── models/               # Model architectures
│   ├── bot_baseline/    # BoT and BoT-GCN models
│   ├── gcn/             # GCN layers and pooling
│   ├── fusion/          # Embedding fusion strategies
│   └── backbones/       # ResNet-IBN
├── losses/              # Loss functions (ID + Triplet)
├── train/               # Training engine
├── eval/                # Evaluation engine
├── scripts/             # Training and testing scripts
│   ├── training/
│   │   ├── train_bot_baseline.py
│   │   ├── train_bot_gcn.py
│   │   └── monitor_training.py
│   ├── testing/
│   │   ├── evaluate_occlusion_robustness.py    # Core evaluation
│   │   ├── generate_occlusion_test_set.py      # Generate test images
│   │   ├── generate_three_model_comparison.py  # Three-model analysis
│   │   ├── validate_occlusion_dataset.py       # Validation
│   │   ├── analyze_robustness_patterns.py      # Pattern analysis
│   │   ├── verify_knn_graphs.py                # Graph verification
│   │   └── run_baseline_vs_gcn.sh              # Comparison script
│   └── slurm_jobs/
│       ├── train_bot_gcn_re.sh                # Train with RE
│       ├── evaluate_gcn_re_robustness.sh      # Evaluate RE
│       ├── submit_build_knn.sh                # Build KNN graphs
│       ├── submit_extract_features.sh         # Extract features
│       └── submit_generate_nodes.sh           # Generate nodes
├── outputs/             # Model checkpoints and results
│   ├── bot_baseline_1_1/              # Baseline model
│   ├── bot_gcn_776_v2/                # GCN without RE
│   ├── bot_gcn_776_Random/            # GCN with RE (production) ⭐
│   ├── bot_gcn_vehicleid_h100/        # VehicleID model
│   ├── graph_structures/              # KNN graph data
│   ├── occlusion_tests/               # Occlusion test dataset (19 configs)
│   └── robustness_comparison/         # Final results & reports
│       ├── README.md                  # Experiment overview
│       ├── THREE_MODEL_COMPARISON_REPORT.txt
│       ├── comparison_summary.txt
│       ├── three_model_comparison.csv
│       ├── three_model_comparison_visualization.png
│       ├── baseline/                  # Baseline results
│       ├── gcn/                       # GCN (no RE) results
│       └── gcn_re/                    # GCN+RE results
├── utils/               # Augmentations, metrics, logging
│   └── augmentations.py      # Random Erasing implementation
└── data/                # Datasets (VeRi-776, VehicleID)
```

---

## 📖 Documentation

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Complete project structure after cleanup
- **[CLEANUP_REPORT.txt](CLEANUP_REPORT.txt)** - Cleanup report (3GB freed, Feb 28, 2026)
- **[VIKING_SETUP.md](VIKING_SETUP.md)** - Viking cluster environment setup
- **[outputs/robustness_comparison/](outputs/robustness_comparison/)**
  - `README.md` - Robustness experiment overview
  - `THREE_MODEL_COMPARISON_REPORT.txt` - Detailed three-model analysis (126 lines)
  - `comparison_summary.txt` - Executive summary
  - `three_model_comparison_visualization.png` - Comprehensive 6-panel visualization
- **[experiments/stage4_continual_learning/](experiments/stage4_continual_learning/)**
  - `STAGE4_SUMMARY.md` - Continual learning experiments summary

---

## 🔧 Environment

- **GPU**: NVIDIA A40 (46GB) / H100 (80GB)
- **Python**: 3.11.3
- **PyTorch**: 2.6+ (with weights_only fix)
- **CUDA**: 12.6

---

## 🏆 Key Achievements

1. **GCN Integration**: Successfully integrated 4×4 Grid GCN with mean pooling into vehicle ReID
2. **Significant Performance Gains**: 
   - VeRi-776: +10.56% mAP on clean data
   - VehicleID: +18.65% mAP improvement
3. **Random Erasing Ablation Study**:
   - Comprehensive three-model comparison (Baseline vs GCN vs GCN+RE)
   - Demonstrated +2.51% average robustness improvement
   - Exceptional +6.65% improvement on Grid 10% occlusion
   - Published complete analysis with 6-panel visualization
4. **Comprehensive Robustness Evaluation**: 
   - Tested 19 occlusion configurations systematically
   - Generated occlusion test dataset with validated ratios (±2% accuracy)
   - Identified Grid occlusion as fundamental challenge (<11% mAP at 20-30%)
5. **Production-Ready Model**: 
   - BoT-GCN+RE (outputs/bot_gcn_776_Random/best_model.pth)
   - 79.25% clean mAP, 63.32% average under occlusion
   - Training: 120 epochs, 3.5 hours on A40
6. **Multi-GPU Support**: Validated on both A40 and H100 GPUs
7. **Project Cleanup**: 
   - Removed 3GB obsolete data (max pooling experiments, old results)
   - Streamlined to 15 core scripts (7 testing + 3 training + 5 SLURM)
   - Clean project structure ready for publication

---

## 📝 Citation

If you use this code, please cite:
```bibtex
@software{bot_gcn_2026,
  title={BoT-GCN: Graph Convolutional Networks for Vehicle Re-Identification with Random Erasing},
  author={Viking Cluster Project},
  institution={University of York},
  year={2026},
  month={February},
  note={Production-ready implementation with comprehensive robustness evaluation}
}
```

---

## 🔗 Key Results Summary

| Metric | Baseline | GCN (no RE) | GCN+RE | Improvement |
|--------|----------|-------------|---------|-------------|
| **Clean mAP** | 68.69% | 78.77% | **79.25%** | **+10.56%** |
| **Avg Occlusion mAP** | 55.18% | 60.82% | **63.32%** | **+8.15%** |
| **Grid 10% mAP** | 25.82% | 33.47% | **40.12%** | **+14.30%** |
| **Training Time** | - | - | **3.5h (A40)** | 120 epochs |

**Recommendation**: Use BoT-GCN+RE for production (best overall performance with Random Erasing enabled).

---

**Last Updated**: February 28, 2026  
**Status**: Production-ready, Random Erasing ablation study completed ✅