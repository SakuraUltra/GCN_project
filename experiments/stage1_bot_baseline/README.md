# Stage 1: BoT-Baseline Implementation

## Experiment Overview
Implementation of Bag of Tricks (BoT) baseline for Vehicle Re-identification following the standard protocol.

## BoT-Baseline Configuration
- **Backbone**: ResNet50-IBN-a (single network)
- **Loss Functions**: 
  - ID Loss (Cross-Entropy with Label Smoothing)
  - Triplet Loss (with hard mining)
- **Training Tricks**:
  - Random Erasing augmentation
  - Warmup learning rate scheduling
  - Center Loss (optional)
- **Evaluation Metrics**: mAP + Rank-1

## Expected Performance
Target performance on VeRi-776:
- mAP: ~72-75%
- Rank-1: ~85-88%

This baseline will serve as the foundation for all subsequent experiments and comparisons.

## Files Structure
```
stage1_bot_baseline/
├── train_bot_baseline.py    # Training script
├── config_bot.yaml          # BoT configuration
├── results/                 # Training outputs
│   ├── logs/
│   ├── checkpoints/
│   └── metrics/
└── README.md               # This file
```

## Usage
```bash
cd experiments/stage1_bot_baseline/
python train_bot_baseline.py --config config_bot.yaml
```