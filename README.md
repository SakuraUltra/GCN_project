# GCN-Transformer Vehicle Re-ID Framework

Stage 1 (BoT-Baseline) implementation complete ✅

## Core Components
- `models/bot_baseline/bot_model.py` - BoT-Baseline with ResNet50-IBN
- `models/bot_baseline/veri_dataset.py` - VeRi-776 dataset loader  
- `scripts/train_bot_baseline.py` - Training script
- `configs/baseline_configs/bot_baseline.yaml` - Configuration

## Usage
```bash
python scripts/train_bot_baseline.py
```