# Stage 2: SOTA Competition vs IBN-Net

## Experiment Overview
After establishing BoT-Baseline performance, this stage compares our GCN-Transformer + LoRA approach against IBN-Net (current SOTA) on standard VeRi-776.

## Hypothesis
GCN-Transformer + LoRA can match IBN-Net performance while providing better parameter efficiency and continual learning capabilities.

## Experimental Setup
- **Foundation**: BoT-Baseline (Stage 1 results)
- **SOTA Competitor**: IBN-Net (ResNet-50-IBN-a)
- **Our Method**: GCN-Transformer + LoRA variants (G, T, Hybrid)
- **Dataset**: VeRi-776 standard split
- **Metrics**: mAP, CMC@1, CMC@5, CMC@10, Training time, VRAM usage

## Expected Results
- Match or exceed IBN-Net mAP (75-80%)
- Comparable CMC scores
- Reduced memory footprint
- Foundation for continual learning experiments

## Files
- `train_ibnt.py`: Training script for IBN-Net baseline
- `train_lora_variants.py`: Training script for our LoRA variants
- `performance_comparison.py`: Detailed performance analysis
- `results/`: Performance logs and model checkpoints