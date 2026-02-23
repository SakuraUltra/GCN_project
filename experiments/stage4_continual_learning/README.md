# Continual Learning Experiments - Core Contribution

## Experiment Overview
This is the core contribution experiment demonstrating our framework's ability to handle continual learning scenarios while preventing catastrophic forgetting.

## Hypothesis
1. GCN-Transformer + LoRA prevents catastrophic forgetting in continual learning
2. Our approach outperforms existing CL methods (INC-VeReID) 
3. LoRA variants show different CL performance characteristics

## Experimental Setup
- **Baselines**: 
  - IBN-Net (FFT) - shows catastrophic forgetting
  - INC-VeReID - existing CL method
  - CL-Adapter - PEFT CL baseline
- **Our Methods**: LoRA-G, LoRA-T, LoRA-Hybrid
- **Dataset**: VeRi-776 split into 4 continual tasks
- **Metrics**: Average accuracy, Forgetting measure, Intransigence, Memory efficiency

## Task Definition
- **Task 1**: Cameras 1-5 (Initial deployment)
- **Task 2**: Cameras 6-10 (Expansion phase 1) 
- **Task 3**: Cameras 11-15 (Expansion phase 2)
- **Task 4**: Cameras 16-20 (Final deployment)

## Expected Results
- Minimal forgetting compared to FFT methods
- Superior performance to existing CL approaches
- LoRA-G shows best balance between adaptation and stability

## Files
- `cl_trainer.py`: Continual learning training pipeline
- `forgetting_analysis.py`: Catastrophic forgetting analysis
- `task_performance_tracker.py`: Per-task performance monitoring
- `results/`: CL performance curves and final results