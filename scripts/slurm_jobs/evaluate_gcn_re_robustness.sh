#!/bin/bash
#SBATCH --job-name=gcn_re_robust
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/gcn_re_robustness_%j.out
#SBATCH --error=logs/gcn_re_robustness_%j.err

# GCN + Random Erasing Robustness Evaluation on A40

echo "=========================================="
echo "GCN+RE Occlusion Robustness Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Environment
source /users/sl3753/scratch/GCN_project/venv_t4/bin/activate
cd /users/sl3753/scratch/GCN_project

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Evaluation
echo "Starting robustness evaluation..."
python scripts/testing/evaluate_occlusion_robustness.py \
    --model outputs/bot_gcn_776_Random/best_model.pth \
    --occlusion-dir outputs/occlusion_tests/veri776_query \
    --gallery-dir data/dataset/776_DataSet/image_test \
    --output-dir outputs/robustness_comparison/gcn_re \
    --device cuda

echo ""
echo "Evaluation completed at: $(date)"
echo "=========================================="
