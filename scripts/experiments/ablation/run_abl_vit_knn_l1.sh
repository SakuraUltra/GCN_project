#!/bin/bash
#SBATCH --job-name=abl_vit_knn_l1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/abl_vit_gcn_knn_l1_%j.out
#SBATCH --error=logs/abl_vit_gcn_knn_l1_%j.err

# ABL-15: ViT-Base Native768 + GCN, Edge=kNN(k=8), Layers=1
echo "=========================================="
echo "Job: ABL-15 (ViT+GCN kNN L1)"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

cd /users/sl3753/scratch/GCN_project
source venv_t4/bin/activate

python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_vit_gcn_knn_l1.yaml

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
