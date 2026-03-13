#!/bin/bash
#SBATCH --job-name=abl_vit_4nb_l3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/abl_vit_gcn_4nb_l3_%j.out
#SBATCH --error=logs/abl_vit_gcn_4nb_l3_%j.err

# ABL-12: ViT-Base Native768 + GCN, Edge=4-neighbor, Layers=3
echo "=========================================="
echo "Job: ABL-12 (ViT+GCN 4nb L3)"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

cd /users/sl3753/scratch/GCN_project
source venv_t4/bin/activate

python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_vit_gcn_4nb_l3.yaml

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
