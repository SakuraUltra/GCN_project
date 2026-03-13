#!/bin/bash
#SBATCH --job-name=vid_occ_all
#SBATCH --output=/users/sl3753/scratch/GCN_project/logs/vid_occlusion_all_%j.out
#SBATCH --error=/users/sl3753/scratch/GCN_project/logs/vid_occlusion_all_%j.err
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 环境设置
cd /users/sl3753/scratch/GCN_project
source venv_t4/bin/activate

# GPU 信息
nvidia-smi

echo ""
echo "=========================================="
echo "Starting VehicleID Occlusion Evaluation"
echo "All 8 models (4 CNN + 4 ViT)"
echo "Test Size: small (800 IDs)"
echo "Occlusion Levels: 0% - 30% (11 levels)"
echo "=========================================="
echo ""

# 运行评估（small test set）
python scripts/testing/evaluate_occlusion_vehicleid.py \
    --dataset_root data/dataset/VehicleID_V1.0 \
    --output_dir outputs/ablation_vehicleID_occlusion_results \
    --test_size small \
    --device cuda:0

echo ""
echo "=========================================="
echo "VehicleID Occlusion Test Completed!"
echo "End Time: $(date)"
echo "=========================================="
