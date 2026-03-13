#!/bin/bash
# ============================================================
# run_vehicleid_ablation_study.sh
# 批量提交 VehicleID 数据集的消融实验训练任务
# ============================================================

#SBATCH --job-name=vehicleid_abl
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/vehicleid_abl_%A_%a.out
#SBATCH --error=logs/vehicleid_abl_%A_%a.err
#SBATCH --array=1-8

# 配置列表（8个实验）
CONFIGS=(
    "abl_vehicleid_cnn_gcn_4nb_l1"
    "abl_vehicleid_cnn_gcn_4nb_l2"
    "abl_vehicleid_cnn_gcn_4nb_l3"
    "abl_vehicleid_cnn_gcn_knn_l1"
    "abl_vehicleid_vit_gcn_4nb_l1"
    "abl_vehicleid_vit_gcn_4nb_l2"
    "abl_vehicleid_vit_gcn_4nb_l3"
    "abl_vehicleid_vit_gcn_knn_l1"
)

# 获取当前任务的配置
CONFIG_NAME=${CONFIGS[$SLURM_ARRAY_TASK_ID-1]}
CONFIG_FILE="configs/gcn_transformer_configs/${CONFIG_NAME}.yaml"

echo "================================================================"
echo "🚀 VehicleID Ablation Study - Task ${SLURM_ARRAY_TASK_ID}/8"
echo "================================================================"
echo "📋 Config: ${CONFIG_NAME}"
echo "📁 Config File: ${CONFIG_FILE}"
echo "🖥️  Node: $(hostname)"
echo "🎯 GPU: ${CUDA_VISIBLE_DEVICES}"
echo "⏰ Start Time: $(date)"
echo "================================================================"

# 激活环境
source venv_t4/bin/activate

# 设置 GPU（通过环境变量）
export CUDA_VISIBLE_DEVICES=0

# 运行训练
python scripts/training/train_bot_gcn.py \
    --config ${CONFIG_FILE}

EXIT_CODE=$?

echo "================================================================"
echo "⏱️  End Time: $(date)"
echo "📊 Exit Code: ${EXIT_CODE}"
echo "================================================================"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
