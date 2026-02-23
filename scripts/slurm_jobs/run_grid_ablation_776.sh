#!/bin/bash
# 批量提交VeRi-776的grid scale实验 (8x8, 12x12, 16x16)

PROJECT_DIR="/users/sl3753/scratch/GCN_project"

echo "========================================"
echo "Submitting Grid Scale Experiments (VeRi-776)"
echo "========================================"

for GRID_SIZE in 8x8 12x12 16x16; do
    GRID_H=$(echo $GRID_SIZE | cut -d'x' -f1)
    GRID_W=$(echo $GRID_SIZE | cut -d'x' -f2)
    
    # 确定输出目录（不在这里创建，由Python脚本创建）
    BASE_DIR="${PROJECT_DIR}/outputs/grid_scale_1_1/${GRID_SIZE}/776"
    RUN_IDX=1
    while [ -d "${BASE_DIR}/baseline_run_$(printf '%02d' $RUN_IDX)" ]; do
        RUN_IDX=$((RUN_IDX + 1))
    done
    OUTPUT_DIR="${BASE_DIR}/baseline_run_$(printf '%02d' $RUN_IDX)"
    
    echo "Submitting ${GRID_SIZE} -> ${OUTPUT_DIR}"
    
    # 直接嵌入sbatch命令
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=grid_${GRID_SIZE}
#SBATCH --partition=gpuplus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

cd ${PROJECT_DIR}

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.6.0
source ${PROJECT_DIR}/venv_t4/bin/activate

echo "Grid: ${GRID_H}x${GRID_W}, Output: ${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/train_bot_baseline.py \\
    --config configs/baseline_configs/bot_baseline.yaml \\
    --data_root data/dataset/776_DataSet \\
    --output_dir "${OUTPUT_DIR}" \\
    --grid_h ${GRID_H} \\
    --grid_w ${GRID_W}
EOF
    
    sleep 1
done

echo ""
echo "All jobs submitted! Check with: squeue -u $USER"
