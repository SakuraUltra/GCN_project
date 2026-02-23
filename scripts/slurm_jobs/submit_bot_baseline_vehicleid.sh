#!/bin/bash
#SBATCH --job-name=bot_vid
#SBATCH --partition=gpuplus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# 切换到项目目录
cd /users/sl3753/scratch/GCN_project

# 确定输出目录（与train_bot_baseline.py逻辑一致）
BASE_DIR="outputs/bot_baseline_1_1/VehicleID"
RUN_IDX=1
while [ -d "${BASE_DIR}/baseline_run_$(printf '%02d' $RUN_IDX)" ]; do
    RUN_IDX=$((RUN_IDX + 1))
done
OUTPUT_DIR="${BASE_DIR}/baseline_run_$(printf '%02d' $RUN_IDX)"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "Dataset: VehicleID"
echo "Output Dir: $OUTPUT_DIR"
echo "========================================"

# 加载模块
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.6.0

# 激活虚拟环境
source /users/sl3753/scratch/GCN_project/venv_t4/bin/activate

# 验证GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 验证PyTorch
echo ""
echo "PyTorch Information:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 运行训练
echo ""
echo "=========================================="
echo "Starting BoT-Baseline Training (VehicleID)"
echo "=========================================="

export SLURM_LOG_FILE="${OUTPUT_DIR}/slurm_${SLURM_JOB_ID}.log"

python scripts/train_bot_baseline.py \
    --config configs/baseline_configs/bot_baseline.yaml \
    --data_root data/dataset/VehicleID_V1.0 \
    --output_dir "${OUTPUT_DIR}" \
    --grid_h 0 \
    --grid_w 0

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
