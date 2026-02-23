#!/bin/bash
#SBATCH --job-name=bot_776
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# 切换到项目目录（需要在作业提交前确定路径）
cd /users/sl3753/scratch/GCN_project

# 注意：不在这里创建输出目录，由Python脚本统一管理
# 这样避免Shell和Python之间的竞态条件

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
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
echo "========================================"
echo "Starting BoT-Baseline Training (VeRi-776)"
echo "========================================"

export SLURM_LOG_FILE="${OUTPUT_DIR}/slurm_${SLURM_JOB_ID}.log"

python scripts/train_bot_baseline.py \
    --config configs/baseline_configs/bot_baseline.yaml \
    --data_root data/dataset/776_DataSet \
    --output_dir "${OUTPUT_DIR}" \
    --grid_h 0 \
    --grid_w 0

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
