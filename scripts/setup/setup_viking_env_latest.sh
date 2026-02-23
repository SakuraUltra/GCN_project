#!/bin/bash
# ============================================
# Viking环境安装脚本 (新一代GPU专用)
# PyTorch 2.10.0 + CUDA 13.0
# 适用于: A100, H100, V100, RTX 40系列等
# ============================================

echo "=========================================="
echo "Viking GCN项目环境配置 (最新GPU)"
echo "PyTorch 2.10.0 + CUDA 12.6"
echo "=========================================="

# 1. 加载基础Python模块
echo "1. 加载Python 3.11.3..."
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.6.0  # 加载CUDA 12.6运行时

# 2. 创建Python虚拟环境
echo "2. 创建虚拟环境..."
cd /users/sl3753/scratch/GCN_project
python -m venv venv_latest

# 3. 激活虚拟环境
echo "3. 激活虚拟环境..."
source venv_latest/bin/activate

# 4. 升级pip
echo "4. 升级pip..."
pip install --upgrade pip

# 5. 安装PyTorch 2.10.0 (CUDA 12.6)
echo "5. 安装PyTorch 2.10.0 (CUDA 12.6)..."
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 6. 安装PyTorch Geometric
echo "6. 安装PyTorch Geometric..."
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.10.0+cu126.html

# 7. 安装核心依赖
echo "7. 安装核心依赖..."
pip install numpy scipy scikit-learn pandas
pip install pillow opencv-python
pip install matplotlib seaborn
pip install pyyaml tqdm
pip install networkx lxml

# 8. 安装Transformers和PEFT
echo "8. 安装Transformers和PEFT..."
pip install transformers>=4.30.0
pip install peft>=0.4.0

# 9. 安装其他工具
echo "9. 安装其他工具..."
pip install timm>=0.6.0
pip install xmltodict
pip install tensorboard
pip install gpustat
pip install wandb

# 10. 安装开发工具（可选）
echo "10. 安装开发工具..."
pip install pytest black flake8 isort
pip install jupyter ipykernel plotly

# 11. 安装项目
echo "11. 安装项目..."
pip install -e .

echo ""
echo "=========================================="
echo "验证安装..."
echo "=========================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import torch; print(f'GPU Compute Capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

echo ""
echo "=========================================="
echo "✅ 最新GPU环境配置完成！"
echo "=========================================="
echo ""
echo "激活方法:"
echo "  module load Python/3.11.3-GCCcore-12.3.0 CUDA/12.6.0"
echo "  source /users/sl3753/scratch/GCN_project/venv_latest/bin/activate"
echo ""
echo "验证环境:"
echo "  python scripts/verify_environment.py"
