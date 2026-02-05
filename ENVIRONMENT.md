# Environment Setup for GCN-Transformer Vehicle Re-ID

## System Information
- **OS**: macOS (Apple Silicon - ARM64)
- **Python**: 3.12.9 (Anaconda)
- **PyTorch**: 2.9.0
- **Accelerator**: MPS (Metal Performance Shaders)
- **CUDA**: Not Available (macOS)

## Quick Start

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd GCN_transformer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

### Option 2: Using Conda

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate gcn-reid

# Install project
pip install -e .
```

## Detailed Installation Steps

### 1. Prerequisites

**Python Version**: >= 3.8, < 3.13

**System Requirements**:
- CUDA-enabled GPU (NVIDIA) with CUDA 11.7+ **OR**
- Apple Silicon Mac with macOS 12.3+ (MPS support) **OR**
- CPU (slower training)

### 2. PyTorch Installation

#### For NVIDIA GPU (CUDA 11.8)
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

#### For NVIDIA GPU (CUDA 12.1)
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

#### For Apple Silicon (MPS)
```bash
pip install torch==2.1.0 torchvision==0.16.0
```

#### For CPU Only
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python scripts/verify_environment.py
```

Expected output:
```
✓ PyTorch: 2.1.0
✓ CUDA Available: True (or MPS/CPU)
✓ Device: cuda:0 / mps / cpu
✓ All core modules imported successfully
```

## Environment Variables

Create a `.env` file (optional):

```bash
# Data paths
DATA_ROOT=./data/776_DataSet
OUTPUT_DIR=./outputs

# Training settings
CUDA_VISIBLE_DEVICES=0  # GPU selection
OMP_NUM_THREADS=4       # CPU threads

# Logging
WANDB_API_KEY=your_key_here  # If using Weights & Biases
```

## Dependency Versions

### Core Framework
- **PyTorch**: >= 1.12.0 (tested on 2.1.0, 2.9.0)
- **torchvision**: >= 0.13.0
- **CUDA**: 11.7+ (for NVIDIA GPUs)

### Deep Learning
- torch-geometric >= 2.0.0
- transformers >= 4.20.0
- peft >= 0.4.0 (LoRA support)

### Data Processing
- numpy >= 1.21.0
- Pillow >= 8.3.0
- opencv-python >= 4.5.0

### Visualization
- matplotlib >= 3.4.0
- tensorboard >= 2.7.0
- wandb >= 0.12.0 (optional)

## GPU/Driver Information

### NVIDIA GPU Users

Check your setup:
```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

Required:
- **NVIDIA Driver**: >= 470.x
- **CUDA Toolkit**: 11.7+ or 12.1+
- **cuDNN**: 8.x

### Apple Silicon Users

Check MPS support:
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Required:
- **macOS**: 12.3 or later
- **PyTorch**: 1.12.0 or later

## Known Issues & Solutions

### Issue 1: CUDA Out of Memory
```bash
# Reduce batch size in config
vim configs/baseline_configs/bot_baseline.yaml
# Change BATCH_SIZE from 64 to 32 or 16
```

### Issue 2: torch-geometric Installation Fails
```bash
# Install PyTorch first, then torch-geometric
pip install torch torchvision
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### Issue 3: MPS Backend Not Available
```bash
# Update macOS and PyTorch
pip install --upgrade torch torchvision
```

### Issue 4: Import Error for transformers/peft
```bash
# Install specific versions
pip install transformers==4.30.0 peft==0.4.0
```

## Testing Your Environment

Run comprehensive tests:

```bash
# Test core functionality
python -m pytest tests/

# Test training pipeline
python train.py --config configs/baseline_configs/bot_baseline.yaml --epochs 1

# Test evaluation
python eval.py --checkpoint outputs/bot_baseline/best_model.pth
```

## Performance Optimization

### For CUDA Users
```bash
# Enable TF32 for faster training (Ampere GPUs)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Enable cuDNN benchmark
# (automatically enabled in our trainer)
```

### For Apple Silicon Users
```bash
# MPS is automatically used when available
# No additional configuration needed
```

### For CPU Users
```bash
# Set number of threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## Docker Support (Optional)

```bash
# Build Docker image
docker build -t gcn-reid:latest .

# Run container
docker run --gpus all -v $(pwd):/workspace gcn-reid:latest
```

## Development Environment

For development, install additional tools:

```bash
pip install -r requirements-dev.txt

# Code formatting
black .
isort .

# Linting
flake8 .

# Type checking
mypy .
```

## Continuous Integration

Our CI/CD pipeline tests against:
- **Python**: 3.8, 3.9, 3.10, 3.11
- **PyTorch**: 1.12.0, 2.0.0, 2.1.0
- **CUDA**: 11.8, 12.1
- **OS**: Ubuntu 20.04, macOS 12+

## Support & Contact

For environment setup issues:
1. Check [GitHub Issues](your-repo/issues)
2. Read the [Troubleshooting Guide](docs/troubleshooting.md)
3. Open a new issue with your environment details

---

**Last Updated**: 2026-01-28  
**Tested On**: macOS (Apple Silicon), Ubuntu 20.04 (CUDA 11.8), Windows 11 (CUDA 12.1)
