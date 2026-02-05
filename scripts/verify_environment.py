"""
Environment Verification Script
检查环境配置是否正确
"""

import sys
import platform


def print_separator(title=""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print("=" * 60)


def check_python():
    """检查Python版本"""
    print_separator("Python Environment")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    version_info = sys.version_info
    if version_info.major == 3 and version_info.minor >= 8:
        print("✓ Python version is compatible (>= 3.8)")
        return True
    else:
        print("✗ Python version must be >= 3.8")
        return False


def check_pytorch():
    """检查PyTorch安装"""
    print_separator("PyTorch Configuration")
    
    try:
        import torch
        print(f"✓ PyTorch Version: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA Available: Yes")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"    Memory: {mem_total:.2f} GB")
        # Check MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"✓ MPS Available: Yes (Apple Silicon)")
            print(f"  Metal Performance Shaders enabled")
        else:
            print(f"⚠ CUDA/MPS Not Available - using CPU")
            print(f"  Warning: Training will be slow on CPU")
        
        # Check backends
        print(f"\nBackends:")
        print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        
        return True
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False


def check_torchvision():
    """检查torchvision"""
    try:
        import torchvision
        print(f"✓ torchvision: {torchvision.__version__}")
        return True
    except ImportError:
        print(f"✗ torchvision not installed")
        return False


def check_core_dependencies():
    """检查核心依赖"""
    print_separator("Core Dependencies")
    
    dependencies = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
    }
    
    all_ok = True
    for module_name, package_name in dependencies.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name}: {version}")
        except ImportError:
            print(f"✗ {package_name} not installed")
            all_ok = False
    
    return all_ok


def check_specialized_dependencies():
    """检查专业依赖"""
    print_separator("Specialized Dependencies")
    
    dependencies = {
        'torch_geometric': 'PyTorch Geometric',
        'transformers': 'Transformers',
        'peft': 'PEFT (LoRA)',
        'networkx': 'NetworkX',
        'tensorboard': 'TensorBoard',
    }
    
    all_ok = True
    for module_name, package_name in dependencies.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name}: {version}")
        except ImportError:
            print(f"⚠ {package_name} not installed (optional)")
    
    return all_ok


def check_project_modules():
    """检查项目模块"""
    print_separator("Project Modules")
    
    modules = ['losses', 'train', 'eval', 'models', 'utils']
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}/ module")
        except ImportError as e:
            print(f"✗ {module}/ module: {e}")
            all_ok = False
    
    return all_ok


def check_system_info():
    """检查系统信息"""
    print_separator("System Information")
    
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"RAM: {mem.total / 1024**3:.2f} GB (Available: {mem.available / 1024**3:.2f} GB)")
    except ImportError:
        print("psutil not installed - cannot check memory")


def test_simple_training():
    """测试简单的训练流程"""
    print_separator("Testing Training Pipeline")
    
    try:
        import torch
        import torch.nn as nn
        
        # Create dummy model and data
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                            else 'cpu')
        
        print(f"Using device: {device}")
        
        model = nn.Linear(10, 2).to(device)
        data = torch.randn(4, 10).to(device)
        target = torch.randint(0, 2, (4,)).to(device)
        
        # Forward pass
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        print(f"✓ Training pipeline test passed")
        print(f"  Loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Training pipeline test failed: {e}")
        return False


def main():
    """主函数"""
    print_separator("Environment Verification Script")
    print("Checking your environment setup for GCN-Transformer Vehicle Re-ID")
    
    results = []
    
    # Run all checks
    results.append(("Python", check_python()))
    results.append(("PyTorch", check_pytorch()))
    results.append(("torchvision", check_torchvision()))
    results.append(("Core Dependencies", check_core_dependencies()))
    results.append(("Specialized Dependencies", check_specialized_dependencies()))
    results.append(("Project Modules", check_project_modules()))
    check_system_info()
    results.append(("Training Pipeline", test_simple_training()))
    
    # Summary
    print_separator("Summary")
    all_passed = all(result for _, result in results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    print_separator()
    
    if all_passed:
        print("🎉 All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("  1. Prepare your dataset: data/776_DataSet/")
        print("  2. Run training: python train.py")
        print("  3. Run evaluation: python eval.py --checkpoint outputs/bot_baseline/best_model.pth")
        return 0
    else:
        print("⚠ Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
