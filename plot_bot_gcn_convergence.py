#!/usr/bin/env python3
"""
绘制BoT-GCN训练收敛曲线
支持VeRi-776和VehicleID数据集
"""

import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# 数据集配置
DATASET_CONFIG = {
    'veri776': {
        'name': 'VeRi-776',
        'log_path': 'outputs/bot_gcn_776_v2/training.log',
        'output_path': 'outputs/bot_gcn_776_v2/training_curves.png',
        'baseline_mAP': 0.6449,
        'baseline_rank1': 0.8897
    },
    'vehicleid': {
        'name': 'VehicleID',
        'log_path': 'outputs/bot_gcn_vehicleid_h100/training.log',
        'output_path': 'outputs/bot_gcn_vehicleid_h100/training_curves.png',
        'baseline_mAP': 0.7628,
        'baseline_rank1': 0.7084
    }
}

def parse_log(log_file):
    """解析训练日志"""
    epochs = []
    train_losses = []
    train_accs = []
    eval_maps = []
    eval_rank1 = []
    eval_epochs = []

    with open(log_file, 'r') as f:
        for line in f:
            # 解析训练指标
            train_match = re.search(r'Epoch (\d+) Training - Loss: ([\d.]+), ID_Acc: ([\d.]+)%', line)
            if train_match:
                epoch = int(train_match.group(1))
                loss = float(train_match.group(2))
                acc = float(train_match.group(3))
                epochs.append(epoch)
                train_losses.append(loss)
                train_accs.append(acc)
            
            # 解析评估指标
            eval_match = re.search(r'Epoch (\d+) Evaluation - mAP: ([\d.]+), Rank-1: ([\d.]+)', line)
            if eval_match:
                epoch = int(eval_match.group(1))
                mAP = float(eval_match.group(2))
                rank1 = float(eval_match.group(3))
                eval_epochs.append(epoch)
                eval_maps.append(mAP)
                eval_rank1.append(rank1)
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'eval_epochs': eval_epochs,
        'eval_maps': eval_maps,
        'eval_rank1': eval_rank1
    }

def plot_convergence(data, config, dataset_name):
    """绘制收敛曲线"""
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'BoT-GCN Training Convergence on {dataset_name}', fontsize=16, fontweight='bold')

    # 1. Training Loss
    ax1 = axes[0, 0]
    ax1.plot(data['epochs'], data['train_losses'], 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # 2. Training Accuracy
    ax2 = axes[0, 1]
    ax2.plot(data['epochs'], data['train_accs'], 'g-', linewidth=2, label='Training ID Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training Accuracy Curve', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    # 3. Evaluation mAP
    ax3 = axes[1, 0]
    ax3.plot(data['eval_epochs'], data['eval_maps'], 'r-o', linewidth=2, markersize=6, label='Evaluation mAP')
    ax3.axhline(y=config['baseline_mAP'], color='orange', linestyle='--', linewidth=1.5, 
                label=f'BoT Baseline ({config["baseline_mAP"]:.4f})')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('mAP', fontsize=12)
    ax3.set_title('Evaluation mAP (Query vs Gallery)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)

    # 添加最终mAP标注
    if data['eval_maps']:
        final_mAP = data['eval_maps'][-1]
        ax3.annotate(f'Final: {final_mAP:.4f}', 
                     xy=(data['eval_epochs'][-1], final_mAP), 
                     xytext=(data['eval_epochs'][-1]-15, final_mAP+0.02),
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'))

    # 4. Evaluation Rank-1
    ax4 = axes[1, 1]
    ax4.plot(data['eval_epochs'], data['eval_rank1'], 'm-o', linewidth=2, markersize=6, label='Evaluation Rank-1')
    ax4.axhline(y=config['baseline_rank1'], color='orange', linestyle='--', linewidth=1.5, 
                label=f'BoT Baseline ({config["baseline_rank1"]:.4f})')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Rank-1 Accuracy', fontsize=12)
    ax4.set_title('Evaluation Rank-1 Accuracy', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11)

    # 添加最终Rank-1标注
    if data['eval_rank1']:
        final_rank1 = data['eval_rank1'][-1]
        ax4.annotate(f'Final: {final_rank1:.4f}', 
                     xy=(data['eval_epochs'][-1], final_rank1), 
                     xytext=(data['eval_epochs'][-1]-15, final_rank1-0.03),
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='magenta'))

    plt.tight_layout()
    return fig

def print_summary(data, config, dataset_name):
    """打印训练总结"""
    print("\n" + "="*60)
    print(f"📊 BoT-GCN 训练总结 ({dataset_name})")
    print("="*60)
    print(f"训练轮数: {max(data['epochs'])} epochs")
    print(f"最终训练Loss: {data['train_losses'][-1]:.4f}")
    print(f"最终训练准确率: {data['train_accs'][-1]:.2f}%")
    
    best_mAP = max(data['eval_maps'])
    best_mAP_epoch = data['eval_epochs'][data['eval_maps'].index(best_mAP)]
    best_rank1 = max(data['eval_rank1'])
    best_rank1_epoch = data['eval_epochs'][data['eval_rank1'].index(best_rank1)]
    
    print(f"最佳mAP: {best_mAP:.4f} (Epoch {best_mAP_epoch})")
    print(f"最佳Rank-1: {best_rank1:.4f} (Epoch {best_rank1_epoch})")
    print(f"\n🎯 vs BoT Baseline:")
    
    mAP_improvement = best_mAP - config['baseline_mAP']
    rank1_improvement = best_rank1 - config['baseline_rank1']
    
    print(f"   mAP提升: {mAP_improvement:.4f} ({mAP_improvement / config['baseline_mAP'] * 100:+.2f}%)")
    print(f"   Rank-1提升: {rank1_improvement:.4f} ({rank1_improvement / config['baseline_rank1'] * 100:+.2f}%)")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='绘制BoT-GCN训练收敛曲线')
    parser.add_argument('--dataset', type=str, default='veri776', choices=['veri776', 'vehicleid'],
                        help='数据集选择: veri776 或 vehicleid')
    parser.add_argument('--log_path', type=str, default=None,
                        help='自定义日志路径（可选）')
    parser.add_argument('--output_path', type=str, default=None,
                        help='自定义输出路径（可选）')
    args = parser.parse_args()
    
    # 获取配置
    config = DATASET_CONFIG[args.dataset]
    dataset_name = config['name']
    
    # 使用自定义路径或默认路径
    project_root = Path(__file__).parent
    log_path = args.log_path if args.log_path else project_root / config['log_path']
    output_path = args.output_path if args.output_path else project_root / config['output_path']
    
    # 检查日志文件是否存在
    if not Path(log_path).exists():
        print(f"❌ 错误: 日志文件不存在: {log_path}")
        return
    
    print(f"📖 正在解析 {dataset_name} 训练日志...")
    data = parse_log(log_path)
    
    print(f"📊 正在生成收敛曲线...")
    fig = plot_convergence(data, config, dataset_name)
    
    # 保存图表
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存到: {output_path}")
    
    # 打印总结
    print_summary(data, config, dataset_name)

if __name__ == '__main__':
    main()
