#!/usr/bin/env python3
"""
生成三模型对比分析: Baseline vs GCN vs GCN+RE
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取三个模型的结果
baseline = pd.read_csv('outputs/robustness_comparison/baseline/occlusion_robustness_results.csv')
gcn = pd.read_csv('outputs/robustness_comparison/gcn/occlusion_robustness_results.csv')
gcn_re = pd.read_csv('outputs/robustness_comparison/gcn_re/occlusion_robustness_results.csv')

# 重命名列以便合并
baseline = baseline.rename(columns={'mAP': 'baseline_mAP', 'rank1': 'baseline_rank1'})
gcn = gcn.rename(columns={'mAP': 'gcn_mAP', 'rank1': 'gcn_rank1'})
gcn_re = gcn_re.rename(columns={'mAP': 'gcn_re_mAP', 'rank1': 'gcn_re_rank1'})

# 合并数据
comparison = baseline[['config_name', 'occlusion_ratio', 'occlusion_type', 'baseline_mAP', 'baseline_rank1']].copy()
comparison = comparison.merge(gcn[['config_name', 'gcn_mAP', 'gcn_rank1']], on='config_name')
comparison = comparison.merge(gcn_re[['config_name', 'gcn_re_mAP', 'gcn_re_rank1']], on='config_name')

# 计算改进
comparison['gcn_improvement'] = (comparison['gcn_mAP'] - comparison['baseline_mAP']) * 100
comparison['gcn_re_improvement'] = (comparison['gcn_re_mAP'] - comparison['baseline_mAP']) * 100
comparison['re_effect'] = (comparison['gcn_re_mAP'] - comparison['gcn_mAP']) * 100

# 保存对比数据
output_dir = Path('outputs/robustness_comparison')
comparison.to_csv(output_dir / 'three_model_comparison.csv', index=False)
print(f"✓ Saved comparison data to {output_dir / 'three_model_comparison.csv'}")

# ============================================================================
# 生成综合报告
# ============================================================================
report_lines = []
report_lines.append("=" * 120)
report_lines.append("THREE-MODEL ROBUSTNESS COMPARISON REPORT")
report_lines.append("Baseline vs GCN (no Random Erasing) vs GCN+RE (with Random Erasing)")
report_lines.append("=" * 120)
report_lines.append("")

# 模型配置信息
report_lines.append("MODEL CONFIGURATIONS")
report_lines.append("-" * 120)
report_lines.append("1. Baseline: ResNet50-IBN + GAP")
report_lines.append("   - Clean mAP: 68.69%")
report_lines.append("   - Data Augmentation: None (除标准transforms)")
report_lines.append("")
report_lines.append("2. GCN (no RE): Baseline + 4x4 Grid GCN + Mean Pooling + Concat Fusion")
report_lines.append("   - Clean mAP: 78.77%")
report_lines.append("   - Data Augmentation: None (除标准transforms)")
report_lines.append("   - Improvement over Baseline: +10.08%")
report_lines.append("")
report_lines.append("3. GCN+RE: Same as GCN + Random Erasing augmentation")
report_lines.append("   - Clean mAP: 79.25%")
report_lines.append("   - Data Augmentation: Random Erasing (p=0.5, area=2-40%, ratio=0.3-3.33)")
report_lines.append("   - Improvement over Baseline: +10.56%")
report_lines.append("   - Improvement over GCN: +0.48%")
report_lines.append("")
report_lines.append("=" * 120)

# 详细对比表
report_lines.append("DETAILED PERFORMANCE COMPARISON")
report_lines.append("=" * 120)
report_lines.append(f"{'Config':<15} {'Ratio':>6} {'Type':<8} {'Baseline':>10} {'GCN':>10} {'GCN+RE':>10} "
                   f"{'GCN Imp':>10} {'RE Effect':>10}")
report_lines.append("-" * 120)

for _, row in comparison.iterrows():
    report_lines.append(
        f"{row['config_name']:<15} {row['occlusion_ratio']*100:>5.0f}% {row['occlusion_type']:<8} "
        f"{row['baseline_mAP']*100:>9.2f}% {row['gcn_mAP']*100:>9.2f}% {row['gcn_re_mAP']*100:>9.2f}% "
        f"{row['gcn_improvement']:>9.2f}% {row['re_effect']:>9.2f}%"
    )

report_lines.append("")
report_lines.append("=" * 120)

# Grid遮挡专项分析
report_lines.append("GRID OCCLUSION ANALYSIS")
report_lines.append("=" * 120)
grid_data = comparison[comparison['occlusion_type'] == 'grid'].copy()

report_lines.append("\nGrid Occlusion Performance Summary:")
report_lines.append("-" * 120)
report_lines.append(f"{'Ratio':>8} {'Baseline':>12} {'GCN (no RE)':>15} {'GCN+RE':>15} {'GCN Imp':>12} {'RE Effect':>12}")
report_lines.append("-" * 120)
for _, row in grid_data.iterrows():
    report_lines.append(
        f"{row['occlusion_ratio']*100:>7.0f}% {row['baseline_mAP']*100:>11.2f}% "
        f"{row['gcn_mAP']*100:>14.2f}% {row['gcn_re_mAP']*100:>14.2f}% "
        f"{row['gcn_improvement']:>11.2f}% {row['re_effect']:>11.2f}%"
    )

report_lines.append("\nKey Findings:")
report_lines.append("  • Grid 10%: Random Erasing provides +6.65% improvement over GCN")
report_lines.append("  • Grid 20%: Random Erasing provides +4.24% improvement (but still low at 10.16%)")
report_lines.append("  • Grid 30%: Random Erasing provides +1.70% improvement (catastrophic failure at 5.24%)")
report_lines.append("  • Conclusion: Random Erasing helps but CANNOT solve Grid occlusion fundamentally")
report_lines.append("")

# 统计分析
report_lines.append("=" * 120)
report_lines.append("STATISTICAL SUMMARY")
report_lines.append("=" * 120)

non_grid = comparison[comparison['occlusion_type'] != 'grid'].copy()

report_lines.append("\nNon-Grid Occlusions (15 configurations):")
report_lines.append("-" * 120)
report_lines.append(f"  Average Baseline mAP:    {non_grid['baseline_mAP'].mean()*100:.2f}%")
report_lines.append(f"  Average GCN mAP:         {non_grid['gcn_mAP'].mean()*100:.2f}%")
report_lines.append(f"  Average GCN+RE mAP:      {non_grid['gcn_re_mAP'].mean()*100:.2f}%")
report_lines.append(f"  GCN Improvement:         {non_grid['gcn_improvement'].mean():+.2f}%")
report_lines.append(f"  GCN+RE Improvement:      {non_grid['gcn_re_improvement'].mean():+.2f}%")
report_lines.append(f"  Random Erasing Effect:   {non_grid['re_effect'].mean():+.2f}%")

report_lines.append("\nGrid Occlusions (3 configurations):")
report_lines.append("-" * 120)
report_lines.append(f"  Average Baseline mAP:    {grid_data['baseline_mAP'].mean()*100:.2f}%")
report_lines.append(f"  Average GCN mAP:         {grid_data['gcn_mAP'].mean()*100:.2f}%")
report_lines.append(f"  Average GCN+RE mAP:      {grid_data['gcn_re_mAP'].mean()*100:.2f}%")
report_lines.append(f"  GCN Improvement:         {grid_data['gcn_improvement'].mean():+.2f}%")
report_lines.append(f"  GCN+RE Improvement:      {grid_data['gcn_re_improvement'].mean():+.2f}%")
report_lines.append(f"  Random Erasing Effect:   {grid_data['re_effect'].mean():+.2f}%")

report_lines.append("\nOverall (18 configurations):")
report_lines.append("-" * 120)
all_occluded = comparison[comparison['occlusion_ratio'] > 0].copy()
report_lines.append(f"  Average Baseline mAP:    {all_occluded['baseline_mAP'].mean()*100:.2f}%")
report_lines.append(f"  Average GCN mAP:         {all_occluded['gcn_mAP'].mean()*100:.2f}%")
report_lines.append(f"  Average GCN+RE mAP:      {all_occluded['gcn_re_mAP'].mean()*100:.2f}%")
report_lines.append(f"  GCN Improvement:         {all_occluded['gcn_improvement'].mean():+.2f}%")
report_lines.append(f"  GCN+RE Improvement:      {all_occluded['gcn_re_improvement'].mean():+.2f}%")
report_lines.append(f"  Random Erasing Effect:   {all_occluded['re_effect'].mean():+.2f}%")

report_lines.append("")
report_lines.append("=" * 120)
report_lines.append("CONCLUSIONS")
report_lines.append("=" * 120)
report_lines.append("\n1. GCN Architecture Impact:")
report_lines.append("   • Improves non-Grid occlusions by +6.35% on average")
report_lines.append("   • Spatial relationship modeling helps with continuous occlusions")
report_lines.append("   • Consistent improvements across center/top/bottom/left/right occlusions")
report_lines.append("")
report_lines.append("2. Random Erasing Augmentation Impact:")
report_lines.append("   • Provides +0.88% additional improvement on non-Grid occlusions")
report_lines.append("   • Significant help on Grid 10% occlusion (+6.65%)")
report_lines.append("   • Moderate help on Grid 20% occlusion (+4.24%)")
report_lines.append("   • Limited help on Grid 30% occlusion (+1.70%)")
report_lines.append("")
report_lines.append("3. Grid Occlusion Challenge:")
report_lines.append("   • Fundamentally different from continuous occlusions")
report_lines.append("   • Scattered disruption breaks spatial coherence")
report_lines.append("   • All three models fail at 20-30% Grid occlusion (<11% mAP)")
report_lines.append("   • Requires specialized approaches beyond data augmentation")
report_lines.append("")
report_lines.append("4. Recommendations:")
report_lines.append("   • Use GCN+RE for general robustness (best overall performance)")
report_lines.append("   • Random Erasing is beneficial and should be enabled")
report_lines.append("   • Grid occlusion needs domain-specific solutions:")
report_lines.append("     - Attention mechanisms to focus on visible patches")
report_lines.append("     - Part-based models to handle fragmentary information")
report_lines.append("     - Specialized training strategies for scattered occlusions")
report_lines.append("")
report_lines.append("=" * 120)

# 保存报告
report_path = output_dir / 'THREE_MODEL_COMPARISON_REPORT.txt'
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))
print(f"✓ Saved comprehensive report to {report_path}")

# ============================================================================
# 生成可视化图表
# ============================================================================
print("\nGenerating visualizations...")

# 创建大型综合图表
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 按遮挡类型的性能对比 (不含Grid)
ax1 = fig.add_subplot(gs[0, :2])
non_grid_types = comparison[comparison['occlusion_type'] != 'grid'].copy()
for occ_type in ['center', 'top', 'bottom', 'left', 'right']:
    type_data = non_grid_types[non_grid_types['occlusion_type'] == occ_type]
    ratios = type_data['occlusion_ratio'] * 100
    ax1.plot(ratios, type_data['baseline_mAP'] * 100, 'o--', label=f'{occ_type.capitalize()}-Baseline', alpha=0.6)
    ax1.plot(ratios, type_data['gcn_mAP'] * 100, 's-', label=f'{occ_type.capitalize()}-GCN', alpha=0.8)
    ax1.plot(ratios, type_data['gcn_re_mAP'] * 100, '^-', label=f'{occ_type.capitalize()}-GCN+RE', linewidth=2)

ax1.set_xlabel('Occlusion Ratio (%)', fontsize=12)
ax1.set_ylabel('mAP (%)', fontsize=12)
ax1.set_title('Non-Grid Occlusion Performance: Three Models', fontsize=14, fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=3)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(40, 85)

# 2. Grid遮挡专项对比
ax2 = fig.add_subplot(gs[0, 2])
grid_data_sorted = grid_data.sort_values('occlusion_ratio')
x = np.arange(len(grid_data_sorted))
width = 0.25

bars1 = ax2.bar(x - width, grid_data_sorted['baseline_mAP'] * 100, width, label='Baseline', color='#1f77b4', alpha=0.8)
bars2 = ax2.bar(x, grid_data_sorted['gcn_mAP'] * 100, width, label='GCN (no RE)', color='#ff7f0e', alpha=0.8)
bars3 = ax2.bar(x + width, grid_data_sorted['gcn_re_mAP'] * 100, width, label='GCN+RE', color='#2ca02c', alpha=0.8)

ax2.set_xlabel('Grid Occlusion Ratio', fontsize=11)
ax2.set_ylabel('mAP (%)', fontsize=11)
ax2.set_title('Grid Occlusion:\nCatastrophic Failure', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f"{int(r*100)}%" for r in grid_data_sorted['occlusion_ratio']])
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# 3. Random Erasing效果分析 (非Grid)
ax3 = fig.add_subplot(gs[1, 0])
non_grid_by_ratio = non_grid.groupby('occlusion_ratio').agg({
    're_effect': 'mean'
}).reset_index()

ax3.bar(non_grid_by_ratio['occlusion_ratio'] * 100, non_grid_by_ratio['re_effect'], 
        color='#2ca02c', alpha=0.7, edgecolor='black')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax3.set_xlabel('Occlusion Ratio (%)', fontsize=11)
ax3.set_ylabel('RE Effect (mAP %)', fontsize=11)
ax3.set_title('Random Erasing Effect\n(Non-Grid Average)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Random Erasing效果分析 (Grid)
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(grid_data_sorted['occlusion_ratio'] * 100, grid_data_sorted['re_effect'],
        color='#d62728', alpha=0.7, edgecolor='black')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax4.set_xlabel('Grid Occlusion Ratio (%)', fontsize=11)
ax4.set_ylabel('RE Effect (mAP %)', fontsize=11)
ax4.set_title('Random Erasing Effect\n(Grid Only)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, (ratio, effect) in enumerate(zip(grid_data_sorted['occlusion_ratio'] * 100, grid_data_sorted['re_effect'])):
    ax4.text(ratio, effect + 0.2, f'{effect:.2f}%', ha='center', fontsize=9)

# 5. 整体改进对比
ax5 = fig.add_subplot(gs[1, 2])
categories = ['Non-Grid\n(15 configs)', 'Grid\n(3 configs)', 'Overall\n(18 configs)']
gcn_improvements = [
    non_grid['gcn_improvement'].mean(),
    grid_data['gcn_improvement'].mean(),
    all_occluded['gcn_improvement'].mean()
]
re_effects = [
    non_grid['re_effect'].mean(),
    grid_data['re_effect'].mean(),
    all_occluded['re_effect'].mean()
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax5.bar(x - width/2, gcn_improvements, width, label='GCN Improvement', color='#ff7f0e', alpha=0.8)
bars2 = ax5.bar(x + width/2, re_effects, width, label='RE Effect', color='#2ca02c', alpha=0.8)

ax5.set_ylabel('Improvement (mAP %)', fontsize=11)
ax5.set_title('Average Improvements\nby Category', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(categories, fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

# 6. 三模型综合性能曲线 (所有遮挡类型平均)
ax6 = fig.add_subplot(gs[2, :])
by_ratio = comparison.groupby('occlusion_ratio').agg({
    'baseline_mAP': 'mean',
    'gcn_mAP': 'mean',
    'gcn_re_mAP': 'mean'
}).reset_index()

ax6.plot(by_ratio['occlusion_ratio'] * 100, by_ratio['baseline_mAP'] * 100, 
         'o-', label='Baseline', linewidth=3, markersize=10, color='#1f77b4')
ax6.plot(by_ratio['occlusion_ratio'] * 100, by_ratio['gcn_mAP'] * 100,
         's-', label='GCN (no RE)', linewidth=3, markersize=10, color='#ff7f0e')
ax6.plot(by_ratio['occlusion_ratio'] * 100, by_ratio['gcn_re_mAP'] * 100,
         '^-', label='GCN+RE', linewidth=3, markersize=10, color='#2ca02c')

ax6.set_xlabel('Occlusion Ratio (%)', fontsize=12)
ax6.set_ylabel('Average mAP (%)', fontsize=12)
ax6.set_title('Overall Performance Comparison (Average Across All Occlusion Types)', fontsize=14, fontweight='bold')
ax6.legend(fontsize=12)
ax6.grid(True, alpha=0.3)
ax6.set_xticks([0, 10, 20, 30])

# 添加数值标签
for ratio, b_map, g_map, gr_map in zip(by_ratio['occlusion_ratio'] * 100, 
                                         by_ratio['baseline_mAP'] * 100,
                                         by_ratio['gcn_mAP'] * 100,
                                         by_ratio['gcn_re_mAP'] * 100):
    ax6.text(ratio, b_map - 2, f'{b_map:.1f}', ha='center', fontsize=9, color='#1f77b4')
    ax6.text(ratio, g_map + 2, f'{g_map:.1f}', ha='center', fontsize=9, color='#ff7f0e')
    ax6.text(ratio, gr_map + 2, f'{gr_map:.1f}', ha='center', fontsize=9, color='#2ca02c')

plt.suptitle('Three-Model Robustness Comparison: Baseline vs GCN vs GCN+RE', 
             fontsize=16, fontweight='bold', y=0.995)

# 保存图表
vis_path = output_dir / 'three_model_comparison_visualization.png'
plt.savefig(vis_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved comprehensive visualization to {vis_path}")

print("\n" + "=" * 80)
print("✅ Three-model comparison analysis completed!")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  1. {output_dir / 'three_model_comparison.csv'}")
print(f"  2. {output_dir / 'THREE_MODEL_COMPARISON_REPORT.txt'}")
print(f"  3. {output_dir / 'three_model_comparison_visualization.png'}")
