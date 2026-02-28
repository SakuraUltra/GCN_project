#!/usr/bin/env python3
"""
Detailed analysis of robustness patterns - why GCN fails on edge occlusions
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load results
baseline_df = pd.read_csv('outputs/robustness_comparison/baseline/occlusion_robustness_results.csv')
gcn_df = pd.read_csv('outputs/robustness_comparison/gcn/occlusion_robustness_results.csv')

# Merge
merged_df = baseline_df.merge(
    gcn_df, 
    on=['config_name', 'occlusion_ratio', 'occlusion_type'],
    suffixes=('_baseline', '_gcn')
)

# Calculate improvements
merged_df['mAP_improvement'] = ((merged_df['mAP_gcn'] - merged_df['mAP_baseline']) / merged_df['mAP_baseline'] * 100)
merged_df['rank1_improvement'] = ((merged_df['rank1_gcn'] - merged_df['rank1_baseline']) / merged_df['rank1_baseline'] * 100)

# Filter out 0% occlusion for analysis
occluded_df = merged_df[merged_df['occlusion_ratio'] > 0].copy()

# Create comprehensive analysis figure
fig = plt.figure(figsize=(20, 12))

# 1. Degradation curves by occlusion type (mAP)
ax1 = plt.subplot(2, 4, 1)
for occ_type in ['center', 'top', 'bottom', 'left', 'right', 'grid']:
    type_data = merged_df[merged_df['occlusion_type'] == occ_type].sort_values('occlusion_ratio')
    ax1.plot(type_data['occlusion_ratio'], type_data['mAP_baseline'], 
             marker='o', linestyle='--', alpha=0.6, label=f'Baseline-{occ_type}')
    ax1.plot(type_data['occlusion_ratio'], type_data['mAP_gcn'], 
             marker='s', linestyle='-', linewidth=2, label=f'GCN-{occ_type}')
ax1.set_xlabel('Occlusion Ratio', fontsize=10)
ax1.set_ylabel('mAP', fontsize=10)
ax1.set_title('mAP Degradation by Occlusion Type', fontsize=12, fontweight='bold')
ax1.legend(fontsize=7, ncol=2)
ax1.grid(True, alpha=0.3)

# 2. Degradation curves by occlusion type (Rank-1)
ax2 = plt.subplot(2, 4, 2)
for occ_type in ['center', 'top', 'bottom', 'left', 'right', 'grid']:
    type_data = merged_df[merged_df['occlusion_type'] == occ_type].sort_values('occlusion_ratio')
    ax2.plot(type_data['occlusion_ratio'], type_data['rank1_baseline'], 
             marker='o', linestyle='--', alpha=0.6, label=f'Baseline-{occ_type}')
    ax2.plot(type_data['occlusion_ratio'], type_data['rank1_gcn'], 
             marker='s', linestyle='-', linewidth=2, label=f'GCN-{occ_type}')
ax2.set_xlabel('Occlusion Ratio', fontsize=10)
ax2.set_ylabel('Rank-1 Accuracy', fontsize=10)
ax2.set_title('Rank-1 Degradation by Occlusion Type', fontsize=12, fontweight='bold')
ax2.legend(fontsize=7, ncol=2)
ax2.grid(True, alpha=0.3)

# 3. Heatmap: mAP by ratio and type
ax3 = plt.subplot(2, 4, 3)
pivot_baseline = merged_df.pivot(index='occlusion_type', columns='occlusion_ratio', values='mAP_baseline')
pivot_baseline = pivot_baseline.reindex(['center', 'top', 'bottom', 'left', 'right', 'grid'])
sns.heatmap(pivot_baseline, annot=True, fmt='.3f', cmap='YlOrRd_r', ax=ax3, cbar_kws={'label': 'mAP'})
ax3.set_title('Baseline mAP Heatmap', fontsize=12, fontweight='bold')
ax3.set_xlabel('Occlusion Ratio')
ax3.set_ylabel('Occlusion Type')

# 4. Heatmap: GCN mAP by ratio and type
ax4 = plt.subplot(2, 4, 4)
pivot_gcn = merged_df.pivot(index='occlusion_type', columns='occlusion_ratio', values='mAP_gcn')
pivot_gcn = pivot_gcn.reindex(['center', 'top', 'bottom', 'left', 'right', 'grid'])
sns.heatmap(pivot_gcn, annot=True, fmt='.3f', cmap='YlOrRd_r', ax=ax4, cbar_kws={'label': 'mAP'})
ax4.set_title('GCN mAP Heatmap', fontsize=12, fontweight='bold')
ax4.set_xlabel('Occlusion Ratio')
ax4.set_ylabel('Occlusion Type')

# 5. Performance gap by occlusion type (mAP)
ax5 = plt.subplot(2, 4, 5)
gap_by_type = occluded_df.groupby('occlusion_type').agg({
    'mAP_improvement': 'mean',
    'rank1_improvement': 'mean'
}).reset_index()
colors = ['green' if x > 0 else 'red' for x in gap_by_type['mAP_improvement']]
ax5.barh(gap_by_type['occlusion_type'], gap_by_type['mAP_improvement'], color=colors, alpha=0.7)
ax5.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax5.set_xlabel('Average mAP Improvement (%)', fontsize=10)
ax5.set_ylabel('Occlusion Type', fontsize=10)
ax5.set_title('GCN vs Baseline: mAP by Type', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(gap_by_type['mAP_improvement']):
    ax5.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

# 6. Performance gap by occlusion type (Rank-1)
ax6 = plt.subplot(2, 4, 6)
colors = ['green' if x > 0 else 'red' for x in gap_by_type['rank1_improvement']]
ax6.barh(gap_by_type['occlusion_type'], gap_by_type['rank1_improvement'], color=colors, alpha=0.7)
ax6.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax6.set_xlabel('Average Rank-1 Improvement (%)', fontsize=10)
ax6.set_ylabel('Occlusion Type', fontsize=10)
ax6.set_title('GCN vs Baseline: Rank-1 by Type', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(gap_by_type['rank1_improvement']):
    ax6.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)

# 7. Performance gap by occlusion ratio
ax7 = plt.subplot(2, 4, 7)
gap_by_ratio = occluded_df.groupby('occlusion_ratio').agg({
    'mAP_improvement': 'mean',
    'rank1_improvement': 'mean'
}).reset_index()
x = np.arange(len(gap_by_ratio))
width = 0.35
bars1 = ax7.bar(x - width/2, gap_by_ratio['mAP_improvement'], width, label='mAP', alpha=0.8)
bars2 = ax7.bar(x + width/2, gap_by_ratio['rank1_improvement'], width, label='Rank-1', alpha=0.8)
ax7.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax7.set_xlabel('Occlusion Ratio', fontsize=10)
ax7.set_ylabel('Average Improvement (%)', fontsize=10)
ax7.set_title('GCN vs Baseline: Gap by Ratio', fontsize=12, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels([f'{int(r*100)}%' for r in gap_by_ratio['occlusion_ratio']])
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

# Color bars based on positive/negative
for bar in bars1:
    if bar.get_height() < 0:
        bar.set_color('tomato')
    else:
        bar.set_color('lightgreen')
for bar in bars2:
    if bar.get_height() < 0:
        bar.set_color('salmon')
    else:
        bar.set_color('mediumseagreen')

# 8. Worst case analysis
ax8 = plt.subplot(2, 4, 8)
worst_cases = occluded_df.nsmallest(10, 'rank1_improvement')[['config_name', 'rank1_improvement', 'rank1_baseline', 'rank1_gcn']]
y_pos = np.arange(len(worst_cases))
ax8.barh(y_pos, worst_cases['rank1_improvement'], color='darkred', alpha=0.7)
ax8.set_yticks(y_pos)
ax8.set_yticklabels(worst_cases['config_name'], fontsize=8)
ax8.set_xlabel('Rank-1 Improvement (%)', fontsize=10)
ax8.set_title('Top 10 Worst GCN Degradations', fontsize=12, fontweight='bold')
ax8.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax8.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(worst_cases.iterrows()):
    ax8.text(row['rank1_improvement'] - 5, i, 
             f"{row['rank1_improvement']:.1f}%", 
             va='center', ha='right', fontsize=8, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/robustness_comparison/detailed_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: outputs/robustness_comparison/detailed_analysis.png")

# Generate detailed text report
with open('outputs/robustness_comparison/detailed_analysis_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DETAILED ROBUSTNESS ANALYSIS: GCN vs BASELINE\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("1. SUMMARY STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total configurations tested: {len(merged_df)}\n")
    f.write(f"Occluded configurations: {len(occluded_df)}\n\n")
    
    f.write("Overall performance (clean images, 0% occlusion):\n")
    clean_data = merged_df[merged_df['occlusion_ratio'] == 0].iloc[0]
    f.write(f"  Baseline mAP: {clean_data['mAP_baseline']:.4f}\n")
    f.write(f"  GCN mAP:      {clean_data['mAP_gcn']:.4f}\n")
    f.write(f"  Improvement:  {clean_data['mAP_improvement']:.2f}%\n")
    f.write(f"  Baseline Rank-1: {clean_data['rank1_baseline']:.4f}\n")
    f.write(f"  GCN Rank-1:      {clean_data['rank1_gcn']:.4f}\n")
    f.write(f"  Improvement:     {clean_data['rank1_improvement']:.2f}%\n\n")
    
    f.write("Average performance under occlusion:\n")
    f.write(f"  mAP improvement:    {occluded_df['mAP_improvement'].mean():.2f}%\n")
    f.write(f"  Rank-1 improvement: {occluded_df['rank1_improvement'].mean():.2f}%\n\n")
    
    f.write("\n2. PERFORMANCE BY OCCLUSION TYPE\n")
    f.write("-" * 80 + "\n")
    for occ_type in ['center', 'top', 'bottom', 'left', 'right', 'grid']:
        type_data = occluded_df[occluded_df['occlusion_type'] == occ_type]
        f.write(f"\n{occ_type.upper()} Occlusion:\n")
        f.write(f"  Configurations tested: {len(type_data)}\n")
        f.write(f"  Average mAP improvement:    {type_data['mAP_improvement'].mean():+7.2f}%\n")
        f.write(f"  Average Rank-1 improvement: {type_data['rank1_improvement'].mean():+7.2f}%\n")
        f.write(f"  Best case (mAP):  {type_data['mAP_improvement'].max():+6.2f}% @ {type_data.loc[type_data['mAP_improvement'].idxmax(), 'config_name']}\n")
        f.write(f"  Worst case (mAP): {type_data['mAP_improvement'].min():+6.2f}% @ {type_data.loc[type_data['mAP_improvement'].idxmin(), 'config_name']}\n")
    
    f.write("\n\n3. PERFORMANCE BY OCCLUSION RATIO\n")
    f.write("-" * 80 + "\n")
    for ratio in sorted(occluded_df['occlusion_ratio'].unique()):
        ratio_data = occluded_df[occluded_df['occlusion_ratio'] == ratio]
        f.write(f"\n{int(ratio*100)}% Occlusion:\n")
        f.write(f"  Configurations tested: {len(ratio_data)}\n")
        f.write(f"  Average mAP improvement:    {ratio_data['mAP_improvement'].mean():+7.2f}%\n")
        f.write(f"  Average Rank-1 improvement: {ratio_data['rank1_improvement'].mean():+7.2f}%\n")
        f.write(f"  GCN wins: {(ratio_data['mAP_improvement'] > 0).sum()}/{len(ratio_data)} configs\n")
    
    f.write("\n\n4. CRITICAL FINDINGS\n")
    f.write("-" * 80 + "\n")
    
    # Find where GCN wins
    gcn_wins = occluded_df[occluded_df['mAP_improvement'] > 0]
    f.write(f"\nGCN outperforms Baseline in {len(gcn_wins)}/{len(occluded_df)} occluded configs ({len(gcn_wins)/len(occluded_df)*100:.1f}%)\n")
    f.write("\nConfigurations where GCN wins (sorted by improvement):\n")
    for idx, row in gcn_wins.nlargest(5, 'mAP_improvement').iterrows():
        f.write(f"  {row['config_name']:20s} | mAP: {row['mAP_improvement']:+6.2f}% | Rank-1: {row['rank1_improvement']:+6.2f}%\n")
    
    # Find where Baseline wins
    baseline_wins = occluded_df[occluded_df['mAP_improvement'] < 0]
    f.write(f"\nBaseline outperforms GCN in {len(baseline_wins)}/{len(occluded_df)} occluded configs ({len(baseline_wins)/len(occluded_df)*100:.1f}%)\n")
    f.write("\nWorst GCN degradations (sorted by Rank-1 drop):\n")
    for idx, row in baseline_wins.nsmallest(10, 'rank1_improvement').iterrows():
        f.write(f"  {row['config_name']:20s} | mAP: {row['mAP_improvement']:+6.2f}% | Rank-1: {row['rank1_improvement']:+6.2f}%\n")
        f.write(f"    Baseline: mAP={row['mAP_baseline']:.4f}, Rank-1={row['rank1_baseline']:.4f}\n")
        f.write(f"    GCN:      mAP={row['mAP_gcn']:.4f}, Rank-1={row['rank1_gcn']:.4f}\n")
    
    f.write("\n\n5. HYPOTHESIS ANALYSIS\n")
    f.write("-" * 80 + "\n")
    f.write("Original Hypothesis H2: '遮挡越严重，GCN的相对优势越明显'\n")
    f.write("(The more severe the occlusion, the more pronounced GCN's advantage)\n\n")
    
    f.write("VERDICT: **REJECTED**\n\n")
    f.write("Evidence:\n")
    for ratio in sorted(occluded_df['occlusion_ratio'].unique()):
        ratio_data = occluded_df[occluded_df['occlusion_ratio'] == ratio]
        avg_improvement = ratio_data['mAP_improvement'].mean()
        f.write(f"  {int(ratio*100):2d}% occlusion: {avg_improvement:+7.2f}% avg mAP improvement\n")
    
    f.write("\nPattern observed: GCN's advantage **DECREASES** with occlusion severity\n")
    f.write("GCN is MORE sensitive to occlusion than Baseline, especially for:\n")
    f.write("  - Bottom occlusions (vehicle body)\n")
    f.write("  - Top occlusions (vehicle roof)\n")
    f.write("  - Grid/distributed occlusions\n\n")
    
    f.write("Possible explanations:\n")
    f.write("  1. GCN's graph structure may emphasize specific spatial regions (center)\n")
    f.write("  2. Graph connections could propagate errors when key nodes are occluded\n")
    f.write("  3. Baseline's simpler feature extraction may be more robust to partial occlusion\n")
    f.write("  4. GCN was NOT trained with occlusion augmentation\n")
    
    f.write("\n\n6. RECOMMENDATIONS\n")
    f.write("-" * 80 + "\n")
    f.write("1. **Do NOT claim superior robustness** in paper/presentation\n")
    f.write("2. **Reframe narrative**:\n")
    f.write("   - GCN achieves higher accuracy on clean data (+14.67% mAP)\n")
    f.write("   - But exhibits accuracy-robustness trade-off under occlusion\n")
    f.write("   - Baseline maintains robustness at cost of clean accuracy\n\n")
    f.write("3. **Future work**:\n")
    f.write("   - Retrain GCN with occlusion-aware augmentation\n")
    f.write("   - Analyze graph attention patterns under occlusion\n")
    f.write("   - Investigate graph structure modifications for robustness\n")
    f.write("   - Consider hybrid approach: GCN for clean, Baseline for occluded\n\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF DETAILED ANALYSIS\n")
    f.write("=" * 80 + "\n")

print(f"✓ Saved: outputs/robustness_comparison/detailed_analysis_report.txt")

# Print key statistics
print("\n" + "=" * 80)
print("KEY FINDINGS SUMMARY")
print("=" * 80)
print(f"\nClean data (0% occlusion):")
print(f"  GCN mAP advantage: +{clean_data['mAP_improvement']:.2f}%")
print(f"\nOccluded data (average):")
print(f"  GCN mAP change:    {occluded_df['mAP_improvement'].mean():+.2f}%")
print(f"  GCN Rank-1 change: {occluded_df['rank1_improvement'].mean():+.2f}%")
print(f"\nGCN wins: {len(gcn_wins)}/{len(occluded_df)} configs ({len(gcn_wins)/len(occluded_df)*100:.1f}%)")
print(f"\nWorst degradation: {occluded_df['rank1_improvement'].min():.2f}% @ {occluded_df.loc[occluded_df['rank1_improvement'].idxmin(), 'config_name']}")
print("\n" + "=" * 80)
