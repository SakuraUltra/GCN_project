# 消融实验最终结果汇总
**实验日期**: 2026-03-07  
**数据集**: VeRi-776  
**训练轮数**: 120 epochs  
**评估周期**: 每10 epochs  

---

## 实验配置总览

| 实验ID | Backbone | Edge Type | GCN Layers | Job ID | 配置文件 | Checkpoint保存路径 |
|--------|----------|-----------|------------|--------|----------|--------------------|
| ABL-01/03/07 | ResNet50-IBN | 4-neighbor | 1 | 31433548 | abl_cnn_gcn_4nb_l1.yaml | outputs/ablation/cnn_gcn_4nb_l1/ |
| ABL-04 | ResNet50-IBN | 4-neighbor | 2 | 31433549 | abl_cnn_gcn_4nb_l2.yaml | outputs/ablation/cnn_gcn_4nb_l2/ |
| ABL-08 | ResNet50-IBN | kNN(k=8) | 1 | 31433550 | abl_cnn_gcn_knn_l1.yaml | outputs/ablation/cnn_gcn_knn_l1/ |
| ABL-02/10/14 | ViT-Base-768 | 4-neighbor | 1 | 31433551 | abl_vit_gcn_4nb_l1.yaml | outputs/ablation/vit_gcn_4nb_l1/ |
| ABL-11 | ViT-Base-768 | 4-neighbor | 2 | 31433552 | abl_vit_gcn_4nb_l2.yaml | outputs/ablation/vit_gcn_4nb_l2/ |
| ABL-15 | ViT-Base-768 | kNN(k=8) | 1 | 31433315 ✅ | abl_vit_gcn_knn_l1.yaml | outputs/ablation/vit_gcn_knn_l1/ ✅ |

---

## Epoch 120 最终结果

### ResNet50-IBN + GCN

| 实验 | mAP | Rank-1 | Rank-5 | Rank-10 | 备注 |
|------|-----|--------|--------|---------|------|
| **ABL-01/03/07** (4nb+L1) | **73.82%** | **93.56%** | **97.02%** | 98.21% | ⭐ CNN最佳 |
| **ABL-04** (4nb+L2) | 73.29% | 93.09% | 97.08% | **98.39%** | 2层略降 |
| **ABL-08** (kNN+L1) | 72.67% | 92.91% | 96.90% | 98.15% | 动态图 |

### ViT-Base-768 + GCN

| 实验 | mAP | Rank-1 | Rank-5 | Rank-10 | 备注 |
|------|-----|--------|--------|---------|------|
| **ABL-02/10/14** (4nb+L1) | 72.52% | **94.76%** | **97.85%** | 98.75% | ⭐ Rank-1最佳 |
| **ABL-11** (4nb+L2) | 71.51% | 93.74% | 97.74% | **98.93%** | 2层略降 |
| **ABL-15** (kNN+L1) | 71.91% | 94.10% | 97.74% | **98.93%** | 动态图 |

---

## 关键发现

### 1. Backbone比较 (CNN vs ViT)
- **ResNet50最佳mAP**: 73.82% (ABL-01, 4nb+L1)
- **ViT最佳mAP**: 72.52% (ABL-02, 4nb+L1)
- **ViT最佳Rank-1**: 94.76% (ABL-02, 4nb+L1) ⭐ **高于CNN**
- **结论**: ResNet50在mAP上领先约1.3%，但ViT在Rank-1上领先1.2%

### 2. GCN深度影响 (L1 vs L2)
- **ResNet50**: L1 (73.82%) > L2 (73.29%), 差距0.53%
- **ViT**: L1 (72.52%) > L2 (71.51%), 差距1.01%
- **结论**: 1层GCN优于2层，ViT对深度更敏感

### 3. 边类型影响 (4-neighbor vs kNN)
- **ResNet50**: 4nb (73.82%) > kNN (72.67%), 差距1.15%
- **ViT**: 4nb (72.52%) > kNN (71.91%), 差距0.61%
- **结论**: 固定4邻域优于动态kNN，ResNet50对边类型更敏感

### 4. 最佳配置
- **最高mAP**: ABL-01 (ResNet50+4nb+L1) - **73.82%**
- **最高Rank-1**: ABL-02 (ViT+4nb+L1) - **94.76%**
- **最高Rank-10**: ABL-11/ABL-15 (ViT+L2/kNN) - **98.93%**

---

## 实验细节

### 数据增强配置
- **Random Erasing**: 
  - Probability: 0.5
  - Area: 2% ~ 20% (标准Re-ID配置)
  - Aspect Ratio: 0.3 ~ 3.33
  - Mode: random pixel fill

### 输入尺寸
- **ResNet50**: 256×256
- **ViT-Base**: 224×224 (模型要求)

### 优化器配置
- **ResNet50**: 
  - Optimizer: AdamW
  - Base LR: 3.5e-4
  - Weight Decay: 5e-4
- **ViT-Base**:
  - Optimizer: AdamW
  - Base LR: 3.5e-5 (10倍更小)
  - Weight Decay: 1e-4
  - LR Groups: backbone(1×), cls/gcn(10×)

### 训练时间 (每epoch)
- **ResNet50**: ~8秒
- **ViT-Base**: ~12秒

---

## 性能对比矩阵

### mAP排名
1. ResNet50+4nb+L1: **73.82%** ⭐
2. ResNet50+4nb+L2: 73.29%
3. ResNet50+kNN+L1: 72.67%
4. ViT+4nb+L1: 72.52%
5. ViT+kNN+L1: 71.91%
6. ViT+4nb+L2: 71.51%

### Rank-1排名
1. ViT+4nb+L1: **94.76%** ⭐
2. ViT+kNN+L1: 94.10%
3. ViT+4nb+L2: 93.74%
4. ResNet50+4nb+L1: 93.56%
5. ResNet50+4nb+L2: 93.09%
6. ResNet50+kNN+L1: 92.91%

---

## 结论与建议

### 主要结论
1. **ResNet50+4-neighbor+1层GCN** 是最佳mAP配置 (73.82%)
2. **ViT+4-neighbor+1层GCN** 在Rank-1上表现最佳 (94.76%)
3. **1层GCN优于2层GCN**，更深的图网络带来性能下降
4. **固定4邻域优于动态kNN**，空间先验更有效
5. ViT在top-1检索上更准确，ResNet50在整体排序上更好

### 实际应用建议
- **追求最高mAP**: 使用ResNet50+4nb+L1
- **追求最高Rank-1**: 使用ViT+4nb+L1
- **平衡性能与速度**: 使用ResNet50+4nb+L1（训练更快）
- **消融研究**: 1层GCN足够，无需堆叠更深

### 下一步工作
1. 混合图结构（4-neighbor + kNN）
2. 自适应邻域大小
3. 注意力增强的GCN（GAT）
4. 模型集成（ResNet50 + ViT）

---

**实验完成时间**: 2026-03-07 20:27 (第一批) / 重新训练中 (第二批)  
**GPU资源**: H100 PCIe (Viking SLURM集群)  
**总训练时间**: 约1.5小时 (6个模型)  
**Checkpoint保存**: 所有实验独立保存到 outputs/ablation/[experiment_name]/

## 配置修复说明

**问题**: 第一批实验使用了错误的配置字段 `OUTPUT_DIR`，导致所有模型checkpoint覆盖保存到默认目录  
**解决**: 已修复为正确的嵌套结构 `OUTPUT.DIR`，第二批实验将正确保存到独立目录  
**已保留**: ABL-15 (ViT+kNN+L1) 的完整checkpoint (Job 31433315)  
**重新训练**: ABL-01, 04, 08, 02, 11 (Jobs 31433548-31433552)
