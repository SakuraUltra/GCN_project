#!/usr/bin/env python3
"""
evaluate_occlusion_abl19.py
ABL-19: 消融实验遮挡鲁棒性评测

输出结构（对齐项目现有约定）：
  outputs/ablation_occlusion_results/
  ├── results_summary.csv              # 完整汇总表（66行数据）
  ├── model_comparison.png             # 可视化对比图（4个子图）
  ├── ABLATION_OCCLUSION_REPORT.md     # Markdown 格式详细报告
  └── individual_results/
      ├── ABL01_CNN_4nb_L1.csv
      ├── ABL04_CNN_4nb_L2.csv
      ├── ABL08_CNN_kNN_L1.csv
      ├── ABL02_ViT_4nb_L1.csv
      ├── ABL11_ViT_4nb_L2.csv
      └── ABL15_ViT_kNN_L1.csv

复用项目现有工具（接口已确认）：
  - ReIDEvaluator(model, use_flip_test=True, device=None)
      .evaluate(query_loader, gallery_loader)
      → dict: {mAP, cmc, rank1, rank5, rank10}
  - VeRiDataset(root, mode='test', transform=None)

用法：
  python scripts/testing/evaluate_occlusion_abl19.py
  python scripts/testing/evaluate_occlusion_abl19.py \
      --occ_root outputs/occlusion_tests_v2 \
      --output_dir outputs/ablation_occlusion_results
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── 项目根目录加入 sys.path ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.bot_baseline.bot_gcn_model import BoTGCN
from models.bot_baseline.veri_dataset import VeRiDataset
from eval.evaluator import ReIDEvaluator

# =====================================================================
# 6 个消融模型配置
# =====================================================================

ABLATION_MODELS = [
    {
        "name":        "CNN+GCN 4nb L1",
        "abl_id":      "ABL-01",
        "group":       "ResNet50",
        "description": "ResNet50-IBN + 4-neighbor + 1 layer GCN",
        "csv_name":    "ABL01_CNN_4nb_L1",
        "ckpt":        "outputs/ablation/cnn_gcn_4nb_l1/best_model.pth",
        "input_size":  (256, 256),
    },
    {
        "name":        "CNN+GCN 4nb L2",
        "abl_id":      "ABL-04",
        "group":       "ResNet50",
        "description": "ResNet50-IBN + 4-neighbor + 2 layer GCN",
        "csv_name":    "ABL04_CNN_4nb_L2",
        "ckpt":        "outputs/ablation/cnn_gcn_4nb_l2/best_model.pth",
        "input_size":  (256, 256),
    },
    {
        "name":        "CNN+GCN 4nb L3",
        "abl_id":      "ABL-05",
        "group":       "ResNet50",
        "description": "ResNet50-IBN + 4-neighbor + 3 layer GCN",
        "csv_name":    "ABL05_CNN_4nb_L3",
        "ckpt":        "outputs/ablation/cnn_gcn_4nb_l3/best_model.pth",
        "input_size":  (256, 256),
    },
    {
        "name":        "CNN+GCN kNN L1",
        "abl_id":      "ABL-08",
        "group":       "ResNet50",
        "description": "ResNet50-IBN + kNN(k=8) + 1 layer GCN",
        "csv_name":    "ABL08_CNN_kNN_L1",
        "ckpt":        "outputs/ablation/cnn_gcn_knn_l1/best_model.pth",
        "input_size":  (256, 256),
    },
    {
        "name":        "ViT+GCN 4nb L1",
        "abl_id":      "ABL-02",
        "group":       "ViT-Base",
        "description": "ViT-Base-768 + 4-neighbor + 1 layer GCN",
        "csv_name":    "ABL02_ViT_4nb_L1",
        "ckpt":        "outputs/ablation/vit_gcn_4nb_l1/best_model.pth",
        "input_size":  (224, 224),
    },
    {
        "name":        "ViT+GCN 4nb L2",
        "abl_id":      "ABL-11",
        "group":       "ViT-Base",
        "description": "ViT-Base-768 + 4-neighbor + 2 layer GCN",
        "csv_name":    "ABL11_ViT_4nb_L2",
        "ckpt":        "outputs/ablation/vit_gcn_4nb_l2/best_model.pth",
        "input_size":  (224, 224),
    },
    {
        "name":        "ViT+GCN kNN L1",
        "abl_id":      "ABL-15",
        "group":       "ViT-Base",
        "description": "ViT-Base-768 + kNN(k=8) + 1 layer GCN",
        "csv_name":    "ABL15_ViT_kNN_L1",
        "ckpt":        "outputs/ablation/vit_gcn_knn_l1/best_model.pth",
        "input_size":  (224, 224),
    },
]

OCC_LEVELS  = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
DATASET_ROOT = "data/dataset/776_DataSet"  # VeRi-776 dataset root

# 绘图颜色
COLORS = {
    "CNN+GCN 4nb L1": "#1f77b4",
    "CNN+GCN 4nb L2": "#aec7e8",
    "CNN+GCN 4nb L3": "#6baed6",
    "CNN+GCN kNN L1": "#ff7f0e",
    "ViT+GCN 4nb L1": "#2ca02c",
    "ViT+GCN 4nb L2": "#98df8a",
    "ViT+GCN kNN L1": "#d62728",
}
MARKERS = {
    "CNN+GCN 4nb L1": "o",
    "CNN+GCN 4nb L2": "s",
    "CNN+GCN 4nb L3": "D",
    "CNN+GCN kNN L1": "^",
    "ViT+GCN 4nb L1": "d",
    "ViT+GCN 4nb L2": "v",
    "ViT+GCN kNN L1": "P",
}


# =====================================================================
# 从 checkpoint 恢复模型（strict=True）
# =====================================================================

def load_model(ckpt_path: str, device: torch.device) -> BoTGCN:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        raise KeyError(f"Checkpoint 缺少 'config' 字段: {ckpt_path}")

    cfg = ckpt["config"]
    mc  = cfg["MODEL"]
    gcn = mc.get("GCN", {})
    fus = mc.get("FUSION", {})
    bb  = mc.get("BACKBONE", {})

    if isinstance(bb, str):
        bb_type, bb = "resnet", {}
    else:
        bb_type = "vit"

    model = BoTGCN(
        num_classes        = mc.get("NUM_CLASSES", 576),
        last_stride        = mc.get("LAST_STRIDE", 1),
        pretrain_path      = "",
        backbone_type      = bb_type,
        vit_model_name     = bb.get("NAME", "vit_base_patch16_224.augreg_in21k_ft_in1k"),
        vit_pretrained     = False,
        vit_native_dim     = bb.get("NATIVE_DIM", False),
        vit_proj_channels  = bb.get("OUT_CHANNELS", 2048),
        vit_target_spatial = bb.get("TARGET_SPATIAL", 8),
        use_gcn            = gcn.get("USE_GCN", True),
        grid_h             = gcn.get("GRID_H", 4),
        grid_w             = gcn.get("GRID_W", 4),
        adjacency_type     = gcn.get("ADJACENCY_TYPE", "4"),
        gcn_hidden_dim     = gcn.get("HIDDEN_CHANNELS", 512),
        gcn_out_dim        = gcn.get("OUT_CHANNELS", None),
        gcn_num_layers     = gcn.get("NUM_LAYERS", 1),
        gcn_dropout        = gcn.get("DROPOUT", 0.5),
        knn_k              = gcn.get("KNN_K", 8),
        knn_metric         = gcn.get("KNN_METRIC", "cosine"),
        knn_detach         = gcn.get("KNN_DETACH", True),
        gnn_type           = gcn.get("GNN_TYPE", "gcn"),
        gat_heads          = gcn.get("HEADS", 4),
        pooling_type       = gcn.get("POOLING_TYPE", "mean"),
        pooling_hidden_dim = gcn.get("POOLING_HIDDEN_DIM", 128),
        fusion_type        = fus.get("TYPE", "concat"),
        fusion_hidden_dim  = fus.get("HIDDEN_DIM", 512),
        fusion_dropout     = fus.get("DROPOUT", 0.5),
        neck               = mc.get("NECK", "bnneck"),
    )

    # 实际 checkpoint 使用 "model" 键，而非 "model_state_dict"
    sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


# =====================================================================
# 简单的图像目录数据集（用于遮挡查询集）
# =====================================================================

class SimpleImageDataset(Dataset):
    """直接从目录加载图像，解析VeRi文件名格式"""
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = self._parse_data()
    
    def _parse_data(self):
        import glob
        import re
        img_paths = glob.glob(os.path.join(self.img_dir, '*.jpg'))
        data = []
        for img_path in img_paths:
            filename = os.path.basename(img_path)
            # VeRi format: XXXX_cYYY_ZZZZZZZZ_W.jpg
            pattern = r'(\d+)_c(\d+)_(\d+)_(\d+)\.jpg'
            match = re.match(pattern, filename)
            if match:
                vehicle_id = int(match.group(1))
                camera_id = int(match.group(2))
                data.append((img_path, vehicle_id, camera_id))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, pid, camid = self.data[idx]
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid


# =====================================================================
# 数据加载器
# =====================================================================

def build_loader(img_dir: str, input_size: tuple, mode: str = "simple") -> DataLoader:
    """
    构建数据加载器
    Args:
        img_dir: 图像目录或数据集根目录
        input_size: 输入尺寸 (H, W)
        mode: "veri_root" 使用VeRiDataset (需要dataset root)
              "simple" 直接加载目录中的图像
    """
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    if mode == "veri_root":
        # VeRiDataset 需要数据集根目录
        dataset = VeRiDataset(img_dir, transform=transform, mode="gallery")
    else:
        # SimpleImageDataset 直接加载指定目录
        dataset = SimpleImageDataset(img_dir, transform=transform)
    
    return DataLoader(dataset, batch_size=64, shuffle=False,
                      num_workers=8, pin_memory=True)


# =====================================================================
# 主评测循环
# =====================================================================

def run_abl19(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    individual_dir = output_dir / "individual_results"
    individual_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ABL-19 遮挡鲁棒性评测")
    print(f"Device: {device} | 模型: {len(ABLATION_MODELS)} | 遮挡等级: {len(OCC_LEVELS)}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")

    # all_rows: 每行对应一个 (model, occ_level) 的结果
    all_rows = []

    for cfg in ABLATION_MODELS:
        name       = cfg["name"]
        abl_id     = cfg["abl_id"]
        ckpt_path  = os.path.join(PROJECT_ROOT, cfg["ckpt"])
        input_size = cfg["input_size"]

        print(f"\n{'─'*55}")
        print(f"[{abl_id}] {name}  input={input_size}")

        if not os.path.exists(ckpt_path):
            print(f"  ✗ Checkpoint 不存在，跳过: {ckpt_path}")
            continue

        try:
            model = load_model(ckpt_path, device)
            print(f"  ✓ 模型加载成功")
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            continue

        evaluator      = ReIDEvaluator(model, use_flip_test=True, device=device)
        # Gallery使用VeRiDataset（需要dataset root）
        gallery_loader = build_loader(
            os.path.join(PROJECT_ROOT, DATASET_ROOT), input_size, mode="veri_root"
        )
        model_rows = []

        for occ_level in OCC_LEVELS:
            occ_dir = os.path.join(
                PROJECT_ROOT, args.occ_root,
                f"query_{occ_level:02d}pct"
            )
            if not os.path.exists(occ_dir):
                print(f"  ✗ 遮挡目录不存在，跳过: query_{occ_level:02d}pct")
                continue

            # Query使用SimpleImageDataset（直接加载目录）
            query_loader = build_loader(occ_dir, input_size, mode="simple")
            metrics      = evaluator.evaluate(query_loader, gallery_loader)

            row = {
                "model_name":     name,
                "abl_id":         abl_id,
                "model_group":    cfg["group"],
                "description":    cfg["description"],
                "occlusion_level": occ_level,
                "mAP":   round(metrics["mAP"]   * 100, 4),
                "rank1": round(metrics["rank1"] * 100, 4),
                "rank5": round(metrics["rank5"] * 100, 4),
                "rank10":round(metrics["rank10"]* 100, 4),
            }
            model_rows.append(row)
            all_rows.append(row)

            print(f"  occ={occ_level:2d}%  "
                  f"mAP={row['mAP']:.2f}%  "
                  f"R1={row['rank1']:.2f}%  "
                  f"R5={row['rank5']:.2f}%")

        # 保存单模型 individual CSV
        if model_rows:
            ind_path = individual_dir / f"{cfg['csv_name']}.csv"
            _save_individual_csv(model_rows, ind_path)
            print(f"  ✅ Saved: {ind_path.name}")

        # 每个模型完成后保存中间 JSON，防止中断
        _save_json(all_rows, output_dir / "_checkpoint.json")

    if not all_rows:
        print("\n⚠️ 无结果，请检查 checkpoint 路径和遮挡目录")
        return

    df = pd.DataFrame(all_rows)

    # 保存汇总表
    summary_path = output_dir / "results_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\n✓ results_summary.csv 已保存")

    # 生成可视化图
    fig_path = output_dir / "model_comparison.png"
    _plot_comparison(df, fig_path)
    print(f"✓ model_comparison.png 已保存")

    # 生成 Markdown 报告
    md_path = output_dir / "ABLATION_OCCLUSION_REPORT.md"
    _generate_report(df, md_path)
    print(f"✓ ABLATION_OCCLUSION_REPORT.md 已保存")

    _print_summary(df)
    print(f"\n✓ 全部结果已保存至: {output_dir}")


# =====================================================================
# 可视化：4 个子图
# =====================================================================

def _plot_comparison(df: pd.DataFrame, save_path: Path):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("ABL-19: Ablation Occlusion Robustness\n"
                 "VeRi-776 | RE(sh=0.2, random) | 4×4 Grid GCN",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── 子图1：mAP vs 遮挡等级 ────────────────────────────────────────
    for cfg in ABLATION_MODELS:
        name = cfg["name"]
        sub  = df[df["model_name"] == name].sort_values("occlusion_level")
        if sub.empty:
            continue
        ax1.plot(sub["occlusion_level"], sub["mAP"],
                 color=COLORS[name], marker=MARKERS[name],
                 linewidth=1.8, markersize=6, label=name)
    ax1.set_title("mAP vs Occlusion Level", fontweight="bold")
    ax1.set_xlabel("Occlusion Level (%)")
    ax1.set_ylabel("mAP (%)")
    ax1.set_xticks(OCC_LEVELS)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── 子图2：Rank-1 vs 遮挡等级 ────────────────────────────────────
    for cfg in ABLATION_MODELS:
        name = cfg["name"]
        sub  = df[df["model_name"] == name].sort_values("occlusion_level")
        if sub.empty:
            continue
        ax2.plot(sub["occlusion_level"], sub["rank1"],
                 color=COLORS[name], marker=MARKERS[name],
                 linewidth=1.8, markersize=6, label=name)
    ax2.set_title("Rank-1 vs Occlusion Level", fontweight="bold")
    ax2.set_xlabel("Occlusion Level (%)")
    ax2.set_ylabel("Rank-1 (%)")
    ax2.set_xticks(OCC_LEVELS)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ── 子图3：mAP 相对下降（相对 0% 遮挡） ──────────────────────────
    for cfg in ABLATION_MODELS:
        name  = cfg["name"]
        sub   = df[df["model_name"] == name].sort_values("occlusion_level")
        if sub.empty:
            continue
        clean = sub[sub["occlusion_level"] == 0]["mAP"].values
        if len(clean) == 0:
            continue
        drop = sub["mAP"].values - clean[0]
        ax3.plot(sub["occlusion_level"].values, drop,
                 color=COLORS[name], marker=MARKERS[name],
                 linewidth=1.8, markersize=6, label=name)
    ax3.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax3.set_title("mAP Drop (relative to 0% occlusion)", fontweight="bold")
    ax3.set_xlabel("Occlusion Level (%)")
    ax3.set_ylabel("ΔmAP (pp)")
    ax3.set_xticks(OCC_LEVELS)
    ax3.legend(fontsize=7, loc="lower left")
    ax3.grid(True, alpha=0.3)

    # ── 子图4：热力图（模型 × 遮挡等级，mAP） ────────────────────────
    model_names = [c["name"] for c in ABLATION_MODELS]
    heatmap_data = []
    for name in model_names:
        sub = df[df["model_name"] == name].sort_values("occlusion_level")
        row_data = []
        for lvl in OCC_LEVELS:
            val = sub[sub["occlusion_level"] == lvl]["mAP"].values
            row_data.append(val[0] if len(val) > 0 else float("nan"))
        heatmap_data.append(row_data)

    heatmap_arr = np.array(heatmap_data)
    im = ax4.imshow(heatmap_arr, aspect="auto", cmap="RdYlGn",
                    vmin=np.nanmin(heatmap_arr), vmax=np.nanmax(heatmap_arr))
    ax4.set_xticks(range(len(OCC_LEVELS)))
    ax4.set_xticklabels([f"{l}%" for l in OCC_LEVELS], fontsize=8)
    ax4.set_yticks(range(len(model_names)))
    ax4.set_yticklabels([f"{c['abl_id']} {n[:14]}"
                         for c, n in zip(ABLATION_MODELS, model_names)],
                        fontsize=8)
    ax4.set_title("mAP Heatmap (Model × Occlusion)", fontweight="bold")
    ax4.set_xlabel("Occlusion Level (%)")
    plt.colorbar(im, ax=ax4, label="mAP (%)", fraction=0.03)

    # 在热力图格子内标注数值
    for i in range(len(model_names)):
        for j in range(len(OCC_LEVELS)):
            val = heatmap_arr[i, j]
            if not np.isnan(val):
                ax4.text(j, i, f"{val:.1f}", ha="center", va="center",
                         fontsize=6.5, color="black")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# Markdown 报告
# =====================================================================

def _generate_report(df: pd.DataFrame, save_path: Path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 关键指标计算
    clean_df = df[df["occlusion_level"] == 0][["model_name", "abl_id", "mAP", "rank1"]]
    occ21_df = df[df["occlusion_level"] == 21][["model_name", "mAP", "rank1"]].rename(
        columns={"mAP": "mAP_21", "rank1": "rank1_21"})
    occ30_df = df[df["occlusion_level"] == 30][["model_name", "mAP", "rank1"]].rename(
        columns={"mAP": "mAP_30", "rank1": "rank1_30"})

    merged = clean_df.merge(occ21_df, on="model_name", how="left") \
                     .merge(occ30_df, on="model_name", how="left")
    merged["drop21"] = merged["mAP_21"] - merged["mAP"]
    merged["drop30"] = merged["mAP_30"] - merged["mAP"]

    best_clean  = merged.loc[merged["mAP"].idxmax()]
    best_robust = merged.loc[merged["drop30"].idxmax()]   # drop 最小（值最大）

    lines = [
        f"# ABLATION_OCCLUSION_REPORT",
        f"",
        f"> 生成时间: {now}",
        f"> 数据集: VeRi-776 | 遮挡协议: Random Erasing Re-ID 标准 | 遮挡档位: 0%–30%（步长3%）",
        f"",
        f"---",
        f"",
        f"## 一、汇总表格",
        f"",
        f"| ABL_ID | 模型 | 组 | clean mAP | clean R1 | occ21% mAP | occ30% mAP | Drop@21 | Drop@30 |",
        f"|--------|------|-----|-----------|----------|------------|------------|---------|---------|",
    ]
    for _, row in merged.iterrows():
        lines.append(
            f"| {row['abl_id']} | {row['model_name']} | "
            f"{'CNN' if 'CNN' in row['model_name'] else 'ViT'} | "
            f"**{row['mAP']:.2f}%** | {row['rank1']:.2f}% | "
            f"{row['mAP_21']:.2f}% | {row['mAP_30']:.2f}% | "
            f"{row['drop21']:+.2f}% | {row['drop30']:+.2f}% |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## 二、关键结论",
        f"",
        f"### 最佳性能模型（0% 遮挡）",
        f"",
        f"- **{best_clean['abl_id']} {best_clean['model_name']}**",
        f"  - clean mAP: **{best_clean['mAP']:.2f}%**",
        f"  - clean Rank-1: **{best_clean['rank1']:.2f}%**",
        f"",
        f"### 最鲁棒模型（30% 遮挡 Drop 最小）",
        f"",
        f"- **{best_robust['abl_id']} {best_robust['model_name']}**",
        f"  - Drop@30: **{best_robust['drop30']:+.2f}%**",
        f"  - occ30% mAP: **{best_robust['mAP_30']:.2f}%**",
        f"",
        f"---",
        f"",
        f"## 三、Backbone 对比分析",
        f"",
    ]

    cnn_rows = merged[merged["model_name"].str.contains("CNN")]
    vit_rows = merged[merged["model_name"].str.contains("ViT")]

    lines += [
        f"| 指标 | ResNet50 最优 | ViT-Base 最优 |",
        f"|------|--------------|--------------|",
        f"| clean mAP   | {cnn_rows['mAP'].max():.2f}% | {vit_rows['mAP'].max():.2f}% |",
        f"| clean Rank-1 | {cnn_rows['rank1'].max():.2f}% | {vit_rows['rank1'].max():.2f}% |",
        f"| occ30% mAP  | {cnn_rows['mAP_30'].max():.2f}% | {vit_rows['mAP_30'].max():.2f}% |",
        f"| Drop@30（最小）| {cnn_rows['drop30'].max():+.2f}% | {vit_rows['drop30'].max():+.2f}% |",
        f"",
        f"---",
        f"",
        f"## 四、Depth 消融遮挡对比（4-neighbor）",
        f"",
        f"| Backbone | L=1 clean | L=2 clean | L=1 Drop@30 | L=2 Drop@30 |",
        f"|----------|-----------|-----------|-------------|-------------|",
    ]

    for backbone, prefix in [("ResNet50", "CNN"), ("ViT-Base", "ViT")]:
        l1 = merged[merged["model_name"].str.contains(f"{prefix}.*4nb.*L1")]
        l2 = merged[merged["model_name"].str.contains(f"{prefix}.*4nb.*L2")]
        if not l1.empty and not l2.empty:
            l1, l2 = l1.iloc[0], l2.iloc[0]
            lines.append(
                f"| {backbone} | {l1['mAP']:.2f}% | {l2['mAP']:.2f}% | "
                f"{l1['drop30']:+.2f}% | {l2['drop30']:+.2f}% |"
            )

    lines += [
        f"",
        f"## 五、Edge 消融遮挡对比（L=1）",
        f"",
        f"| Backbone | 4-neighbor clean | kNN clean | 4nb Drop@30 | kNN Drop@30 |",
        f"|----------|-----------------|-----------|-------------|-------------|",
    ]

    for backbone, prefix in [("ResNet50", "CNN"), ("ViT-Base", "ViT")]:
        nb  = merged[merged["model_name"].str.contains(f"{prefix}.*4nb.*L1")]
        knn = merged[merged["model_name"].str.contains(f"{prefix}.*kNN")]
        if not nb.empty and not knn.empty:
            nb, knn = nb.iloc[0], knn.iloc[0]
            lines.append(
                f"| {backbone} | {nb['mAP']:.2f}% | {knn['mAP']:.2f}% | "
                f"{nb['drop30']:+.2f}% | {knn['drop30']:+.2f}% |"
            )

    save_path.write_text("\n".join(lines), encoding="utf-8")


# =====================================================================
# 保存工具
# =====================================================================

def _save_individual_csv(rows: list, path: Path):
    if not rows:
        return
    keys = ["abl_id", "model_name", "model_group", "description",
            "occlusion_level", "mAP", "rank1", "rank5", "rank10"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def _save_json(data, path: Path):
    """保存JSON，自动转换numpy类型为Python原生类型"""
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(convert_numpy(data), f, indent=2, ensure_ascii=False)


def _print_summary(df: pd.DataFrame):
    print(f"\n{'='*72}")
    print(f"{'ABL':8} {'模型':22} {'clean':7} {'occ21%':7} {'occ30%':7} "
          f"{'Drop@21':9} {'Drop@30':9}")
    print(f"{'─'*72}")
    for cfg in ABLATION_MODELS:
        name   = cfg["name"]
        sub    = df[df["model_name"] == name]
        if sub.empty:
            continue
        clean  = sub[sub["occlusion_level"] ==  0]["mAP"].values
        m21    = sub[sub["occlusion_level"] == 21]["mAP"].values
        m30    = sub[sub["occlusion_level"] == 30]["mAP"].values
        c      = clean[0] if len(clean) else float("nan")
        v21    = m21[0]   if len(m21)   else float("nan")
        v30    = m30[0]   if len(m30)   else float("nan")
        d21    = v21 - c
        d30    = v30 - c
        print(f"{cfg['abl_id']:8} {name:22} "
              f"{c:6.2f}%  {v21:6.2f}%  {v30:6.2f}%  "
              f"{d21:+7.2f}%  {d30:+7.2f}%")
    print(f"{'='*72}")


# =====================================================================
# 入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABL-19 遮挡鲁棒性评测")
    parser.add_argument("--occ_root",  default="outputs/occlusion_tests_v2",
                        help="遮挡数据集根目录（含 query_00pct ~ query_30pct）")
    parser.add_argument("--output_dir", default="outputs/ablation_occlusion_results",
                        help="结果输出目录")
    args = parser.parse_args()
    run_abl19(args)
