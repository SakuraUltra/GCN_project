#!/usr/bin/env python3
"""
evaluate_occlusion_vehicleid.py
VehicleID 数据集的遮挡鲁棒性评估脚本
评估 8 个消融模型在 0-30% Random Erasing 下的性能
"""

import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.bot_baseline.bot_gcn_model import BoTGCN
from models.bot_baseline.veri_dataset import VehicleIDDataset, split_vehicleid_test, build_transforms
from eval.evaluator import compute_mAP_cmc


def load_checkpoint_safe(model, checkpoint_path):
    """安全加载权重：自动跳过 shape 不匹配的参数"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 兼容不同 checkpoint 格式
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Bug 2 修复：去掉 DataParallel 的 module. 前缀
    state_dict = strip_module_prefix(state_dict)

    model_state = model.state_dict()
    
    filtered = {}
    skipped = []
    
    for k, v in state_dict.items():
        if k not in model_state:
            skipped.append(f"  [KEY_MISSING]   {k}")
        elif v.shape != model_state[k].shape:
            skipped.append(f"  [SHAPE_MISMATCH] {k}: ckpt={tuple(v.shape)} vs model={tuple(model_state[k].shape)}")
        else:
            filtered[k] = v
    
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    
    if skipped:
        print(f"⚠️  跳过 {len(skipped)} 个参数：")
        for s in skipped[:10]:  # 只打印前10个
            print(s)
        if len(skipped) > 10:
            print(f"  ... (共 {len(skipped)} 个)")
    print(f"✅ 成功加载 {len(filtered)}/{len(state_dict)} 个参数")
    
    return model


def strip_module_prefix(state_dict):
    """去掉 DataParallel 的 module. 前缀"""
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k[len('module.'):] if k.startswith('module.') else k
        new_sd[new_key] = v
    return new_sd


def build_model_from_checkpoint(checkpoint_path, num_classes, backbone_type, device='cpu'):
    """从 checkpoint 内保存的 config 重建模型，确保结构完全匹配"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取state_dict用于fallback推断
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Bug 2 修复：去掉 DataParallel 的 module. 前缀
    state_dict = strip_module_prefix(state_dict)
    
    # 从 state_dict 反推真实配置（最可靠）
    gcn_layer_keys = [k for k in state_dict.keys() if k.startswith('gcn.convs.') and k.endswith('.bias')]
    gcn_num_layers_actual = len(gcn_layer_keys)
    
    # Bug 1 修复：hidden_dim 取 shape[1]（input dim），不是 shape[0]（output dim）
    if 'gcn.convs.0.weight' in state_dict:
        # gcn.convs.0.weight: shape = (output_dim, input_dim)
        # 我们需要 input_dim，即 hidden_dim
        gcn_hidden_dim_actual = state_dict['gcn.convs.0.weight'].shape[1]
    elif 'gcn.convs.0.bias' in state_dict:
        # fallback: bias shape[0] = output_dim = hidden_dim (仅当第一层output=hidden时)
        gcn_hidden_dim_actual = state_dict['gcn.convs.0.bias'].shape[0]
    else:
        gcn_hidden_dim_actual = 512
    
    # 尝试从config读取（但要验证）
    adjacency_type = '4'  # 默认
    if 'config' in checkpoint and checkpoint['config']:
        cfg = checkpoint['config']
        config_layers = cfg.get('gcn_layers', cfg.get('gcn_num_layers', 1))
        config_hidden = cfg.get('gcn_hidden_dim', 512)
        adjacency_type = cfg.get('adjacency_type', '4')
        
        # 验证config与state_dict是否一致
        if config_layers != gcn_num_layers_actual or config_hidden != gcn_hidden_dim_actual:
            print(f"⚠️  Config不一致! config说: layers={config_layers}, hidden={config_hidden}")
            print(f"   实际state_dict: layers={gcn_num_layers_actual}, hidden={gcn_hidden_dim_actual}")
            print(f"   ✅ 使用state_dict推断的值")
    
    gcn_num_layers = gcn_num_layers_actual
    gcn_hidden_dim = gcn_hidden_dim_actual
    
    # 对于ViT模型，检查是否有投影层
    vit_native_dim = True  # 默认
    vit_proj_channels = 2048  # 投影维度
    
    if backbone_type == 'vit':
        # 检查投影层：cls_proj 或 patch_proj
        if 'backbone.cls_proj.0.weight' in state_dict or 'backbone.patch_proj.0.weight' in state_dict:
            vit_native_dim = False  # 使用了投影
            if 'backbone.cls_proj.0.weight' in state_dict:
                proj_shape = state_dict['backbone.cls_proj.0.weight'].shape
                vit_proj_channels = proj_shape[0]
                print(f"   检测到 ViT 投影: {proj_shape[1]} → {proj_shape[0]}")
        else:
            vit_native_dim = True   # 使用原生768维
            print(f"   检测到 ViT 原生维度: 768")
    
    print(f"📋 最终配置: hidden_dim={gcn_hidden_dim}, num_layers={gcn_num_layers}, adjacency={adjacency_type}")
    
    # 创建模型（根据backbone类型选择参数）
    if backbone_type == 'vit':
        model = BoTGCN(
            num_classes=num_classes,
            backbone_type='vit',
            vit_model_name='vit_base_patch16_224',
            vit_pretrained=False,  # 从checkpoint加载
            vit_native_dim=vit_native_dim,
            vit_proj_channels=vit_proj_channels,
            gcn_hidden_dim=gcn_hidden_dim,
            gcn_num_layers=gcn_num_layers,
            adjacency_type=adjacency_type,
        )
    else:  # resnet
        model = BoTGCN(
            num_classes=num_classes,
            backbone_type='resnet',
            gcn_hidden_dim=gcn_hidden_dim,
            gcn_num_layers=gcn_num_layers,
            adjacency_type=adjacency_type,
        )
    
    return model

# =====================================================================
# 8 个 VehicleID 消融模型配置
# =====================================================================

ABLATION_MODELS = [
    # CNN 深度消融
    {
        "name":        "CNN+GCN 4nb L1",
        "abl_id":      "ABL-VID-01",
        "group":       "ResNet50",
        "description": "ResNet50-IBN + 4-neighbor + 1 layer GCN",
        "csv_name":    "VID_ABL01_CNN_4nb_L1",
        "ckpt":        "outputs/ablation_vehicleid/cnn_gcn_4nb_l1/best_model.pth",
        "input_size":  (256, 256),
        "backbone_type": "resnet",
    },
    {
        "name":        "CNN+GCN 4nb L2",
        "abl_id":      "ABL-VID-02",
        "group":       "ResNet50",
        "description": "ResNet50-IBN + 4-neighbor + 2 layer GCN",
        "csv_name":    "VID_ABL02_CNN_4nb_L2",
        "ckpt":        "outputs/ablation_vehicleid/cnn_gcn_4nb_l2/best_model.pth",
        "input_size":  (256, 256),
        "backbone_type": "resnet",
    },
    {
        "name":        "CNN+GCN 4nb L3",
        "abl_id":      "ABL-VID-03",
        "group":       "ResNet50",
        "description": "ResNet50-IBN + 4-neighbor + 3 layer GCN",
        "csv_name":    "VID_ABL03_CNN_4nb_L3",
        "ckpt":        "outputs/ablation_vehicleid/cnn_gcn_4nb_l3/best_model.pth",
        "input_size":  (256, 256),
        "backbone_type": "resnet",
    },
    {
        "name":        "CNN+GCN kNN L1",
        "abl_id":      "ABL-VID-04",
        "group":       "ResNet50",
        "description": "ResNet50-IBN + kNN(k=8) + 1 layer GCN",
        "csv_name":    "VID_ABL04_CNN_kNN_L1",
        "ckpt":        "outputs/ablation_vehicleid/cnn_gcn_knn_l1/best_model.pth",
        "input_size":  (256, 256),
        "backbone_type": "resnet",
    },
    # ViT 深度消融（已确认：真正的ViT backbone）
    {
        "name":        "ViT+GCN 4nb L1",
        "abl_id":      "ABL-VID-05",
        "group":       "ViT-Base",
        "description": "ViT-Base + 4-neighbor + 1 layer GCN",
        "csv_name":    "VID_ABL05_ViT_4nb_L1",
        "ckpt":        "outputs/ablation_vehicleid/vit_gcn_4nb_l1/best_model.pth",
        "input_size":  (224, 224),
        "backbone_type": "vit",
    },
    {
        "name":        "ViT+GCN 4nb L2",
        "abl_id":      "ABL-VID-06",
        "group":       "ViT-Base",
        "description": "ViT-Base + 4-neighbor + 2 layer GCN",
        "csv_name":    "VID_ABL06_ViT_4nb_L2",
        "ckpt":        "outputs/ablation_vehicleid/vit_gcn_4nb_l2/best_model.pth",
        "input_size":  (224, 224),
        "backbone_type": "vit",
    },
    {
        "name":        "ViT+GCN 4nb L3",
        "abl_id":      "ABL-VID-07",
        "group":       "ViT-Base",
        "description": "ViT-Base + 4-neighbor + 3 layer GCN",
        "csv_name":    "VID_ABL07_ViT_4nb_L3",
        "ckpt":        "outputs/ablation_vehicleid/vit_gcn_4nb_l3/best_model.pth",
        "input_size":  (224, 224),
        "backbone_type": "vit",
    },
    {
        "name":        "ViT+GCN kNN L1",
        "abl_id":      "ABL-VID-08",
        "group":       "ViT-Base",
        "description": "ViT-Base + kNN(k=8) + 1 layer GCN",
        "csv_name":    "VID_ABL08_ViT_kNN_L1",
        "ckpt":        "outputs/ablation_vehicleid/vit_gcn_knn_l1/best_model.pth",
        "input_size":  (224, 224),
        "backbone_type": "vit",
    },
]

# =====================================================================
# 遮挡评估函数
# =====================================================================

class OcclusionTransform:
    """自定义遮挡增强类，用于评估"""
    def __init__(self, occlusion_level):
        self.occlusion_level = occlusion_level
        self.re = transforms.RandomErasing(
            p=1.0, 
            scale=(occlusion_level, occlusion_level),
            ratio=(1.0, 1.0),
            value=0
        )
    
    def __call__(self, img):
        return self.re(img)

def extract_features(model, dataloader, device):
    """提取特征"""
    model.eval()
    features = []
    labels = []
    camids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # VehicleIDDataset 返回 (img, pid, camid)
            if len(batch) == 3:
                imgs, pids, cams = batch
            else:
                imgs, pids = batch[:2]
                cams = torch.zeros_like(pids)  # VehicleID没有相机ID，用0填充
            
            imgs = imgs.to(device)
            feats = model(imgs)
            feats = F.normalize(feats, p=2, dim=1)
            features.append(feats.cpu())
            labels.append(pids)
            camids.append(cams)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    camids = torch.cat(camids, dim=0)
    return features, labels, camids

def evaluate_model_with_occlusion(model_cfg, dataset_root, occlusion_level, device, test_size="small"):
    """评估单个模型在指定遮挡级别下的性能"""
    
    # 创建数据增强（包含 Random Erasing）
    input_size = model_cfg["input_size"]
    
    # 基础变换（不含遮挡）
    base_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # 添加遮挡变换
    if occlusion_level > 0:
        occlusion_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            OcclusionTransform(occlusion_level),
        ])
    else:
        occlusion_transform = base_transform
    
    # 使用 split_vehicleid_test 分割测试集
    test_list_name = f"test_list_{test_size}.txt"
    query_data, gallery_data = split_vehicleid_test(dataset_root, test_list_name)
    
    # 创建 query 和 gallery 数据集
    query_dataset = VehicleIDDataset(
        root=dataset_root,
        mode='query',
        transform=occlusion_transform,
        test_data=query_data
    )
    
    gallery_dataset = VehicleIDDataset(
        root=dataset_root,
        mode='gallery',
        transform=occlusion_transform,
        test_data=gallery_data
    )
    
    # 创建 DataLoader
    query_loader = DataLoader(
        query_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # 加载模型
    # 从checkpoint构建模型（自动推断GCN配置）
    backbone_type = model_cfg.get("backbone_type", "resnet")
    model = build_model_from_checkpoint(
        checkpoint_path=model_cfg["ckpt"],
        num_classes=13164,  # VehicleID 类别数
        backbone_type=backbone_type,
        device='cpu'
    )
    
    # 使用安全加载函数加载权重
    model = load_checkpoint_safe(model, model_cfg["ckpt"])
    model = model.to(device)
    model.eval()
    
    # 提取特征
    query_feats, query_labels, query_cams = extract_features(model, query_loader, device)
    gallery_feats, gallery_labels, gallery_cams = extract_features(model, gallery_loader, device)
    
    # 评估 (compute_mAP_cmc期望torch.Tensor和np.ndarray混合)
    mAP, cmc = compute_mAP_cmc(
        query_feats,           # torch.Tensor
        gallery_feats,         # torch.Tensor
        query_labels.numpy(),  # np.ndarray
        gallery_labels.numpy(), # np.ndarray
        query_cams.numpy(),    # np.ndarray
        gallery_cams.numpy()   # np.ndarray
    )
    
    # 构造结果字典
    results = {
        'mAP': mAP,
        'CMC': cmc
    }
    
    return results

# =====================================================================
# 主函数
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="data/dataset/VehicleID_V1.0")
    parser.add_argument("--output_dir", default="outputs/ablation_occlusion_results_vehicleid")
    parser.add_argument("--test_size", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = output_dir / "individual_results"
    individual_dir.mkdir(exist_ok=True)
    
    # 遮挡级别（0-30%，每3%一档）
    occlusion_levels = np.arange(0.0, 0.33, 0.03)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    print(f"📊 Test size: {args.test_size}\n")
    
    # 评估所有模型（跳过已有结果的CNN模型）
    for model_cfg in ABLATION_MODELS:
        # 检查是否已有结果
        csv_path = individual_dir / f"{model_cfg['csv_name']}.csv"
        if csv_path.exists():
            print(f"\n⏭️  Skipping {model_cfg['name']} - results already exist: {csv_path}")
            continue
            
        print(f"\n{'='*70}")
        print(f"📊 Evaluating: {model_cfg['name']} ({model_cfg['abl_id']})")
        print(f"{'='*70}\n")
        
        results_list = []
        
        for occ_level in occlusion_levels:
            occ_pct = int(occ_level * 100)
            print(f"  🔍 Occlusion {occ_pct}%...", end=" ", flush=True)
            
            try:
                results = evaluate_model_with_occlusion(
                    model_cfg, args.dataset_root, occ_level, device, args.test_size
                )
                
                results_list.append({
                    "Model": model_cfg["name"],
                    "ABL_ID": model_cfg["abl_id"],
                    "Occlusion_Level": occ_pct,
                    "mAP": results["mAP"],
                    "Rank-1": results["CMC"][0],
                    "Rank-5": results["CMC"][4],
                    "Rank-10": results["CMC"][9],
                })
                
                print(f"mAP: {results['mAP']:.4f}, Rank-1: {results['CMC'][0]:.4f}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 保存单个模型结果
        if results_list:
            df = pd.DataFrame(results_list)
            csv_path = individual_dir / f"{model_cfg['csv_name']}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n✅ Saved: {csv_path}")
    
    print(f"\n{'='*70}")
    print("🎉 All evaluations completed!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
