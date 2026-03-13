#!/usr/bin/env python3
"""
批量生成 VehicleID 数据集的消融实验配置文件
对应 VeRi-776 的 ABL-01 到 ABL-15
"""

import os
import yaml

# 配置模板
CONFIGS = [
    # CNN + GCN 深度消融 (对应 ABL-01, 04, 05)
    {"name": "abl_vehicleid_cnn_gcn_4nb_l1", "backbone": "resnet50_ibn_a", "feature_dim": 2048, "adj_type": "4", "layers": 1, "gnn_type": "gcn"},
    {"name": "abl_vehicleid_cnn_gcn_4nb_l2", "backbone": "resnet50_ibn_a", "feature_dim": 2048, "adj_type": "4", "layers": 2, "gnn_type": "gcn"},
    {"name": "abl_vehicleid_cnn_gcn_4nb_l3", "backbone": "resnet50_ibn_a", "feature_dim": 2048, "adj_type": "4", "layers": 3, "gnn_type": "gcn"},
    
    # CNN + GCN k-NN (对应 ABL-08)
    {"name": "abl_vehicleid_cnn_gcn_knn_l1", "backbone": "resnet50_ibn_a", "feature_dim": 2048, "adj_type": "knn", "layers": 1, "gnn_type": "gcn"},
    
    # ViT + GCN 深度消融 (对应 ABL-02, 11, 12)
    {"name": "abl_vehicleid_vit_gcn_4nb_l1", "backbone": "vit_base_patch16_224", "feature_dim": 768, "adj_type": "4", "layers": 1, "gnn_type": "gcn", "input_size": 224},
    {"name": "abl_vehicleid_vit_gcn_4nb_l2", "backbone": "vit_base_patch16_224", "feature_dim": 768, "adj_type": "4", "layers": 2, "gnn_type": "gcn", "input_size": 224},
    {"name": "abl_vehicleid_vit_gcn_4nb_l3", "backbone": "vit_base_patch16_224", "feature_dim": 768, "adj_type": "4", "layers": 3, "gnn_type": "gcn", "input_size": 224},
    
    # ViT + GCN k-NN (对应 ABL-15)
    {"name": "abl_vehicleid_vit_gcn_knn_l1", "backbone": "vit_base_patch16_224", "feature_dim": 768, "adj_type": "knn", "layers": 1, "gnn_type": "gcn", "input_size": 224},
]

def create_config(cfg):
    """创建单个配置文件"""
    is_vit = "vit" in cfg["backbone"]
    input_size = cfg.get("input_size", 256 if not is_vit else 224)
    
    config = {
        "DATA": {
            "ROOT": "data/dataset/VehicleID_V1.0",
            "BATCH_SIZE": 64,
            "NUM_WORKERS": 24,  # VehicleID 数据量大，增加 workers
            "USE_PK_SAMPLER": True,
            "P": 16,
            "K": 4,
            "TEST_SIZE": "small",  # VehicleID 特有参数
            "AUGMENTATION": {
                "TYPE": "random_erasing",
                "PROBABILITY": 0.5,
                "PARAMS": {
                    "sl": 0.02,
                    "sh": 0.2,
                    "r1": 0.3,
                    "r2": 3.33,
                    "mode": "random"
                }
            }
        },
        "MODEL": {
            "NAME": "BoTGCN",
            "BACKBONE": cfg["backbone"],
            "PRETRAINED": True,
            "PRETRAINED_PATH": "outputs/bot_baseline_1_1/VehicleID/baseline_run_01/best_model.pth",
            "NUM_CLASSES": 13164,  # VehicleID 训练集类别数
            "FEATURE_DIM": cfg["feature_dim"],
            "LAST_STRIDE": 1,
            "GCN": {
                "USE_GCN": True,
                "GRID_H": 4,
                "GRID_W": 4,
                "ADJACENCY_TYPE": cfg["adj_type"],
                "HIDDEN_CHANNELS": 512,
                "OUT_CHANNELS": None,
                "NUM_LAYERS": cfg["layers"],
                "DROPOUT": 0.5,
                "KNN_K": 8,
                "KNN_METRIC": "cosine",
                "KNN_DETACH": True,
                "GNN_TYPE": cfg["gnn_type"],
                "POOLING_TYPE": "mean",
                "POOLING_HIDDEN_DIM": 128
            },
            "FUSION": {
                "TYPE": "concat",
                "HIDDEN_DIM": 512,
                "DROPOUT": 0.5
            }
        },
        "OPTIMIZER": {
            "NAME": "AdamW",
            "BASE_LR": 3.5e-5 if is_vit else 1e-4,  # ViT 用更小的学习率
            "WEIGHT_DECAY": 0.0005,
            "EPS": 1e-8
        },
        "SCHEDULER": {
            "NAME": "WarmupCosineAnnealingLR",
            "WARMUP_EPOCHS": 10
        },
        "LOSS": {
            "LABEL_SMOOTH": 0.1,
            "TRIPLET_MARGIN": 0.3,
            "ID_LOSS_WEIGHT": 1.0,
            "TRIPLET_LOSS_WEIGHT": 1.0
        },
        "EPOCHS": 120,
        "EVAL_PERIOD": 10,
        "LOG_PERIOD": 100,  # VehicleID 数据多，减少日志频率
        "AUGMENTATION": {
            "RANDOM_ERASING": True,
            "RANDOM_ERASING_PROB": 0.5
        },
        "USE_AMP": True,
        "OUTPUT": {
            "DIR": f"outputs/ablation_vehicleid/{cfg['name'].replace('abl_vehicleid_', '')}"
        },
        "SEED": 42
    }
    
    # 如果是 ViT，添加 input_size
    if is_vit:
        config["DATA"]["INPUT_SIZE"] = input_size
    
    return config

def main():
    output_dir = "configs/gcn_transformer_configs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 生成 VehicleID 消融实验配置文件...")
    print(f"📁 输出目录: {output_dir}\n")
    
    for cfg in CONFIGS:
        config = create_config(cfg)
        filename = f"{cfg['name']}.yaml"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        print(f"✅ {filename}")
        print(f"   Backbone: {cfg['backbone']}, Adj: {cfg['adj_type']}, Layers: {cfg['layers']}")
    
    print(f"\n✨ 完成！共生成 {len(CONFIGS)} 个配置文件")

if __name__ == "__main__":
    main()
