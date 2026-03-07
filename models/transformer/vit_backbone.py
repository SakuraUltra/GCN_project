"""
ViT Backbone - 模块化版本（支持两种模式）
对应任务卡: VIT-25

通过 native_dim=True/False 控制：
  native_dim=True  → 原生维度（768-dim），零信息损失，推荐
  native_dim=False → 投影模式（→2048-dim），兼容旧 GCN 权重
"""

import math
import torch
import torch.nn as nn
import timm


class ViTBackbone(nn.Module):
    """
    统一 ViT Backbone，通过 native_dim 开关切换两种模式。

    输入:  (B, 3, 224, 224)
    输出:  feat_map  (B, out_dim, Hp, Wp)
           cls_emb   (B, out_dim)

    native_dim=True  → out_dim = embed_dim (768/384), Hp=Wp=14
    native_dim=False → out_dim = proj_channels (2048), Hp=Wp=target_spatial
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k",
        pretrained: bool = True,
        native_dim: bool = True,       # ← 核心开关
        proj_channels: int = 2048,     # 仅 native_dim=False 时生效
        target_spatial: int = 8,       # 仅 native_dim=False 时生效
    ):
        super().__init__()

        self.native_dim = native_dim

        # ① 加载 timm ViT
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        vit_dim = self.vit.embed_dim   # 768 (Base) or 384 (Small)

        # ② patch 网格
        patch_size = self.vit.patch_embed.patch_size
        self.patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size
        self.Hp = self.Hw = 224 // self.patch_size   # 14

        if native_dim:
            # ── 原生模式：只做 LayerNorm，不投影 ──────────────────
            self.out_dim = vit_dim
            self.cls_norm = nn.LayerNorm(vit_dim)

        else:
            # ── 投影模式：vit_dim → proj_channels ─────────────────
            self.out_dim = proj_channels
            self.patch_proj = nn.Sequential(
                nn.Conv2d(vit_dim, proj_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(proj_channels),
                nn.ReLU(inplace=True),
            )
            self.spatial_adapt = nn.AdaptiveAvgPool2d((target_spatial, target_spatial))
            self.cls_proj = nn.Sequential(
                nn.Linear(vit_dim, proj_channels, bias=False),
                nn.BatchNorm1d(proj_channels),
            )


    def forward(self, x: torch.Tensor):
        B = x.size(0)
        tokens = self.vit.forward_features(x)          # (B, N+1, vit_dim)

        cls_token    = tokens[:, 0, :]                  # (B, vit_dim)
        patch_tokens = tokens[:, 1:, :]                 # (B, N, vit_dim)

        feat_map = (
            patch_tokens
            .transpose(1, 2)
            .reshape(B, self.vit.embed_dim, self.Hp, self.Hw)
        )

        if self.native_dim:
            cls_emb = self.cls_norm(cls_token)          # (B, D)
            # feat_map 保持 (B, D, 14, 14)
        else:
            cls_emb  = self.cls_proj(cls_token)         # (B, 2048)
            feat_map = self.patch_proj(feat_map)        # (B, 2048, 14, 14)
            feat_map = self.spatial_adapt(feat_map)     # (B, 2048, 8, 8)

        return feat_map, cls_emb


    def _log_shapes(self):
        mode    = "native" if self.native_dim else "projected"
        spatial = f"{self.Hp}×{self.Hw}" if self.native_dim else "8×8(after pool)"
        print(
            f"[ViTBackbone:{mode}] embed_dim={self.vit.embed_dim} | "
            f"out_dim={self.out_dim} | patch_grid={self.Hp}×{self.Hw}"
        )
        print(
            f"[ViTBackbone:{mode}] output: "
            f"feat_map=[B,{self.out_dim},{spatial}]  cls_emb=[B,{self.out_dim}]"
        )

