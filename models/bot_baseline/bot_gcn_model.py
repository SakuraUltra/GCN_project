"""
BoT-GCN Model for Vehicle Re-identification
基于 BoT-Baseline 集成 GCN 和图池化

架构流程:
1. CNN Backbone (ResNet50-IBN) -> 特征图 (2048, 8, 8)
2. 双分支处理:
   a) 全局分支: GAP -> 全局嵌入 (2048)
   b) 图分支: Grid Pooling -> 图节点 (N, 2048) -> GCN -> Graph Pooling -> 图嵌入 (2048)
3. 嵌入融合: 全局嵌入 + 图嵌入 -> 最终嵌入 (2048)
4. BNNeck + Classifier

与 BoT-Baseline 对比:
- Baseline: CNN -> GAP -> BNNeck -> Classifier
- Ours: CNN -> (GAP + GCN) -> Fusion -> BNNeck -> Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.backbones.resnet_ibn import resnet50_ibn_a
from models.transformer.vit_backbone import ViTBackbone
from models.gcn import SimpleGCN, GraphPooling, KNNEdgeBuilder, HybridEdgeBuilder
from models.gcn.gat_conv import SimpleGAT
from models.fusion.embedding_fusion import EmbeddingFusion


class GridPooling(nn.Module):
    """
    网格池化: 将特征图划分为网格节点
    
    Input: (B, C, H, W) 特征图
    Output: (B, grid_h * grid_w, C) 图节点
    """
    
    def __init__(self, grid_h=4, grid_w=4):
        super(GridPooling, self).__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
    
    def forward(self, feat_map):
        """
        Args:
            feat_map: (B, C, H, W) 特征图
        
        Returns:
            nodes: (B, N, C) 图节点，N = grid_h * grid_w
        """
        B, C, H, W = feat_map.shape
        
        # 计算每个网格的高度和宽度
        grid_h_size = H // self.grid_h
        grid_w_size = W // self.grid_w
        
        nodes = []
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                # 提取网格区域
                h_start = i * grid_h_size
                h_end = (i + 1) * grid_h_size if i < self.grid_h - 1 else H
                w_start = j * grid_w_size
                w_end = (j + 1) * grid_w_size if j < self.grid_w - 1 else W
                
                grid_region = feat_map[:, :, h_start:h_end, w_start:w_end]  # (B, C, h, w)
                
                # 对网格区域进行平均池化
                node_feat = F.adaptive_avg_pool2d(grid_region, (1, 1)).squeeze(-1).squeeze(-1)  # (B, C)
                nodes.append(node_feat)
        
        # Stack所有节点
        nodes = torch.stack(nodes, dim=1)  # (B, N, C)
        
        return nodes


class BoTGCN(nn.Module):
    """
    BoT-GCN: BoT-Baseline + GCN + Graph Pooling + Embedding Fusion
    """
    
    def __init__(
        self,
        num_classes,
        # Backbone配置
        backbone_type='vit',       # ← 新增：'resnet' or 'vit'
        last_stride=1,
        pretrain_path='',
        # ViT配置 (当 backbone_type='vit' 时生效)
        vit_model_name="deit_small_patch16_224.fb_in1k",
        vit_pretrained=True,
        vit_native_dim=True,       # ← VIT-25: 原生维度开关
        vit_proj_channels=2048,    # ← native_dim=False 时生效
        vit_target_spatial=8,      # ← native_dim=False 时生效
        # GCN配置
        use_gcn=True,
        grid_h=4,
        grid_w=4,
        adjacency_type='4',  # '4', '8', 'knn', or 'hybrid'
        gcn_hidden_dim=512,
        gcn_out_dim=None,        # ← VIT-25: GCN输出维度（None=与feat_dim一致）
        gcn_num_layers=1,
        gcn_dropout=0.5,
        # GAT 配置 (VIT-26)
        gnn_type='gcn',       # 'gcn' or 'gat'
        gat_heads=4,          # GAT 多头注意力数量
        # kNN 配置 (当 adjacency_type='knn' 或 'hybrid' 时生效)
        knn_k=4,
        knn_metric='cosine',  # 'cosine' or 'euclidean'
        knn_detach=True,      # 是否停止 kNN 的特征梯度
        # 图池化配置
        pooling_type='mean',  # 'mean', 'max', 'attention'
        pooling_hidden_dim=128,
        # 融合配置
        fusion_type='concat',  # 'concat', 'gated', 'add', 'none'
        fusion_hidden_dim=512,
        fusion_dropout=0.5,
        # 其他
        neck='bnneck'
    ):
        """
        Args:
            num_classes: 类别数
            backbone_type: Backbone类型 ('resnet' or 'vit')
            last_stride: 最后一个卷积层的stride (ResNet)
            pretrain_path: 预训练权重路径
            
            vit_native_dim: ViT 原生维度模式 (True=768/384, False=2048)
            vit_proj_channels: 投影维度 (仅 native_dim=False 时生效)
            vit_target_spatial: 目标空间尺寸 (仅 native_dim=False 时生效)
            
            use_gcn: 是否使用GCN分支
            grid_h, grid_w: 网格大小
            adjacency_type: 邻接类型 ('4'=4-neighbor, '8'=8-neighbor, 'knn'=动态kNN, 'hybrid'=固定+kNN)
            gcn_hidden_dim: GCN隐藏层维度
            gcn_num_layers: GCN层数
            gcn_dropout: GCN dropout
            
            knn_k: kNN 邻居数量
            knn_metric: kNN 相似度度量 ('cosine' or 'euclidean')
            knn_detach: 是否停止 kNN 特征梯度
            
            pooling_type: 图池化类型
            pooling_hidden_dim: Attention pooling隐藏层维度
            
            fusion_type: 融合类型
            fusion_hidden_dim: 门控融合隐藏层维度
            fusion_dropout: 融合层dropout
            
            neck: 'bnneck' or 'no'
        """
        super(BoTGCN, self).__init__()
        
        self.num_classes = num_classes
        self.use_gcn = use_gcn
        self.neck = neck
        self.fusion_type = fusion_type
        self.adjacency_type = adjacency_type
        self.knn_k = knn_k
        self.knn_metric = knn_metric
        self.gnn_type = gnn_type  # 'gcn' or 'gat'
        self.backbone_type = backbone_type
        self.is_vit = (backbone_type == 'vit')  # 标记是否为 ViT
        
        # ========== 1. Backbone ==========
        if backbone_type == 'resnet':
            # ResNet Backbone (不支持 last_stride 参数)
            self.backbone = resnet50_ibn_a()
            feat_dim = 2048  # ResNet50 输出维度
            # 注意: 预训练权重在训练脚本中统一加载
        elif backbone_type == 'vit':
            # ViT Backbone - 模块化版本
            self.backbone = ViTBackbone(
                model_name=vit_model_name,
                pretrained=vit_pretrained,
                native_dim=vit_native_dim,
                proj_channels=vit_proj_channels,
                target_spatial=vit_target_spatial,
            )
            feat_dim = self.backbone.out_dim  # 768 (native) or 2048 (projected)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}. Must be 'resnet' or 'vit'")
        
        # GCN 输出维度（可与 feat_dim 不同）
        graph_dim = gcn_out_dim if gcn_out_dim is not None else feat_dim
        self.graph_dim = graph_dim  # 保存为实例变量供forward使用
        
        # ========== 2. 全局分支 (CLS token 直接替代 GAP) ==========
        
        # ========== 3. 图分支 (如果启用) ==========
        if self.use_gcn:
            # 3.1 网格池化
            self.grid_pooling = GridPooling(grid_h, grid_w)
            self.num_nodes = grid_h * grid_w
            
            # 3.2 GCN/GAT层（维度跟随 feat_dim）
            if gnn_type.lower() == 'gat':
                self.gcn = SimpleGAT(
                    in_channels=feat_dim,
                    hidden_channels=gcn_hidden_dim,
                    out_channels=graph_dim,
                    num_layers=gcn_num_layers,
                    heads=gat_heads,
                    dropout=gcn_dropout
                )
                self.gnn_type = 'gat'
            else:
                self.gcn = SimpleGCN(
                    in_channels=feat_dim,
                    hidden_channels=gcn_hidden_dim,
                    out_channels=graph_dim,
                    num_layers=gcn_num_layers,
                    dropout=gcn_dropout
                )
                self.gnn_type = 'gcn'
            
            # 3.3 图池化
            if pooling_type == 'attention':
                self.graph_pooling = GraphPooling(
                    pooling_type=pooling_type,
                    in_channels=graph_dim,
                    hidden_channels=pooling_hidden_dim
                )
            else:
                self.graph_pooling = GraphPooling(pooling_type=pooling_type)
            
            # 3.4 边构建器
            if adjacency_type in ['4', '8']:
                # 固定网格邻接（预计算）
                self.register_buffer('edge_index', self._build_grid_adjacency(grid_h, grid_w, adjacency_type))
                self.dynamic_edges = False
                self.edge_builder = None
            elif adjacency_type == 'knn':
                # 动态 kNN 边
                self.edge_index = None
                self.dynamic_edges = True
                self.edge_builder = KNNEdgeBuilder(k=knn_k, metric=knn_metric, detach_features=knn_detach)
            elif adjacency_type == 'hybrid':
                # 混合边: 固定网格 + 动态 kNN
                self.edge_index = None
                self.dynamic_edges = True
                self.edge_builder = HybridEdgeBuilder(
                    grid_h=grid_h, grid_w=grid_w, 
                    adjacency_type='4',  # 默认使用 4-neighbor 网格
                    knn_k=knn_k, knn_metric=knn_metric
                )
            else:
                raise ValueError(f"Invalid adjacency_type: {adjacency_type}. Must be '4', '8', 'knn', or 'hybrid'.")
            
            # 3.5 嵌入融合（维度跟随 feat_dim）
            self.embedding_fusion = EmbeddingFusion(
                fusion_type=fusion_type,
                global_dim=feat_dim,
                graph_dim=graph_dim,
                output_dim=graph_dim,
                hidden_dim=fusion_hidden_dim,
                dropout=fusion_dropout
            )
        
        # ========== 4. CLS Fusion Layer (仅 ViT+GCN 时使用) ==========
        # 注意: ResNet 的 cls_emb 等于 global_emb，无需二次融合
        if self.use_gcn and self.is_vit:
            # cls_emb(feat_dim) + graph_emb(graph_dim) -> graph_dim
            self.cls_fusion = nn.Sequential(
                nn.Linear(feat_dim + graph_dim, graph_dim, bias=False),
                nn.BatchNorm1d(graph_dim),
                nn.ReLU(inplace=True),
            )
            final_dim = graph_dim
        else:
            final_dim = graph_dim if self.use_gcn else feat_dim
        
        # ========== 5. BNNeck (Batch Normalization Neck，维度跟随 feat_dim) ==========
        if self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(final_dim)
            self.bottleneck.bias.requires_grad_(False)  # 不学习bias
            nn.init.constant_(self.bottleneck.weight, 1)
            nn.init.constant_(self.bottleneck.bias, 0)
        else:
            self.bottleneck = None
        
        # ========== 6. Classifier (维度跟随 feat_dim) ==========
        self.classifier = nn.Linear(final_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
    
    def _build_grid_adjacency(self, grid_h, grid_w, adjacency_type='4'):
        """
        构建网格图的邻接矩阵
        
        Args:
            grid_h: 网格高度
            grid_w: 网格宽度
            adjacency_type: '4' for 4-neighbor, '8' for 8-neighbor
        
        Returns:
            edge_index: (2, E) 边索引
        """
        num_nodes = grid_h * grid_w
        edges = []
        
        if adjacency_type == '4':
            # 4-邻居: 上、下、左、右
            for i in range(grid_h):
                for j in range(grid_w):
                    node_id = i * grid_w + j
                    
                    # 上
                    if i > 0:
                        neighbor = (i - 1) * grid_w + j
                        edges.append([node_id, neighbor])
                    
                    # 下
                    if i < grid_h - 1:
                        neighbor = (i + 1) * grid_w + j
                        edges.append([node_id, neighbor])
                    
                    # 左
                    if j > 0:
                        neighbor = i * grid_w + (j - 1)
                        edges.append([node_id, neighbor])
                    
                    # 右
                    if j < grid_w - 1:
                        neighbor = i * grid_w + (j + 1)
                        edges.append([node_id, neighbor])
        
        elif adjacency_type == '8':
            # 8-邻居: 4-邻居 + 4个对角线
            directions = [
                (-1, 0),   # 上
                (1, 0),    # 下
                (0, -1),   # 左
                (0, 1),    # 右
                (-1, -1),  # 左上
                (-1, 1),   # 右上
                (1, -1),   # 左下
                (1, 1),    # 右下
            ]
            
            for i in range(grid_h):
                for j in range(grid_w):
                    node_id = i * grid_w + j
                    
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        # 检查边界
                        if 0 <= ni < grid_h and 0 <= nj < grid_w:
                            neighbor_id = ni * grid_w + nj
                            edges.append([node_id, neighbor_id])
        
        else:
            raise ValueError(f"Invalid adjacency_type: {adjacency_type}. Must be '4' or '8'.")
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def load_pretrained_weights(self, pretrain_path):
        """加载预训练权重"""
        import logging
        logger = logging.getLogger('BoT-GCN-Training')
        
        if not os.path.exists(pretrain_path):
            logger.warning(f"Pretrained path does not exist: {pretrain_path}")
            return
        
        state_dict = torch.load(pretrain_path, map_location='cpu')
        
        # 处理可能的嵌套字典
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # 加载backbone和bottleneck的权重（跳过classifier和GCN相关层）
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # 加载backbone
            if k.startswith('backbone.') and k in model_dict:
                pretrained_dict[k] = v
            # 加载bottleneck（BNNeck）
            elif k.startswith('bottleneck.') and k in model_dict:
                pretrained_dict[k] = v
            # 跳过classifier（类别数可能不同）
            # 跳过GCN、fusion等新增层（它们不在baseline中）
        
        # 加载权重
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        
        logger.info(f"✅ Loaded pretrained weights from {pretrain_path}")
        logger.info(f"   Loaded {len(pretrained_dict)} / {len(state_dict)} layers from checkpoint")
        logger.info(f"   Backbone: ✓, Bottleneck: ✓, Classifier: ✗ (training from scratch)")
        logger.info(f"   GCN/Fusion: ✗ (new modules, training from scratch)")
    
    def forward(self, x, return_extra=False):
        """
        Args:
            x: (B, 3, H, W) 输入图像
            return_extra: 是否返回额外信息 (用于可视化)
        
        Returns:
            if training:
                logits: (B, num_classes) 分类logits
                feat: (B, 2048) 特征 (用于triplet loss)
            if not training:
                feat: (B, 2048) L2归一化后的特征
        """
        extra = {}
        
        # 1. Backbone 特征提取
        if self.backbone_type == 'resnet':
            feat_map = self.backbone(x)  # (B, 2048, 8, 8)
            # ResNet 没有 CLS token，使用 GAP 作为全局特征
            cls_emb = F.adaptive_avg_pool2d(feat_map, (1, 1)).view(feat_map.size(0), -1)  # (B, 2048)
        elif self.backbone_type == 'vit':
            feat_map, cls_emb = self.backbone(x)  # feat_map: (B, 2048, 8, 8), cls_emb: (B, 2048)
        
        # 2. 全局分支 (CLS token 或 GAP)
        global_emb = cls_emb  # (B, 2048)
        
        # 3. 图分支 (如果启用)
        if self.use_gcn:
            # 3.1 网格池化: (B, 2048, 8, 8) -> (B, N, 2048)
            nodes = self.grid_pooling(feat_map)  # (B, N, 2048)
            B, N, C = nodes.shape
            
            # 3.2 边构建
            if self.dynamic_edges:
                # 动态构建边 (kNN 或 hybrid)
                if isinstance(self.edge_builder, KNNEdgeBuilder):
                    edge_index_list, edge_weight_list = self.edge_builder(nodes)
                elif isinstance(self.edge_builder, HybridEdgeBuilder):
                    edge_index_list = self.edge_builder(nodes)
                    edge_weight_list = None
                else:
                    raise ValueError(f"Unknown edge builder: {type(self.edge_builder)}")
                
                # 合并所有样本的边 (加上偏移量)
                edge_index_batch = []
                for b, edge_idx in enumerate(edge_index_list):
                    edge_index_batch.append(edge_idx + b * N)
                edge_index_batch = torch.cat(edge_index_batch, dim=1)  # (2, total_E)
            else:
                # 使用预计算的固定边
                edge_index = self.edge_index
                edge_index_batch = []
                for b in range(B):
                    edge_index_batch.append(edge_index + b * N)
                edge_index_batch = torch.cat(edge_index_batch, dim=1)  # (2, B*E)
            
            # 3.3 Reshape for GCN: (B, N, C) -> (B*N, C)
            nodes_flat = nodes.view(B * N, C)
            
            # 3.4 GCN处理 (禁用AMP因为稀疏矩阵不支持FP16)
            with torch.amp.autocast('cuda', enabled=False):
                nodes_flat_fp32 = nodes_flat.float()
                nodes_gcn = self.gcn(nodes_flat_fp32, edge_index_batch)  # (B*N, graph_dim)
                if nodes_flat.dtype == torch.float16:
                    nodes_gcn = nodes_gcn.half()
            
            # 3.5 Reshape back: (B*N, graph_dim) -> (B, N, graph_dim)
            nodes_gcn = nodes_gcn.view(B, N, self.graph_dim)
            
            # 3.6 图池化: (B, N, graph_dim) -> (B, graph_dim)
            # 需要为每个batch创建batch索引
            batch_idx = torch.arange(B, device=nodes.device).repeat_interleave(N)
            nodes_gcn_flat = nodes_gcn.view(B * N, self.graph_dim)
            
            graph_emb = self.graph_pooling(nodes_gcn_flat, batch_idx)  # (B, C)
            
            # 3.7 嵌入融合
            fused_emb, fusion_extra = self.embedding_fusion(global_emb, graph_emb)
            extra.update(fusion_extra)
            
            # 3.8 CLS 与 fused_emb 的二次融合 (仅 ViT)
            if self.is_vit:
                # ViT: CLS token 与融合特征二次融合
                fused_emb = self.cls_fusion(
                    torch.cat([cls_emb, fused_emb], dim=1)  # (B, 4096) -> (B, 2048)
                )
            # ResNet: cls_emb == global_emb，已在 fusion 中使用，无需二次融合
            
            final_emb = fused_emb
        else:
            # 只使用全局嵌入 (CLS token)
            final_emb = global_emb
        
        # 4. BNNeck
        if self.bottleneck is not None:
            bn_feat = self.bottleneck(final_emb)
        else:
            bn_feat = final_emb
        
        # 5. 训练 vs 推理
        if self.training:
            # 训练: 返回logits和特征
            logits = self.classifier(bn_feat)
            
            if return_extra:
                return logits, final_emb, extra
            else:
                return logits, final_emb
        else:
            # 推理: 返回L2归一化特征 (使用BN之前的特征,与triplet loss一致)
            feat_norm = F.normalize(final_emb, p=2, dim=1)
            
            if return_extra:
                return feat_norm, extra
            else:
                return feat_norm


def test_bot_gcn():
    """测试BoT-GCN模型"""
    print("=" * 80)
    print("🧪 测试 BoT-GCN 模型")
    print("=" * 80)
    
    batch_size = 4
    num_classes = 576  # VeRi-776
    
    # 模拟输入
    x = torch.randn(batch_size, 3, 256, 256)
    
    print(f"\n输入:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Input Shape: {x.shape}")
    print(f"  Num Classes: {num_classes}")
    
    # 测试不同配置
    configs = [
        {
            'name': 'Baseline (No GCN)',
            'use_gcn': False
        },
        {
            'name': 'GCN + Mean Pooling + Concat Fusion',
            'use_gcn': True,
            'grid_h': 4,
            'grid_w': 4,
            'pooling_type': 'mean',
            'fusion_type': 'concat'
        },
        {
            'name': 'GCN + Attention Pooling + Gated Fusion',
            'use_gcn': True,
            'grid_h': 4,
            'grid_w': 4,
            'pooling_type': 'attention',
            'fusion_type': 'gated'
        },
    ]
    
    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"测试配置: {config['name']}")
        print(f"{'=' * 80}")
        
        config_copy = config.copy()
        name = config_copy.pop('name')
        
        model = BoTGCN(num_classes=num_classes, **config_copy)
        model.eval()
        
        # 参数量
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n模型统计:")
        print(f"  总参数: {num_params:,}")
        print(f"  可训练参数: {num_trainable:,}")
        
        # 推理模式
        with torch.no_grad():
            feat = model(x)
        
        print(f"\n推理输出:")
        print(f"  特征形状: {feat.shape}")
        print(f"  特征范围: [{feat.min().item():.4f}, {feat.max().item():.4f}]")
        print(f"  特征norm: {feat.norm(dim=1).mean().item():.4f} (应接近1.0)")
        
        # 训练模式
        model.train()
        logits, emb = model(x)
        
        print(f"\n训练输出:")
        print(f"  Logits形状: {logits.shape}")
        print(f"  嵌入形状: {emb.shape}")
        
        # 梯度检查
        loss = logits.sum() + emb.sum()
        loss.backward()
        
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.parameters())
        
        print(f"  梯度流动: {'✓' if has_grad else '✗'}")
    
    print(f"\n{'=' * 80}")
    print(f"✅ BoT-GCN 模型测试完成!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    test_bot_gcn()
