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
from models.gcn import SimpleGCN, GraphPooling
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
        # CNN配置
        last_stride=1,
        pretrain_path='',
        # GCN配置
        use_gcn=True,
        grid_h=4,
        grid_w=4,
        gcn_hidden_dim=512,
        gcn_num_layers=1,
        gcn_dropout=0.5,
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
            last_stride: 最后一个卷积层的stride
            pretrain_path: 预训练权重路径
            
            use_gcn: 是否使用GCN分支
            grid_h, grid_w: 网格大小
            gcn_hidden_dim: GCN隐藏层维度
            gcn_num_layers: GCN层数
            gcn_dropout: GCN dropout
            
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
        
        # ========== 1. CNN Backbone ==========
        self.backbone = resnet50_ibn_a(pretrained=False)
        
        # 修改last_stride (如果需要)
        if last_stride == 1:
            self.backbone.layer4[0].conv2.stride = (1, 1)
            self.backbone.layer4[0].downsample[0].stride = (1, 1)
        
        # 如果有预训练权重，加载
        if pretrain_path:
            self.load_pretrained_weights(pretrain_path)
        
        # ========== 2. 全局分支 (GAP) ==========
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # ========== 3. 图分支 (如果启用) ==========
        if self.use_gcn:
            # 3.1 网格池化
            self.grid_pooling = GridPooling(grid_h, grid_w)
            self.num_nodes = grid_h * grid_w
            
            # 3.2 GCN层
            self.gcn = SimpleGCN(
                in_channels=2048,
                hidden_channels=gcn_hidden_dim,
                out_channels=2048,
                num_layers=gcn_num_layers,
                dropout=gcn_dropout
            )
            
            # 3.3 图池化
            if pooling_type == 'attention':
                self.graph_pooling = GraphPooling(
                    pooling_type=pooling_type,
                    in_channels=2048,
                    hidden_channels=pooling_hidden_dim
                )
            else:
                self.graph_pooling = GraphPooling(pooling_type=pooling_type)
            
            # 3.4 预计算邻接矩阵 (4-neighbor grid)
            self.register_buffer('edge_index', self._build_grid_adjacency(grid_h, grid_w))
            
            # 3.5 嵌入融合
            self.embedding_fusion = EmbeddingFusion(
                fusion_type=fusion_type,
                global_dim=2048,
                graph_dim=2048,
                output_dim=2048,
                hidden_dim=fusion_hidden_dim,
                dropout=fusion_dropout
            )
        
        # ========== 4. BNNeck (Batch Normalization Neck) ==========
        if self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(2048)
            self.bottleneck.bias.requires_grad_(False)  # 不学习bias
            nn.init.constant_(self.bottleneck.weight, 1)
            nn.init.constant_(self.bottleneck.bias, 0)
        else:
            self.bottleneck = None
        
        # ========== 5. Classifier ==========
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
    
    def _build_grid_adjacency(self, grid_h, grid_w):
        """
        构建网格图的邻接矩阵 (4-neighbor)
        
        Returns:
            edge_index: (2, E) 边索引
        """
        num_nodes = grid_h * grid_w
        edges = []
        
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
        
        # 1. CNN特征提取
        feat_map = self.backbone(x)  # (B, 2048, 8, 8)
        
        # 2. 全局分支
        global_emb = self.gap(feat_map).view(feat_map.size(0), -1)  # (B, 2048)
        
        # 3. 图分支 (如果启用)
        if self.use_gcn:
            # 3.1 网格池化: (B, 2048, 8, 8) -> (B, N, 2048)
            nodes = self.grid_pooling(feat_map)  # (B, N, 2048)
            B, N, C = nodes.shape
            
            # 3.2 Reshape for GCN: (B, N, C) -> (B*N, C)
            nodes_flat = nodes.view(B * N, C)
            
            # 3.3 Expand edge_index for batch
            # edge_index: (2, E), need (2, B*E)
            edge_index = self.edge_index
            edge_index_batch = []
            for b in range(B):
                edge_index_batch.append(edge_index + b * N)
            edge_index_batch = torch.cat(edge_index_batch, dim=1)  # (2, B*E)
            
            # 3.4 GCN处理 (禁用AMP因为稀疏矩阵不支持FP16)
            with torch.amp.autocast('cuda', enabled=False):
                nodes_flat_fp32 = nodes_flat.float()
                nodes_gcn = self.gcn(nodes_flat_fp32, edge_index_batch)  # (B*N, C)
                if nodes_flat.dtype == torch.float16:
                    nodes_gcn = nodes_gcn.half()
            
            # 3.5 Reshape back: (B*N, C) -> (B, N, C)
            nodes_gcn = nodes_gcn.view(B, N, C)
            
            # 3.6 图池化: (B, N, C) -> (B, C)
            # 需要为每个batch创建batch索引
            batch_idx = torch.arange(B, device=nodes.device).repeat_interleave(N)
            nodes_gcn_flat = nodes_gcn.view(B * N, C)
            
            graph_emb = self.graph_pooling(nodes_gcn_flat, batch_idx)  # (B, C)
            
            # 3.7 嵌入融合
            fused_emb, fusion_extra = self.embedding_fusion(global_emb, graph_emb)
            extra.update(fusion_extra)
            
            final_emb = fused_emb
        else:
            # 只使用全局嵌入
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
