"""
BoT-Baseline for Vehicle Re-identification
ResNet50-IBN + BNNeck + ID Loss + Triplet Loss

Based on "Bag of Tricks and A Strong Baseline for Deep Person Re-identification"
Adapted for Vehicle Re-ID (VeRi-776 dataset)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
# Add project root to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.gcn_lib.graph_generator import GridGraphGenerator


class IBN(nn.Module):
    """Instance-Batch Normalization layer"""
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half_planes = int(planes * ratio)
        self.BN = nn.BatchNorm2d(planes - self.half_planes)
        self.IN = nn.InstanceNorm2d(self.half_planes, affine=True)

    def forward(self, x):
        split = torch.split(x, self.half_planes, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck with IBN support"""
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_IBN(nn.Module):
    """ResNet with IBN layers"""
    def __init__(self, block, layers, ibn_cfg=('a', 'a', 'a', None)):
        self.inplanes = 64
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], ibn=ibn_cfg[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], ibn=ibn_cfg[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], ibn=ibn_cfg[3], stride=2)

    def _make_layer(self, block, planes, blocks, ibn=None, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class BNNeck(nn.Module):
    """Batch Normalization Neck"""
    def __init__(self, in_dim, class_num=776):
        super(BNNeck, self).__init__()
        self.in_dim = in_dim
        self.class_num = class_num
        
        # BN neck
        self.bottleneck = nn.BatchNorm1d(in_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        
        # Classifier
        self.classifier = nn.Linear(in_dim, class_num, bias=False)
        
        # Initialize
        self.bottleneck.apply(self._init_bn)
        self.classifier.apply(self._init_classifier)
    
    def _init_bn(self, m):
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def _init_classifier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
    
    def forward(self, x):
        # Global features (for triplet loss)
        global_feat = x
        
        # BN features (for ID loss)
        bn_feat = self.bottleneck(global_feat)
        
        # Classification logits
        if self.training:
            cls_score = self.classifier(bn_feat)
            return global_feat, cls_score
        else:
            return bn_feat


class BoTBaseline(nn.Module):
    """
    BoT-Baseline for Vehicle Re-ID
    ResNet50-IBN + BNNeck + ID Loss + Triplet Loss
    """
    def __init__(self, num_classes=576, pretrain=True, grid_size=None):
        super(BoTBaseline, self).__init__()
        
        # Backbone: ResNet50-IBN-a
        self.backbone = ResNet_IBN(Bottleneck, [3, 4, 6, 3], 
                                   ibn_cfg=('a', 'a', 'a', None))
        
        # Load ImageNet pretrained weights if needed
        if pretrain:
            self._load_pretrained_weights()
        
        # Feature dimension
        self.in_planes = 2048
        
        # Grid Graph Generator (Optional for Phase 2 Grid Pooling Experiment)
        self.grid_size = grid_size
        if grid_size is not None:
             self.graph_generator = GridGraphGenerator(self.in_planes, grid_size=grid_size)
             print(f"✅ Enabled Grid Pooling with size: {grid_size}")
        else:
             self.graph_generator = None

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # BNNeck
        self.neck = BNNeck(self.in_planes, num_classes)
        
    def _load_pretrained_weights(self):
        """Load ImageNet pretrained weights"""
        try:
            import torchvision.models as models
            from torchvision.models import ResNet50_Weights
            
            print("Loading ImageNet pretrained weights...")
            # Load with progress bar
            pretrained_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=True)
            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.backbone.state_dict()
            
            # Filter out unnecessary keys and keys with mismatched shapes
            filtered_dict = {}
            matched_keys = 0
            for k, v in pretrained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                    matched_keys += 1
                    
            model_dict.update(filtered_dict)
            self.backbone.load_state_dict(model_dict, strict=False)
            print(f"✓ Loaded ImageNet pretrained weights ({matched_keys} layers matched)")
        except KeyboardInterrupt:
            print("\n⚠ Download interrupted - training will continue without pretrained weights")
            raise
        except Exception as e:
            print(f"⚠ Warning: Could not load pretrained weights: {e}")
            print("  Training will continue with random initialization")
        
    def forward(self, x, return_featmap=False):
        # Feature extraction
        feat_map = self.backbone(x)  # [B, 2048, H, W]
        
        if self.graph_generator is not None:
            # Grid Pooling Path (Phase 2 Experiment)
            # 1. Generate nodes [B, N, C]
            nodes = self.graph_generator(feat_map)
            # 2. Mean Pooling back to [B, C] for Baseline Compatibility
            global_feat = nodes.mean(dim=1)
        else:
            # Standard GAP Path
            # Global pooling
            global_feat = self.gap(feat_map)  # [B, 2048, 1, 1]
            global_feat = global_feat.view(global_feat.shape[0], -1)  # [B, 2048]
        
        # BNNeck forward
        if self.training:
            global_feat, cls_score = self.neck(global_feat)
            if return_featmap:
                return global_feat, cls_score, feat_map
            return global_feat, cls_score
        else:
            bn_feat = self.neck(global_feat)
            if return_featmap:
                return bn_feat, feat_map
            return bn_feat


def build_bot_baseline(num_classes=576, pretrain=True, grid_size=None):
    """Build BoT-Baseline model"""
    return BoTBaseline(num_classes=num_classes, pretrain=pretrain, grid_size=grid_size)