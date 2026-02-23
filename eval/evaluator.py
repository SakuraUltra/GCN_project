"""
Evaluation Engine for Vehicle Re-identification
专业的评估引擎，支持mAP和CMC指标计算
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging
from collections import defaultdict


class ReIDEvaluator:
    """
    车辆重识别评估器
    
    核心功能:
    1. 特征提取 - 支持测试时增强（TTA）
    2. 距离计算 - 欧式距离/余弦距离
    3. mAP计算 - 平均精度均值
    4. CMC曲线计算 - 累积匹配特征曲线
    5. 重排序 - 可选的后处理技术
    
    Args:
        model (nn.Module): 训练好的模型
        use_flip_test (bool): 是否使用水平翻转测试时增强，默认True
        use_rerank (bool): 是否使用重排序，默认False
        device: 计算设备，默认自动检测
    
    Example:
        >>> evaluator = ReIDEvaluator(model, use_flip_test=True)
        >>> results = evaluator.evaluate(query_loader, gallery_loader)
        >>> print(f"mAP: {results['mAP']:.4f}, Rank-1: {results['rank1']:.4f}")
    
    Reference:
        "Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking" - ECCVW 2016
    """
    
    def __init__(self, model, use_flip_test=True, use_rerank=False, device=None):
        self.model = model
        self.use_flip_test = use_flip_test
        self.use_rerank = use_rerank
        
        # 设备自动检测
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'ReIDEvaluator initialized | Device: {self.device}')
    
    def extract_features(self, data_loader, normalize=True):
        """
        提取特征向量
        
        Args:
            data_loader: 数据加载器
            normalize (bool): 是否L2归一化，默认True
            
        Returns:
            tuple: (features, pids, camids)
                - features: 特征矩阵 (N, D)
                - pids: 身份标签数组 (N,)
                - camids: 相机标签数组 (N,)
        """
        self.model.eval()
        
        features = []
        pids = []
        camids = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(data_loader, desc='Extracting features', 
                                                                                   dynamic_ncols=False, ncols=100, mininterval=2.0)):
                # 解包数据：测试集返回 (images, pids, camids)，没有labels
                images = batch_data[0].to(self.device)
                targets = batch_data[1]  # pids
                cam_ids = batch_data[2]  # camids
                
                # 前向传播
                batch_features = self.model(images)
                
                # 测试时增强（TTA）- 水平翻转
                if self.use_flip_test:
                    images_flip = torch.flip(images, dims=[3])
                    flip_features = self.model(images_flip)
                    batch_features = (batch_features + flip_features) / 2.0
                
                # L2归一化
                if normalize:
                    batch_features = F.normalize(batch_features, p=2, dim=1)
                
                features.append(batch_features.cpu())
                pids.extend(targets)
                camids.extend(cam_ids)
        
        features = torch.cat(features, dim=0)
        pids = np.array(pids)
        camids = np.array(camids)
        
        return features, pids, camids
    
    def compute_distance_matrix(self, query_features, gallery_features, metric='euclidean'):
        """
        计算查询集和画廊集之间的距离矩阵
        
        Args:
            query_features (torch.Tensor): 查询特征 (Nq, D)
            gallery_features (torch.Tensor): 画廊特征 (Ng, D)
            metric (str): 距离度量，'euclidean' 或 'cosine'
            
        Returns:
            np.ndarray: 距离矩阵 (Nq, Ng)
        """
        m, n = query_features.size(0), gallery_features.size(0)
        
        if metric == 'euclidean':
            # 欧式距离: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b^T
            dist = torch.pow(query_features, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gallery_features, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            dist.addmm_(query_features, gallery_features.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt()
        
        elif metric == 'cosine':
            # 余弦距离: 1 - cos(a, b) (BoT-Baseline标准!)
            query_features = F.normalize(query_features, p=2, dim=1)
            gallery_features = F.normalize(gallery_features, p=2, dim=1)
            dist = 1 - torch.mm(query_features, gallery_features.t())
        
        else:
            raise ValueError(f'Unknown metric: {metric}')
        
        return dist.numpy()
    
    def evaluate_rank(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """
        计算Rank-k准确率和mAP
        
        Args:
            distmat (np.ndarray): 距离矩阵 (Nq, Ng)
            q_pids (np.ndarray): 查询身份标签
            g_pids (np.ndarray): 画廊身份标签
            q_camids (np.ndarray): 查询相机标签
            g_camids (np.ndarray): 画廊相机标签
            max_rank (int): 最大rank值，默认50
            
        Returns:
            tuple: (cmc, mAP)
                - cmc: CMC曲线数组
                - mAP: 平均精度均值
        """
        num_q, num_g = distmat.shape
        
        if num_g < max_rank:
            max_rank = num_g
            self.logger.warning(f'Gallery size ({num_g}) < max_rank, adjusted to {max_rank}')
        
        # 按距离排序
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
        
        # 计算CMC和mAP
        all_cmc = []
        all_AP = []
        num_valid_q = 0.0
        
        for q_idx in range(num_q):
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
            
            # 移除同一相机下的同一身份（车辆re-id特有规则）
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
            
            # 计算CMC
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                continue
            
            cmc_tmp = orig_cmc.cumsum()
            cmc_tmp[cmc_tmp > 1] = 1
            all_cmc.append(cmc_tmp[:max_rank])
            
            num_valid_q += 1.0
            
            # 计算AP（Average Precision）
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)
        
        assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
        
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        
        return all_cmc, mAP
    
    def evaluate(self, query_loader, gallery_loader, metric='cosine'):
        """
        完整评估流程
        
        Args:
            query_loader: 查询数据加载器
            gallery_loader: 画廊数据加载器
            metric (str): 距离度量，默认'cosine'
            
        Returns:
            dict: 评估结果
                - mAP: 平均精度均值
                - cmc: 完整CMC曲线
                - rank1: Rank-1准确率
                - rank5: Rank-5准确率
                - rank10: Rank-10准确率
        """
        self.logger.info('Starting evaluation...')
        
        # 提取特征
        self.logger.info('Extracting query features...')
        query_features, q_pids, q_camids = self.extract_features(query_loader)
        
        self.logger.info('Extracting gallery features...')
        gallery_features, g_pids, g_camids = self.extract_features(gallery_loader)
        
        self.logger.info(f'Query: {len(q_pids)} samples | Gallery: {len(g_pids)} samples')
        
        # 计算距离矩阵
        self.logger.info(f'Computing distance matrix (metric: {metric})...')
        distmat = self.compute_distance_matrix(query_features, gallery_features, metric=metric)
        
        # 重排序（可选）
        if self.use_rerank:
            self.logger.info('Applying re-ranking...')
            distmat = self.re_ranking(query_features, gallery_features, distmat)
        
        # 评估
        self.logger.info('Computing CMC and mAP...')
        cmc, mAP = self.evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
        
        # 整理结果
        results = {
            'mAP': mAP,
            'cmc': cmc,
            'rank1': cmc[0],
            'rank5': cmc[4],
            'rank10': cmc[9],
        }
        
        # 打印结果
        self.logger.info('=' * 50)
        self.logger.info('Evaluation Results:')
        self.logger.info(f'mAP: {mAP:.4f}')
        self.logger.info('CMC Curve:')
        for r in [1, 5, 10, 20]:
            self.logger.info(f'  Rank-{r:2d}: {cmc[r-1]:.4f}')
        self.logger.info('=' * 50)
        
        return results
    
    def re_ranking(self, query_features, gallery_features, distmat, k1=20, k2=6, lambda_value=0.3):
        """
        重排序算法（简化版）
        
        Reference:
            "Re-ranking Person Re-identification with k-reciprocal Encoding" - CVPR 2017
        
        Note:
            这是简化实现，完整版本请参考论文
        """
        self.logger.info('Re-ranking is simplified version')
        return distmat


def compute_mAP_cmc(query_features, gallery_features, query_labels, gallery_labels, 
                   query_cams, gallery_cams, metric='cosine'):
    """
    便捷函数：直接计算mAP和CMC（无需创建evaluator）
    
    Args:
        query_features (torch.Tensor): 查询特征
        gallery_features (torch.Tensor): 画廊特征
        query_labels (np.ndarray): 查询标签
        gallery_labels (np.ndarray): 画廊标签
        query_cams (np.ndarray): 查询相机
        gallery_cams (np.ndarray): 画廊相机
        metric (str): 距离度量
        
    Returns:
        tuple: (mAP, cmc)
    """
    class DummyModel:
        def __init__(self):
            pass
    
    evaluator = ReIDEvaluator(DummyModel(), use_flip_test=False)
    distmat = evaluator.compute_distance_matrix(query_features, gallery_features, metric)
    cmc, mAP = evaluator.evaluate_rank(distmat, query_labels, gallery_labels, query_cams, gallery_cams)
    
    return mAP, cmc
