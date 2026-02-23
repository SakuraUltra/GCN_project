"""
Evaluation metrics for Vehicle Re-Identification
Including mAP and CMC (Cumulative Matching Characteristics)
"""

import numpy as np
import torch


def compute_distance_matrix(query_features, gallery_features, metric='cosine'):
    """
    Compute distance matrix between query and gallery features
    
    Args:
        query_features: Query features [num_query, feature_dim]
        gallery_features: Gallery features [num_gallery, feature_dim]
        metric: Distance metric ('cosine' or 'euclidean')
    
    Returns:
        Distance matrix [num_query, num_gallery]
    """
    if isinstance(query_features, torch.Tensor):
        query_features = query_features.numpy()
    if isinstance(gallery_features, torch.Tensor):
        gallery_features = gallery_features.numpy()
    
    if metric == 'cosine':
        # L2 normalize features
        query_features = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-12)
        gallery_features = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-12)
        # Cosine distance = 1 - cosine similarity
        distmat = 1 - np.dot(query_features, gallery_features.T)
    elif metric == 'euclidean':
        # Euclidean distance
        m, n = query_features.shape[0], gallery_features.shape[0]
        distmat = np.power(query_features, 2).sum(axis=1, keepdims=True).repeat(n, axis=1) + \
                  np.power(gallery_features, 2).sum(axis=1).repeat(m, axis=0).reshape(m, n) - \
                  2 * np.dot(query_features, gallery_features.T)
        distmat = np.sqrt(np.maximum(distmat, 0))
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distmat


def evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """
    Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")
        
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def compute_mAP_cmc(query_features, gallery_features, query_pids, gallery_pids, 
                    query_camids, gallery_camids, metric='cosine'):
    """
    Compute mAP and CMC scores
    
    Returns:
        dict: Results containing mAP, rank-1, rank-5, rank-10 accuracies
    """
    # Compute distance matrix
    distmat = compute_distance_matrix(query_features, gallery_features, metric)
    
    # Compute CMC and mAP
    cmc, mAP = evaluate_rank(distmat, query_pids, gallery_pids, 
                            query_camids, gallery_camids, max_rank=50)
    
    results = {
        'mAP': mAP,
        'CMC-1': cmc[0],
        'CMC-5': cmc[4], 
        'CMC-10': cmc[9],
        'CMC-20': cmc[19]
    }
    
    return results