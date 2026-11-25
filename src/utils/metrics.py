import torch
import numpy as np
from sklearn.metrics import average_precision_score


def calculate_reid_metrics(
    q_feat,
    q_pids,
    q_camids,
    g_feat,
    g_pids,
    g_camids,
    q_scenes=None,
    g_scenes=None,
    ranks=(1, 5, 10),
    return_per_query=False,
):
    """
    Calculate mAP and CMC Rank-k for cross-camera Re-ID.
    By default (q_scenes=None), excludes gallery items sharing BOTH same PID and same cam_id.
    """
    distmat = torch.cdist(q_feat, g_feat).cpu().numpy()
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    if q_scenes is not None and g_scenes is not None:
        q_scenes = np.asarray(q_scenes)
        g_scenes = np.asarray(g_scenes)

    num_q, _ = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_ap = []
    all_cmc = []
    valid_query_indices = []
    max_rank = max(ranks)

    for i in range(num_q):
        filter_mask = (g_pids[indices[i]] == q_pids[i]) & (g_camids[indices[i]] == q_camids[i])
        if q_scenes is not None and g_scenes is not None:
            filter_mask = filter_mask & (g_scenes[indices[i]] == q_scenes[i])

        keep = ~filter_mask
        y_true = matches[i][keep]
        y_score = -distmat[i][indices[i]][keep]

        if not np.any(y_true):
            continue

        ap = average_precision_score(y_true, y_score)
        all_ap.append(float(ap))
        valid_query_indices.append(i)

        cmc = y_true.cumsum()
        cmc[cmc > 1] = 1
        if len(cmc) < max_rank:
            cmc = np.pad(cmc, (0, max_rank - len(cmc)), mode="edge")
        all_cmc.append(cmc[:max_rank])

    if not all_ap:
        out = {
            "mAP": 0.0,
            **{f"rank{r}": 0.0 for r in ranks},
            "num_valid_queries": 0,
        }
        if return_per_query:
            out["per_query_ap"] = []
            out["valid_query_indices"] = []
        return out

    cmc_mean = np.mean(all_cmc, axis=0)
    out = {
        "mAP": float(np.mean(all_ap)),
        **{f"rank{r}": float(cmc_mean[r - 1]) for r in ranks},
        "num_valid_queries": len(all_ap),
    }
    if return_per_query:
        out["per_query_ap"] = all_ap
        out["valid_query_indices"] = valid_query_indices
    return out


def calculate_map(
    q_feat,
    q_pids,
    q_camids,
    g_feat,
    g_pids,
    g_camids,
    q_scenes=None,
    g_scenes=None,
):
    return calculate_reid_metrics(
        q_feat,
        q_pids,
        q_camids,
        g_feat,
        g_pids,
        g_camids,
        q_scenes=q_scenes,
        g_scenes=g_scenes,
    )["mAP"]


def calculate_inter_intra_ratio(features, labels):
    """
    Calculate the ratio of average inter-class distance to average intra-class distance.
    Higher is better for discrimination.
    """
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    unique_labels = torch.unique(labels)
    centroids = []
    intra_distances = []

    for label in unique_labels:
        mask = labels == label
        class_features = features[mask]
        centroid = class_features.mean(dim=0)
        centroids.append(centroid)

        if class_features.size(0) > 1:
            dists = torch.cdist(class_features, centroid.unsqueeze(0))
            intra_distances.append(dists.mean().item())

    if not centroids:
        return 0.0, 0.0, 0.0

    centroids = torch.stack(centroids)

    if centroids.size(0) > 1:
        inter_distmat = torch.cdist(centroids, centroids)
        inter_dist = inter_distmat[torch.triu(torch.ones_like(inter_distmat), diagonal=1) == 1].mean().item()
    else:
        inter_dist = 0.0

    avg_intra_dist = float(np.mean(intra_distances)) if intra_distances else 0.0
    ratio = inter_dist / avg_intra_dist if avg_intra_dist > 0 else 0.0

    return ratio, inter_dist, avg_intra_dist
