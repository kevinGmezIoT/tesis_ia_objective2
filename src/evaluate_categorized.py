import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from model.cnn_embedder import CNNEmbedder
from model.temporal_pooling import TemporalPooling
from model.vit_embedder import ViTEmbedder
from preprocess.my_dataset import MyDataset
from preprocess.transforms import get_transforms
from utils.metrics import calculate_inter_intra_ratio, calculate_reid_metrics


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_cam_ids(cam_list):
    return [
        int(str(c).replace("c", "")) if isinstance(c, str) and "c" in c else int(c)
        for c in cam_list
    ]


def infer_ckpt_dims(state_dict, cfg):
    num_classes = state_dict["classifier.weight"].shape[0] if "classifier.weight" in state_dict else 0
    if "embedding_layer.weight" in state_dict:
        embed_dim = int(state_dict["embedding_layer.weight"].shape[0])
    else:
        embed_dim = int(cfg["model"].get("embed_dim", 512))
    return num_classes, embed_dim


def measure_latency_ms(model, temporal_pool, test_dataset, device, n_warmup=10, n_bench=50):
    if len(test_dataset) == 0:
        return 0.0
    sample_frames, _, _ = test_dataset[0]
    sample_frames = sample_frames.unsqueeze(0).to(device)
    B, T, C, H, W = sample_frames.shape

    with torch.no_grad():
        for _ in range(n_warmup):
            feat_flat = model(sample_frames.view(B * T, C, H, W), return_logits=False)
            _ = temporal_pool(feat_flat.view(B, T, -1))
        if device.type == "cuda":
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        for _ in range(n_bench):
            feat_flat = model(sample_frames.view(B * T, C, H, W), return_logits=False)
            _ = temporal_pool(feat_flat.view(B, T, -1))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

    return ((t_end - t_start) / n_bench) * 1000.0


def compute_subset_metrics(q_feat, q_pids, q_camids, g_feat, g_pids, g_camids, return_per_query=False):
    reid = calculate_reid_metrics(
        q_feat,
        q_pids,
        q_camids,
        g_feat,
        g_pids,
        g_camids,
        q_scenes=None,
        g_scenes=None,
        ranks=(1, 5, 10),
        return_per_query=return_per_query,
    )
    unique_q_pids = sorted(set(q_pids))
    pid_to_idx = {pid: i for i, pid in enumerate(unique_q_pids)}
    q_pids_tensor = torch.tensor([pid_to_idx[p] for p in q_pids], dtype=torch.long)
    ratio, inter, intra = calculate_inter_intra_ratio(q_feat, q_pids_tensor)

    result = {
        "mAP": reid["mAP"],
        "rank1": reid["rank1"],
        "rank5": reid["rank5"],
        "rank10": reid["rank10"],
        "num_valid_queries": reid["num_valid_queries"],
        "ratio_inter_intra": ratio,
        "avg_inter_dist": inter,
        "avg_intra_dist": intra,
        "count": len(q_pids),
    }
    if return_per_query:
        return result, reid["per_query_ap"], reid["valid_query_indices"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--arch", type=str, choices=["vit", "cnn"], required=True)
    parser.add_argument("--input-csv", type=str, default=None, help="Override CSV path")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--allow-unsplit", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--ambiguous-pairs-csv",
        type=str,
        default="inputs/confirmed_ambiguous_identity_pairs.csv",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    index_csv = args.input_csv or cfg["data"]["index_csv"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_check = pd.read_csv(index_csv, nrows=5)
    if "split" not in df_check.columns and not args.allow_unsplit:
        raise ValueError(f"'split' column missing in {index_csv}. Pass --allow-unsplit to force evaluation.")

    test_transforms = get_transforms(tuple(cfg["data"]["image_size"]), is_train=False)
    test_dataset = MyDataset(
        index_csv=index_csv,
        root_dir=cfg["data"]["root"],
        transform=test_transforms,
        sequence_len=cfg["data"]["sequence_len"],
        split=args.split if "split" in df_check.columns else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    num_classes, embed_dim = infer_ckpt_dims(state_dict, cfg)
    print(f"Inferred num_classes={num_classes}, embed_dim={embed_dim} from checkpoint.")

    if args.arch == "vit":
        model = ViTEmbedder(
            backbone_name=cfg["model"]["backbone"],
            pretrained=False,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=cfg["model"].get("dropout", 0.1),
            img_size=tuple(cfg["data"]["image_size"]),
        )
    else:
        model = CNNEmbedder(
            backbone_name=cfg["model"]["backbone"],
            pretrained=False,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=cfg["model"].get("dropout", 0.1),
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    temporal_pool = TemporalPooling().to(device)
    temporal_pool.eval()

    num_params = sum(p.numel() for p in model.parameters())
    avg_latency_ms = measure_latency_ms(model, temporal_pool, test_dataset, device)

    all_features = []
    all_pids = []
    all_camids = []
    all_seq_ids = []
    all_illum = []
    all_occl = []

    print(f"Extracting features for {len(test_dataset)} sequences (split={args.split})...")
    with torch.no_grad():
        for frames, _, metadata in tqdm(test_loader):
            frames = frames.to(device)
            B, T, C, H, W = frames.shape
            feat_flat = model(frames.view(B * T, C, H, W), return_logits=False)
            embedding = temporal_pool(feat_flat.view(B, T, -1))
            if cfg.get("eval", {}).get("normalize_embeddings", True):
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            all_features.append(embedding.cpu())
            all_pids.extend([int(p) for p in metadata["person_id"]])
            all_camids.extend(metadata["cam_id"])
            all_seq_ids.extend(metadata["sequence_id"])
            all_illum.extend(metadata["illum_label"])
            all_occl.extend(metadata["occlusion_label"])

    all_features = torch.cat(all_features, dim=0)
    actual_embed_dim = int(all_features.shape[1])
    camids = parse_cam_ids(all_camids)
    pids = all_pids

    overall_metrics, per_query_ap, valid_q_indices = compute_subset_metrics(
        all_features, pids, camids, all_features, pids, camids, return_per_query=True
    )

    def get_categorized_metrics(labels, category_name):
        unique_cats = sorted(set(labels))
        cat_results = {"OVERALL": overall_metrics}
        for cat in unique_cats:
            mask = np.array([l == cat for l in labels])
            if not np.any(mask):
                continue
            q_feat = all_features[mask]
            q_pids = [pids[i] for i, m in enumerate(mask) if m]
            q_camids = [camids[i] for i, m in enumerate(mask) if m]
            print(f"Calculating metrics for {category_name}: {cat} ({len(q_pids)} sequences)...")
            cat_results[cat] = compute_subset_metrics(q_feat, q_pids, q_camids, all_features, pids, camids)
        return cat_results

    print("\n--- Evaluation by Illumination ---")
    illum_metrics = get_categorized_metrics(all_illum, "Illumination")

    print("\n--- Evaluation by Occlusion ---")
    occl_metrics = get_categorized_metrics(all_occl, "Occlusion")

    unique_a = sorted(set(all_illum))
    unique_b = sorted(set(all_occl))
    cross_metrics = {}
    print("\n--- Cross Evaluation: Illumination x Occlusion ---")
    for val_a in unique_a:
        cross_metrics[val_a] = {}
        for val_b in unique_b:
            mask = np.array([(a == val_a and b == val_b) for a, b in zip(all_illum, all_occl)])
            if not np.any(mask):
                cross_metrics[val_a][val_b] = "N/A"
                continue
            q_feat = all_features[mask]
            q_pids = [pids[i] for i, m in enumerate(mask) if m]
            q_camids = [camids[i] for i, m in enumerate(mask) if m]
            cross_metrics[val_a][val_b] = compute_subset_metrics(q_feat, q_pids, q_camids, all_features, pids, camids)

    ambiguous_eval = {}
    if args.ambiguous_pairs_csv and os.path.exists(args.ambiguous_pairs_csv):
        amb_df = pd.read_csv(args.ambiguous_pairs_csv)
        test_pid_set = set(pids)
        both_pairs = amb_df[
            amb_df["person_global_id_x"].isin(test_pid_set)
            & amb_df["person_global_id_y"].isin(test_pid_set)
        ]
        any_pairs = amb_df[
            amb_df["person_global_id_x"].isin(test_pid_set)
            | amb_df["person_global_id_y"].isin(test_pid_set)
        ]

        amb_pids_both = sorted(
            set(both_pairs["person_global_id_x"].tolist() + both_pairs["person_global_id_y"].tolist())
        )
        amb_pids_any = sorted(
            (set(any_pairs["person_global_id_x"].tolist() + any_pairs["person_global_id_y"].tolist())) & test_pid_set
        )

        ambiguous_eval["num_pairs_both_in_test"] = int(len(both_pairs))
        ambiguous_eval["pids_both_in_test"] = amb_pids_both
        ambiguous_eval["num_pairs_any_in_test"] = int(len(any_pairs))
        ambiguous_eval["pids_any_in_test"] = amb_pids_any

        if len(amb_pids_both) >= 2:
            mask_both = np.array([p in amb_pids_both for p in pids])
            q_feat = all_features[mask_both]
            q_pids = [pids[i] for i, m in enumerate(mask_both) if m]
            q_camids = [camids[i] for i, m in enumerate(mask_both) if m]
            ambiguous_eval["subset_both_in_test_vs_full_gallery"] = compute_subset_metrics(
                q_feat, q_pids, q_camids, all_features, pids, camids
            )
            ambiguous_eval["subset_both_in_test_only"] = compute_subset_metrics(
                q_feat, q_pids, q_camids, q_feat, q_pids, q_camids
            )

        if len(amb_pids_any) >= 2:
            mask_any = np.array([p in amb_pids_any for p in pids])
            q_feat = all_features[mask_any]
            q_pids = [pids[i] for i, m in enumerate(mask_any) if m]
            q_camids = [camids[i] for i, m in enumerate(mask_any) if m]
            ambiguous_eval["subset_any_in_test_vs_full_gallery"] = compute_subset_metrics(
                q_feat, q_pids, q_camids, all_features, pids, camids
            )

    final_results = {
        "experiment_name": cfg["experiment"]["name"],
        "arch": args.arch,
        "backbone": cfg["model"]["backbone"],
        "split": args.split,
        "embed_dim": actual_embed_dim,
        "num_params": int(num_params),
        "avg_latency_ms_per_seq": round(float(avg_latency_ms), 4),
        "illumination": illum_metrics,
        "occlusion": occl_metrics,
        "cross_illum_occl": cross_metrics,
        "ambiguous_pairs": ambiguous_eval,
    }

    print("\nFinal Results Summary (OVERALL):")
    print(json.dumps({k: v for k, v in overall_metrics.items()}, indent=4))

    output_root = cfg["experiment"].get("output_root", "outputs/oe2_sgreid")
    output_dir = args.output_dir or os.path.join(output_root, cfg["experiment"]["name"])
    os.makedirs(output_dir, exist_ok=True)

    out_file = os.path.join(output_dir, f"categorized_eval_{args.arch}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    valid_seq_ids = np.array([all_seq_ids[i] for i in valid_q_indices])
    npz_file = os.path.join(output_dir, f"per_query_ap_{args.arch}.npz")
    np.savez(npz_file, sequence_id=valid_seq_ids, ap=np.asarray(per_query_ap, dtype=np.float64))

    print(f"\nResults saved to {out_file} and {npz_file}")


if __name__ == "__main__":
    main()
