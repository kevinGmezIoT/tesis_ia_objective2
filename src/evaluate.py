import argparse
import json
import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--arch", type=str, choices=["vit", "cnn"], required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--allow-unsplit", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_check = pd.read_csv(cfg["data"]["index_csv"], nrows=5)
    if "split" not in df_check.columns and not args.allow_unsplit:
        raise ValueError(f"'split' column missing in {cfg['data']['index_csv']}. Pass --allow-unsplit to force evaluation.")

    test_transforms = get_transforms(tuple(cfg["data"]["image_size"]), is_train=False)
    test_dataset = MyDataset(
        index_csv=cfg["data"]["index_csv"],
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

    all_features = []
    all_pids = []
    all_camids = []

    with torch.no_grad():
        for frames, _, metadata in tqdm(test_loader, desc="Extracting features"):
            frames = frames.to(device)
            B, T, C, H, W = frames.shape
            feat_flat = model(frames.view(B * T, C, H, W), return_logits=False)
            embedding = temporal_pool(feat_flat.view(B, T, -1))
            all_features.append(embedding.cpu())
            all_pids.extend([int(p) for p in metadata["person_id"]])
            all_camids.extend(parse_cam_ids(metadata["cam_id"]))

    all_features = torch.cat(all_features, dim=0)
    reid_metrics = calculate_reid_metrics(
        all_features,
        all_pids,
        all_camids,
        all_features,
        all_pids,
        all_camids,
        q_scenes=None,
        g_scenes=None,
        ranks=(1, 5, 10),
    )

    unique_pids = sorted(set(all_pids))
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    pids_tensor = torch.tensor([pid_to_idx[p] for p in all_pids], dtype=torch.long)
    ratio, inter, intra = calculate_inter_intra_ratio(all_features, pids_tensor)

    results = {
        "mAP": reid_metrics["mAP"],
        "rank1": reid_metrics["rank1"],
        "rank5": reid_metrics["rank5"],
        "rank10": reid_metrics["rank10"],
        "num_valid_queries": reid_metrics["num_valid_queries"],
        "ratio_inter_intra": ratio,
        "avg_inter_dist": inter,
        "avg_intra_dist": intra,
        "actual_embed_dim": int(all_features.shape[1]),
    }

    print(f"Results for {args.arch}:")
    print(json.dumps(results, indent=4))

    output_root = cfg["experiment"].get("output_root", "outputs/oe2_sgreid")
    output_dir = os.path.join(output_root, cfg["experiment"]["name"])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
