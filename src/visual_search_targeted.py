import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from model.cnn_embedder import CNNEmbedder
from model.temporal_pooling import TemporalPooling
from model.vit_embedder import ViTEmbedder
from preprocess.dataset import SequenceDataset
from preprocess.transforms import get_transforms
from train_base import load_config


def get_representative_frame(dataset, idx):
    frames_dir = dataset.df.iloc[idx]["frames_dir"]
    full_dir = os.path.join(dataset.root_dir, frames_dir)
    imgs = sorted(os.listdir(full_dir))
    if not imgs:
        return None
    with Image.open(os.path.join(full_dir, imgs[len(imgs) // 2])) as im:
        return im.convert("RGB")


def extract_features_selective(model, temporal_pool, dataset, indices, device):
    model.eval()
    temporal_pool.eval()
    features = []
    with torch.no_grad():
        for idx in tqdm(indices, desc="Extracting features"):
            frames, _ = dataset[idx]
            frames = frames.unsqueeze(0).to(device)
            B, T, C, H, W = frames.shape
            f_flat = model(frames.view(B * T, C, H, W), return_logits=False)
            f_pool = temporal_pool(f_flat.view(B, T, -1))
            f_norm = torch.nn.functional.normalize(f_pool, p=2, dim=1)
            features.append(f_norm.cpu())
    return torch.cat(features, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query-ids", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gallery-subset", type=int, default=1000)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = get_transforms(tuple(cfg["data"]["image_size"]), is_train=False)
    dataset = SequenceDataset(
        index_csv=cfg["data"]["index_csv"],
        root_dir=cfg["data"]["root"],
        transform=transforms,
        sequence_len=cfg["data"]["sequence_len"],
        split="test",
    )

    target_ids = [s.strip() for s in args.query_ids.split(",")]
    query_indices = []
    for tid in target_ids:
        idx_list = dataset.df.index[dataset.df["sequence_id"] == tid].tolist()
        if idx_list:
            query_indices.append(idx_list[0])
        else:
            print(f"Warning: {tid} not found in test split.")

    if not query_indices:
        print("Error: No valid query IDs found.")
        return

    gallery_indices = list(range(len(dataset)))
    if len(gallery_indices) > args.gallery_subset:
        other_indices = [i for i in gallery_indices if i not in query_indices]
        np.random.seed(42)
        gallery_indices = query_indices + list(np.random.choice(other_indices, args.gallery_subset, replace=False))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    num_classes = sd["classifier.weight"].shape[0] if "classifier.weight" in sd else 0
    embed_dim = int(sd["embedding_layer.weight"].shape[0]) if "embedding_layer.weight" in sd else int(cfg["model"]["embed_dim"])

    arch = cfg["model"].get("arch", "vit" if "vit" in cfg["experiment"]["name"].lower() else "cnn")
    if arch == "vit":
        model = ViTEmbedder(cfg["model"]["backbone"], pretrained=False, embed_dim=embed_dim, num_classes=num_classes, img_size=tuple(cfg["data"]["image_size"]))
    else:
        model = CNNEmbedder(cfg["model"]["backbone"], pretrained=False, embed_dim=embed_dim, num_classes=num_classes)

    model.load_state_dict(sd)
    model = model.to(device)
    temporal_pool = TemporalPooling().to(device)

    all_feats = extract_features_selective(model, temporal_pool, dataset, gallery_indices, device)
    all_pids = torch.tensor([dataset.df.iloc[i]["person_id"] for i in gallery_indices])
    gallery_idx_map = {abs_idx: local_idx for local_idx, abs_idx in enumerate(gallery_indices)}

    output_root = cfg["experiment"].get("output_root", "outputs/oe2_sgreid")
    output_dir = os.path.join(output_root, cfg["experiment"]["name"], "visual_search")
    os.makedirs(output_dir, exist_ok=True)

    for q_abs_idx in query_indices:
        q_local_idx = gallery_idx_map[q_abs_idx]
        q_feat = all_feats[q_local_idx].unsqueeze(0)
        q_pid = all_pids[q_local_idx].item()
        q_seq_id = dataset.df.iloc[q_abs_idx]["sequence_id"]

        dist = torch.mm(q_feat, all_feats.t()).squeeze(0)
        indices = torch.argsort(dist, descending=True)
        top_indices = [idx.item() for idx in indices if idx.item() != q_local_idx][: args.top_k]

        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, args.top_k + 1, 1)
        ax.imshow(get_representative_frame(dataset, q_abs_idx))
        ax.set_title(f"QUERY\n{q_seq_id}\nID: {q_pid}")
        ax.axis("off")

        for i, match_local_idx in enumerate(top_indices):
            ax = fig.add_subplot(1, args.top_k + 1, i + 2)
            match_abs_idx = gallery_indices[match_local_idx]
            match_pid = all_pids[match_local_idx].item()
            match_seq_id = dataset.df.iloc[match_abs_idx]["sequence_id"]
            color = "green" if match_pid == q_pid else "red"
            ax.imshow(get_representative_frame(dataset, match_abs_idx))
            ax.set_title(f"Rank {i+1}\nPID: {match_pid}\n{match_seq_id}\nSim: {dist[match_local_idx]:.3f}", color=color, fontsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"search_{q_seq_id}.png"))
        plt.close()
        print(f"Result saved for {q_seq_id}")


if __name__ == "__main__":
    main()
