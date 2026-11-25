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
    parser.add_argument("--query-id", type=int, default=None)
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

    indices = list(range(len(dataset)))
    if len(indices) > args.gallery_subset:
        np.random.seed(42)
        indices = list(np.random.choice(indices, args.gallery_subset, replace=False))

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

    all_feats = extract_features_selective(model, temporal_pool, dataset, indices, device)
    all_pids = torch.tensor([dataset.df.iloc[i]["person_id"] for i in indices])

    if args.query_id is not None:
        q_local_indices = np.where(all_pids.numpy() == args.query_id)[0]
        q_local_idx = q_local_indices[0] if len(q_local_indices) > 0 else np.random.randint(len(all_feats))
    else:
        q_local_idx = np.random.randint(len(all_feats))

    q_abs_idx = indices[q_local_idx]
    q_feat = all_feats[q_local_idx].unsqueeze(0)
    q_pid = all_pids[q_local_idx].item()
    q_seq_id = dataset.df.iloc[q_abs_idx]["sequence_id"]

    dist = torch.mm(q_feat, all_feats.t()).squeeze(0)
    sorted_indices = torch.argsort(dist, descending=True)

    top_indices = [idx.item() for idx in sorted_indices if idx.item() != q_local_idx][: args.top_k]

    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, args.top_k + 1, 1)
    ax.imshow(get_representative_frame(dataset, q_abs_idx))
    ax.set_title(f"QUERY\nPID: {q_pid}\n{q_seq_id}")
    ax.axis("off")

    for i, match_local_idx in enumerate(top_indices):
        ax = fig.add_subplot(1, args.top_k + 1, i + 2)
        match_abs_idx = indices[match_local_idx]
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

    plt.suptitle(f"Visual Re-ID Search - Model: {cfg['experiment']['name']}")
    plt.tight_layout()

    output_root = cfg["experiment"].get("output_root", "outputs/oe2_sgreid")
    out_dir = os.path.join(output_root, cfg["experiment"]["name"], "visual_search")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"random_search_pid{q_pid}.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Search results saved to {out_file}")


if __name__ == "__main__":
    main()
