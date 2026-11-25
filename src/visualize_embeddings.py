import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm

from model.cnn_embedder import CNNEmbedder
from model.temporal_pooling import TemporalPooling
from model.vit_embedder import ViTEmbedder
from preprocess.dataset import SequenceDataset
from preprocess.transforms import get_transforms
from train_base import load_config


def extract_features_selective(model, temporal_pool, dataset, indices, device):
    model.eval()
    temporal_pool.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for idx in tqdm(indices, desc="Extracting features"):
            frames, label = dataset[idx]
            frames = frames.unsqueeze(0).to(device)
            B, T, C, H, W = frames.shape
            features_flat = model(frames.view(B * T, C, H, W), return_logits=False)
            features = temporal_pool(features_flat.view(B, T, -1))
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            all_features.append(features.cpu().numpy())
            all_labels.append(label)

    return np.concatenate(all_features), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=1000)
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
        split=args.split,
    )

    indices = list(range(len(dataset)))
    if len(indices) > args.max_samples:
        np.random.seed(42)
        indices = list(np.random.choice(indices, args.max_samples, replace=False))

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

    features, labels = extract_features_selective(model, temporal_pool, dataset, indices, device)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, init="pca", random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        if i >= 20:
            break
        idx = labels == label
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=f"ID {dataset.pids[label]}", alpha=0.7)

    plt.title(f"t-SNE Visualization - {cfg['experiment']['name']} ({args.split})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
    plt.tight_layout()

    output_root = cfg["experiment"].get("output_root", "outputs/oe2_sgreid")
    output_path = os.path.join(output_root, cfg["experiment"]["name"], "tsne_visualization.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
