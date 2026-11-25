import argparse
import glob
import os

import numpy as np
from PIL import Image
import torch
import yaml

from model.cnn_embedder import CNNEmbedder
from model.temporal_pooling import TemporalPooling
from model.vit_embedder import ViTEmbedder
from preprocess.transforms import get_transforms


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Inference Visual Feature Extractor")
    parser.add_argument("--config", type=str, default="src/configs/vit_config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing frames of a sequence")
    parser.add_argument("--output", type=str, default="embedding.npy", help="Output file for embedding")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
    embed_dim = (
        int(state_dict["embedding_layer.weight"].shape[0])
        if "embedding_layer.weight" in state_dict
        else int(cfg["model"].get("embed_dim", 512))
    )

    arch = cfg["model"].get("arch", "vit" if "vit" in cfg["model"]["backbone"].lower() else "cnn")
    if arch == "vit":
        model = ViTEmbedder(
            backbone_name=cfg["model"]["backbone"],
            pretrained=False,
            embed_dim=embed_dim,
            num_classes=0,
            dropout=0.0,
            img_size=tuple(cfg["data"]["image_size"]),
        ).to(device)
    else:
        model = CNNEmbedder(
            backbone_name=cfg["model"]["backbone"],
            pretrained=False,
            embed_dim=embed_dim,
            num_classes=0,
            dropout=0.0,
        ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    temporal_pool = TemporalPooling().to(device)
    temporal_pool.eval()

    transform = get_transforms(tuple(cfg["data"]["image_size"]), is_train=False)
    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    if not frame_paths:
        print(f"No frames found in {args.frames_dir}")
        return

    frames = []
    for p in frame_paths:
        with Image.open(p) as im:
            img = transform(im.convert("RGB"))
        frames.append(img)

    frames = torch.stack(frames).unsqueeze(0).to(device)

    with torch.no_grad():
        B, T, C, H, W = frames.shape
        features_flat = model(frames.view(B * T, C, H, W), return_logits=False)
        embedding = temporal_pool(features_flat.view(B, T, -1))

    emb_np = embedding.cpu().numpy().squeeze()
    np.save(args.output, emb_np)
    print(f"Embedding saved to {args.output} (shape={emb_np.shape})")


if __name__ == "__main__":
    main()
