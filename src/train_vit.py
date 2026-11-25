import argparse
import torch

from model.vit_embedder import ViTEmbedder
from preprocess.dataset import SequenceDataset
from train_base import load_config, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/vit_config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.exp_name is not None:
        cfg["experiment"]["name"] = args.exp_name
    if args.output_root is not None:
        cfg["experiment"]["output_root"] = args.output_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SequenceDataset(cfg["data"]["index_csv"], cfg["data"]["root"], split="train")

    model = ViTEmbedder(
        backbone_name=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        embed_dim=cfg["model"]["embed_dim"],
        num_classes=dataset.num_classes,
        dropout=cfg["model"]["dropout"],
        img_size=tuple(cfg["data"]["image_size"]),
    )

    train_model(cfg, model, device)


if __name__ == "__main__":
    main()
