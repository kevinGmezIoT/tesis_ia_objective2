import hashlib
import json
import math
import os
import platform
import random
import time

import numpy as np
import timm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from model.losses import CrossEntropyLabelSmooth, TripletLoss
from model.temporal_pooling import TemporalPooling
from preprocess.dataset import SequenceDataset, RandomIdentitySampler
from preprocess.transforms import get_transforms
from utils.metrics import calculate_inter_intra_ratio, calculate_reid_metrics


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_cam_ids(cam_list):
    return [
        int(str(c).replace("c", "")) if isinstance(c, str) and "c" in c else int(c)
        for c in cam_list
    ]


def resolve_freeze_stage(epoch: int, pf_cfg: dict) -> str:
    if not pf_cfg or not pf_cfg.get("enabled", False):
        return "full"
    freeze_epochs = int(pf_cfg.get("freeze_epochs", 0))
    partial_epochs = int(pf_cfg.get("partial_unfreeze_epochs", freeze_epochs))
    if epoch < freeze_epochs:
        return "head_only"
    if epoch < partial_epochs:
        return "partial"
    return "full"


def evaluate_validation_reid(model, temporal_pool, val_loader, device):
    model.eval()
    temporal_pool.eval()

    all_features = []
    all_pids = []
    all_camids = []

    with torch.no_grad():
        for frames, _, metadata in tqdm(val_loader, desc="[Val Re-ID]", leave=False):
            frames = frames.to(device, non_blocking=True)
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B * T, C, H, W)
            feat_flat = model(frames_flat, return_logits=False)
            embedding = temporal_pool(feat_flat.view(B, T, -1))
            all_features.append(embedding.cpu())
            all_pids.extend([int(p) for p in metadata["person_id"]])
            all_camids.extend(parse_cam_ids(metadata["cam_id"]))

    if not all_features:
        return {"mAP": 0.0, "rank1": 0.0, "rank5": 0.0, "rank10": 0.0, "riic": 0.0, "num_valid_queries": 0}

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
    pid_tensor = torch.tensor([pid_to_idx[p] for p in all_pids], dtype=torch.long)
    riic, inter_dist, intra_dist = calculate_inter_intra_ratio(all_features, pid_tensor)

    return {
        "mAP": float(reid_metrics["mAP"]),
        "rank1": float(reid_metrics["rank1"]),
        "rank5": float(reid_metrics["rank5"]),
        "rank10": float(reid_metrics["rank10"]),
        "num_valid_queries": int(reid_metrics["num_valid_queries"]),
        "riic": float(riic),
        "avg_inter_dist": float(inter_dist),
        "avg_intra_dist": float(intra_dist),
    }


def train_model(cfg, model, device):
    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    print(f"Loading data (seed={seed})...")
    color_jitter = bool(cfg["data"].get("color_jitter", True))
    train_transforms = get_transforms(tuple(cfg["data"]["image_size"]), is_train=True, color_jitter=color_jitter)
    val_transforms = get_transforms(tuple(cfg["data"]["image_size"]), is_train=False)

    train_dataset = SequenceDataset(
        index_csv=cfg["data"]["index_csv"],
        root_dir=cfg["data"]["root"],
        transform=train_transforms,
        sequence_len=cfg["data"]["sequence_len"],
        split="train",
        is_train=True,
        return_metadata=False,
    )

    val_dataset = SequenceDataset(
        index_csv=cfg["data"]["index_csv"],
        root_dir=cfg["data"]["root"],
        transform=val_transforms,
        sequence_len=cfg["data"]["sequence_len"],
        split="val",
        is_train=False,
        return_metadata=True,
    )

    overlap_train_val = set(train_dataset.pids) & set(val_dataset.pids)
    if overlap_train_val:
        print(f"WARNING: Train and val share {len(overlap_train_val)} PIDs (not open-set).")
    else:
        print(f"Open-set verified: train ({len(train_dataset.pids)} PIDs) and val ({len(val_dataset.pids)} PIDs) are disjoint.")

    num_workers = int(cfg["data"].get("num_workers", 4))
    max_instances = cfg["data"].get("max_instances_per_pid", 32)
    batch_size = int(cfg["train"]["batch"]["P"]) * int(cfg["train"]["batch"]["K"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomIdentitySampler(
            train_dataset,
            batch_size,
            int(cfg["train"]["batch"]["K"]),
            max_instances_per_pid=max_instances,
        ),
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=max(32, batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = model.to(device)
    temporal_pool = TemporalPooling().to(device)

    criterion_ce = CrossEntropyLabelSmooth(
        num_classes=train_dataset.num_classes,
        epsilon=float(cfg["loss"]["ce"]["label_smoothing"]),
        use_gpu=(device.type == "cuda"),
    )
    criterion_tri = TripletLoss(margin=float(cfg["loss"]["triplet"]["margin"]))

    opt_cfg = cfg["train"]["optimizer"]
    base_lr = float(opt_cfg["lr"])
    backbone_lr_mult = float(opt_cfg.get("backbone_lr_mult", 0.2))
    weight_decay = float(opt_cfg.get("weight_decay", 0.05))

    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(base_lr=base_lr, backbone_lr_mult=backbone_lr_mult, include_frozen=True)
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr}]

    optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)

    total_epochs = int(cfg["train"]["epochs"])
    warmup_epochs = int(cfg["train"].get("scheduler", {}).get("warmup_epochs", 0))

    def lr_lambda(epoch_idx: int):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        progress = float(epoch_idx - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    output_root = cfg["experiment"].get("output_root", "outputs/oe2_sgreid")
    output_dir = os.path.join(output_root, cfg["experiment"]["name"])
    save_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    total_params = sum(p.numel() for p in model.parameters())
    run_meta = {
        "experiment": cfg["experiment"]["name"],
        "seed": seed,
        "index_csv": cfg["data"]["index_csv"],
        "index_csv_sha256": sha256_file(cfg["data"]["index_csv"]),
        "train_sequences": len(train_dataset),
        "train_pids": train_dataset.num_classes,
        "val_sequences": len(val_dataset),
        "val_pids": val_dataset.num_classes,
        "total_params": int(total_params),
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "python_version": platform.python_version(),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    pf_cfg = cfg["train"].get("progressive_finetune", {})
    val_freq = int(cfg["train"].get("val_freq", 1))
    ce_w = float(cfg["loss"]["ce"]["weight"])
    tri_w = float(cfg["loss"]["triplet"]["weight"])

    history = {
        "train_loss": [],
        "train_ce": [],
        "train_tri": [],
        "train_acc": [],
        "val_mAP": [],
        "val_rank1": [],
        "val_rank5": [],
        "val_riic": [],
        "lr": [],
        "freeze_stage": [],
    }

    best_val_map = -1.0
    best_epoch = -1
    t0 = time.time()

    print(f"Starting training for {cfg['experiment']['name']} -> {output_dir}")
    for epoch in range(total_epochs):
        stage = resolve_freeze_stage(epoch, pf_cfg)
        if hasattr(model, "set_freeze_stage"):
            model.set_freeze_stage(stage)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model.train()
        temporal_pool.train()

        train_loss = 0.0
        train_ce = 0.0
        train_tri = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [{stage}]")
        for step_idx, (frames, labels) in enumerate(pbar):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B * T, C, H, W)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    feat_flat, logits_flat = model(frames_flat, return_logits=True)
                    embedding = temporal_pool(feat_flat.view(B, T, -1))
                    logits = logits_flat.view(B, T, -1).mean(dim=1)
                    loss_ce = criterion_ce(logits, labels)
                    loss_tri = criterion_tri(embedding, labels)
                    loss = ce_w * loss_ce + tri_w * loss_tri
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                feat_flat, logits_flat = model(frames_flat, return_logits=True)
                embedding = temporal_pool(feat_flat.view(B, T, -1))
                logits = logits_flat.view(B, T, -1).mean(dim=1)
                loss_ce = criterion_ce(logits, labels)
                loss_tri = criterion_tri(embedding, labels)
                loss = ce_w * loss_ce + tri_w * loss_tri
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_ce += loss_ce.item()
            train_tri += loss_tri.item()
            _, preds = torch.max(logits, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{train_loss / (step_idx + 1):.3f}",
                "acc": f"{train_correct / max(1, train_total):.3f}",
            })

        current_head_lr = optimizer.param_groups[-1]["lr"]
        scheduler.step()

        n_batches = max(1, len(train_loader))
        avg_train_loss = train_loss / n_batches
        avg_train_ce = train_ce / n_batches
        avg_train_tri = train_tri / n_batches
        avg_train_acc = train_correct / max(1, train_total)

        run_val = ((epoch + 1) % val_freq == 0) or (epoch + 1 == total_epochs)
        if run_val:
            val_metrics = evaluate_validation_reid(model, temporal_pool, val_loader, device)
        else:
            val_metrics = {
                "mAP": history["val_mAP"][-1] if history["val_mAP"] else 0.0,
                "rank1": history["val_rank1"][-1] if history["val_rank1"] else 0.0,
                "rank5": history["val_rank5"][-1] if history["val_rank5"] else 0.0,
                "riic": history["val_riic"][-1] if history["val_riic"] else 0.0,
            }

        history["train_loss"].append(avg_train_loss)
        history["train_ce"].append(avg_train_ce)
        history["train_tri"].append(avg_train_tri)
        history["train_acc"].append(avg_train_acc)
        history["val_mAP"].append(val_metrics["mAP"])
        history["val_rank1"].append(val_metrics["rank1"])
        history["val_rank5"].append(val_metrics["rank5"])
        history["val_riic"].append(val_metrics["riic"])
        history["lr"].append(current_head_lr)
        history["freeze_stage"].append(stage)

        print(
            f"Epoch {epoch + 1}/{total_epochs} [{stage}, trainable={trainable_params:,}]: "
            f"Train Loss={avg_train_loss:.4f}, Acc={avg_train_acc:.4f} | "
            f"Val mAP={val_metrics['mAP']:.4f}, R1={val_metrics['rank1']:.4f}, "
            f"R5={val_metrics['rank5']:.4f}, RIIC={val_metrics['riic']:.4f}"
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "val_metrics": val_metrics,
            "config": cfg,
            "num_classes": train_dataset.num_classes,
            "embed_dim": int(cfg["model"]["embed_dim"]),
        }
        torch.save(ckpt_payload, os.path.join(save_dir, "last.pth"))

        if run_val and val_metrics["mAP"] > best_val_map:
            best_val_map = val_metrics["mAP"]
            best_epoch = epoch
            torch.save(ckpt_payload, os.path.join(save_dir, "best.pth"))
            print(f"  -> New best checkpoint saved (epoch {epoch + 1}, val_mAP={best_val_map:.4f})")

    elapsed = time.time() - t0
    run_meta["elapsed_seconds"] = round(elapsed, 2)
    run_meta["best_epoch"] = best_epoch + 1
    run_meta["best_val_mAP"] = float(best_val_map)
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training completed in {elapsed:.1f}s. Best val_mAP={best_val_map:.4f} at epoch {best_epoch + 1}.")
    return history
