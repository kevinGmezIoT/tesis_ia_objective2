import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_open_set_split(
    pid_stats: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    """
    Assigns PIDs to train/val/test with disjoint identities.
    PIDs with < 2 cameras go strictly to train.
    Remaining PIDs are stratified by n_seqs quantiles to avoid imbalance.
    """
    rng = random.Random(seed)

    single_cam_pids = set(pid_stats.loc[pid_stats["n_cams"] < 2, "person_id"].tolist())
    multi_df = pid_stats[pid_stats["n_cams"] >= 2].sort_values(
        ["n_seqs", "person_id"], ascending=[True, True]
    ).reset_index(drop=True)

    n_multi = len(multi_df)
    total_pids = len(pid_stats)
    total_r = train_ratio + val_ratio + test_ratio
    val_frac = val_ratio / total_r
    test_frac = test_ratio / total_r

    target_val = max(1, round(total_pids * val_frac))
    target_test = max(1, round(total_pids * test_frac))

    # Build quantile blocks so val/test receive balanced representation of low/mid/high seq counts
    n_eval = target_val + target_test
    block_size = max(3, round(n_multi / max(1, target_val)))

    train_pids = set(single_cam_pids)
    val_pids = set()
    test_pids = set()

    blocks = [
        multi_df.iloc[i : i + block_size]["person_id"].tolist()
        for i in range(0, n_multi, block_size)
    ]

    # Deterministically assign 1 val and 1 test per block while counts remain, then remainder to train
    for idx, blk in enumerate(blocks):
        blk_shuffled = list(blk)
        rng.shuffle(blk_shuffled)
        if len(blk_shuffled) >= 2 and len(val_pids) < target_val and len(test_pids) < target_test:
            # Alternate order per block to avoid systematic bias
            if idx % 2 == 0:
                val_pids.add(blk_shuffled.pop())
                test_pids.add(blk_shuffled.pop())
            else:
                test_pids.add(blk_shuffled.pop())
                val_pids.add(blk_shuffled.pop())
        elif len(blk_shuffled) >= 1 and len(val_pids) < target_val:
            val_pids.add(blk_shuffled.pop())
        elif len(blk_shuffled) >= 1 and len(test_pids) < target_test:
            test_pids.add(blk_shuffled.pop())
        train_pids.update(blk_shuffled)

    # Top up val/test if block sizes didn't fill exact targets
    remaining_train_multi = [p for p in multi_df["person_id"].tolist() if p in train_pids]
    rng.shuffle(remaining_train_multi)
    while len(val_pids) < target_val and remaining_train_multi:
        p = remaining_train_multi.pop()
        train_pids.remove(p)
        val_pids.add(p)
    while len(test_pids) < target_test and remaining_train_multi:
        p = remaining_train_multi.pop()
        train_pids.remove(p)
        test_pids.add(p)

    return train_pids, val_pids, test_pids, single_cam_pids


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SGREID v2 open-set split from metadata CSVs in inputs/ and images in images-root."
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="inputs",
        help="Directory containing sequences.csv, sequences_illumination.csv, gait_cycles_occlusion.csv, gait_cycles_quality.csv, confirmed_ambiguous_identity_pairs.csv.",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="D:/tesis_dev/result1/dataset/output",
        help="Root directory containing the rgb/ image folders referenced by sequences.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="processed/sgreid_v2_split.csv",
        help="Output split CSV path.",
    )
    parser.add_argument(
        "--manifest-json",
        type=str,
        default="processed/sgreid_v2_manifest.json",
        help="Output manifest JSON path.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir).resolve()
    images_root = Path(args.images_root).resolve()
    seq_path = metadata_dir / "sequences.csv"
    ill_path = metadata_dir / "sequences_illumination.csv"
    occ_path = metadata_dir / "gait_cycles_occlusion.csv"
    qual_path = metadata_dir / "gait_cycles_quality.csv"
    amb_path = metadata_dir / "confirmed_ambiguous_identity_pairs.csv"

    for p in [seq_path, ill_path, occ_path, qual_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required metadata CSV not found: {p}")

    seq_df = pd.read_csv(seq_path)
    ill_df = pd.read_csv(
        ill_path,
        usecols=["sequence_id", "illum_level", "illum_label", "illum_y_mean", "illum_y_cv"],
    )
    occ_df = pd.read_csv(
        occ_path,
        usecols=["sequence_id", "occlusion_level", "occlusion_label", "occlusion_pct"],
    )
    qual_df = pd.read_csv(
        qual_path,
        usecols=["sequence_id", "sil_quality_level", "sil_quality_label", "sil_quality_mean"],
    )

    occ_df["occlusion_label"] = occ_df["occlusion_label"].fillna("desconocida")

    merged = seq_df.merge(ill_df, on="sequence_id", how="inner", validate="one_to_one")
    merged = merged.merge(occ_df, on="sequence_id", how="inner", validate="one_to_one")
    merged = merged.merge(qual_df, on="sequence_id", how="inner", validate="one_to_one")

    assert len(merged) == 10672, f"Expected 10672 sequences, got {len(merged)}"
    assert merged["sequence_id"].is_unique, "Duplicate sequence_id detected after merge"

    # Standardized columns while preserving original metadata
    merged["frames_dir"] = merged["rgb_dir"].astype(str).str.replace("\\", "/", regex=False)
    merged["person_id"] = merged["person_global_id"].astype(int)
    merged["scene"] = merged["scene_id"].astype(str)

    pid_stats = (
        merged.groupby("person_id")
        .agg(n_seqs=("sequence_id", "count"), n_cams=("cam_id", "nunique"))
        .reset_index()
    )

    train_pids, val_pids, test_pids, single_cam_pids = stratified_open_set_split(
        pid_stats,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Mandatory disjointness & multi-camera assertions
    assert train_pids.isdisjoint(val_pids), "Overlap between train and val PIDs!"
    assert train_pids.isdisjoint(test_pids), "Overlap between train and test PIDs!"
    assert val_pids.isdisjoint(test_pids), "Overlap between val and test PIDs!"
    assert len(train_pids | val_pids | test_pids) == merged["person_id"].nunique()

    cam_counts = pid_stats.set_index("person_id")["n_cams"].to_dict()
    assert all(cam_counts[p] >= 2 for p in val_pids), "Single-cam PID in val!"
    assert all(cam_counts[p] >= 2 for p in test_pids), "Single-cam PID in test!"

    def map_split(pid: int) -> str:
        if pid in train_pids:
            return "train"
        if pid in val_pids:
            return "val"
        return "test"

    merged["split"] = merged["person_id"].map(map_split)

    # Spot-check frames_dir existence and jpg files on disk
    sample_dirs = merged["frames_dir"].iloc[::50].tolist()
    for rel_dir in sample_dirs:
        full_dir = images_root / rel_dir
        if not full_dir.is_dir():
            raise FileNotFoundError(f"RGB directory missing: {full_dir}")
        jpgs = list(full_dir.glob("*.jpg"))
        if not jpgs:
            raise FileNotFoundError(f"No .jpg frames found in: {full_dir}")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8")

    # Ambiguous pairs coverage per split
    amb_summary = {}
    if amb_path.exists():
        amb_df = pd.read_csv(amb_path)
        for s_name, s_pids in [("train", train_pids), ("val", val_pids), ("test", test_pids)]:
            both_in = amb_df[
                amb_df["person_global_id_x"].isin(s_pids)
                & amb_df["person_global_id_y"].isin(s_pids)
            ]
            any_in = amb_df[
                amb_df["person_global_id_x"].isin(s_pids)
                | amb_df["person_global_id_y"].isin(s_pids)
            ]
            amb_summary[s_name] = {
                "pairs_both_in_split": int(len(both_in)),
                "pairs_at_least_one_in_split": int(len(any_in)),
            }

    split_summary = {}
    for s_name in ["train", "val", "test"]:
        sub = merged[merged["split"] == s_name]
        split_summary[s_name] = {
            "num_sequences": int(len(sub)),
            "num_pids": int(sub["person_id"].nunique()),
            "pids": sorted(int(p) for p in sub["person_id"].unique()),
            "illum_distribution": {str(k): int(v) for k, v in sub["illum_label"].value_counts().items()},
            "occlusion_distribution": {str(k): int(v) for k, v in sub["occlusion_label"].value_counts().items()},
        }

    manifest = {
        "metadata_dir": str(metadata_dir).replace("\\", "/"),
        "images_root": str(images_root).replace("\\", "/"),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "total_sequences": int(len(merged)),
        "total_pids": int(merged["person_id"].nunique()),
        "single_cam_pids_forced_to_train": sorted(int(p) for p in single_cam_pids),
        "disjoint_check": {
            "train_val_overlap": len(train_pids & val_pids),
            "train_test_overlap": len(train_pids & test_pids),
            "val_test_overlap": len(val_pids & test_pids),
        },
        "splits": split_summary,
        "ambiguous_pairs_summary": amb_summary,
        "input_sha256": {
            "sequences.csv": sha256_file(seq_path),
            "sequences_illumination.csv": sha256_file(ill_path),
            "gait_cycles_occlusion.csv": sha256_file(occ_path),
            "gait_cycles_quality.csv": sha256_file(qual_path),
        },
        "output_csv": str(out_csv).replace("\\", "/"),
        "output_csv_sha256": sha256_file(out_csv),
    }

    manifest_path = Path(args.manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved split CSV to {out_csv} ({len(merged)} rows, {merged['person_id'].nunique()} PIDs)")
    print(f"Saved manifest to {manifest_path}")
    for s_name, info in split_summary.items():
        print(f"  {s_name}: {info['num_pids']} PIDs, {info['num_sequences']} sequences")


if __name__ == "__main__":
    main()
