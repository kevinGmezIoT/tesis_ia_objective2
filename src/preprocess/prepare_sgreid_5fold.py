import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_pair_aware_blocks(
    pid_stats: pd.DataFrame,
    amb_edges: list[tuple[int, int]],
    num_folds: int,
    seed: int,
):
    """
    Deterministically partitions multi-camera PIDs into `num_folds` balanced blocks
    while keeping a greedy disjoint matching of ambiguous pairs co-located in the same block.
    Single-camera PIDs (n_cams < 2) are returned separately and forced to train in every fold.
    """
    rng = random.Random(seed)

    single_cam_pids = sorted(
        int(p) for p in pid_stats.loc[pid_stats["n_cams"] < 2, "person_id"].tolist()
    )
    multi_df = pid_stats[pid_stats["n_cams"] >= 2].copy()
    multi_pids = sorted(int(p) for p in multi_df["person_id"].tolist())
    multi_set = set(multi_pids)
    seq_counts = {int(r["person_id"]): int(r["n_seqs"]) for _, r in multi_df.iterrows()}

    # Build adjacency over multi-camera ambiguous pairs
    adj: dict[int, set[int]] = defaultdict(set)
    filtered_edges: list[tuple[int, int]] = []
    for u, v in amb_edges:
        if u in multi_set and v in multi_set and u != v:
            pair = (min(u, v), max(u, v))
            if pair[1] not in adj[pair[0]]:
                filtered_edges.append(pair)
                adj[u].add(v)
                adj[v].add(u)

    # Deterministic tie-breaking order generated from master seed
    nodes_shuffled = sorted(adj.keys())
    rng.shuffle(nodes_shuffled)
    tie_rank = {node: idx for idx, node in enumerate(nodes_shuffled)}

    matched: set[int] = set()
    atomic_pairs: list[tuple[int, int]] = []
    for u in sorted(adj.keys(), key=lambda x: (len(adj[x]), tie_rank[x])):
        if u in matched:
            continue
        candidates = [v for v in adj[u] if v not in matched]
        if not candidates:
            continue
        candidates.sort(key=lambda x: (len(adj[x]), tie_rank[x]))
        v = candidates[0]
        matched.add(u)
        matched.add(v)
        atomic_pairs.append((min(u, v), max(u, v)))

    singletons = [p for p in multi_pids if p not in matched]

    n_multi = len(multi_pids)
    caps = [n_multi // num_folds + (1 if i < (n_multi % num_folds) else 0) for i in range(num_folds)]
    blocks: list[list[int]] = [[] for _ in range(num_folds)]
    block_seqs = [0] * num_folds
    block_pairs = [0] * num_folds

    # Place atomic pairs (heaviest total sequences first) balancing pair counts then total sequences
    sorted_pairs = sorted(
        atomic_pairs,
        key=lambda uv: (-(seq_counts[uv[0]] + seq_counts[uv[1]]), uv[0], uv[1]),
    )
    for u, v in sorted_pairs:
        elig = [b for b in range(num_folds) if len(blocks[b]) + 2 <= caps[b]]
        if not elig:
            raise RuntimeError(
                f"Cannot place atomic pair ({u}, {v}) into {num_folds} blocks with capacities {caps}."
            )
        best_b = min(elig, key=lambda b: (block_pairs[b], block_seqs[b], len(blocks[b]), b))
        blocks[best_b].extend([u, v])
        block_seqs[best_b] += seq_counts[u] + seq_counts[v]
        block_pairs[best_b] += 1

    # Place singletons (heaviest first) balancing total sequences
    for p in sorted(singletons, key=lambda x: (-seq_counts[x], x)):
        elig = [b for b in range(num_folds) if len(blocks[b]) + 1 <= caps[b]]
        if not elig:
            raise RuntimeError(f"Cannot place singleton PID {p} into {num_folds} blocks.")
        best_b = min(elig, key=lambda b: (block_seqs[b], len(blocks[b]), b))
        blocks[best_b].append(p)
        block_seqs[best_b] += seq_counts[p]

    blocks = [sorted(b) for b in blocks]
    return blocks, single_cam_pids, filtered_edges, atomic_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Prepare deterministic pair-aware 5-fold open-set splits of SGREID."
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="inputs",
        help="Directory with sequences.csv, illumination, occlusion, quality, and ambiguous pairs CSVs.",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="D:/tesis_dev/result1/dataset/output",
        help="Root directory containing rgb/ folders referenced by sequences.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed/splits_5fold",
        help="Output directory for fold_0.csv..fold_{K-1}.csv and sgreid_5fold_manifest.json.",
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip disk spot-check of RGB directories (useful when generating splits on a host without images).",
    )
    args = parser.parse_args()

    if args.num_folds < 3:
        raise ValueError("--num-folds must be at least 3 so train, val, and test are all non-empty.")

    metadata_dir = Path(args.metadata_dir).resolve()
    images_root = Path(args.images_root).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_path = metadata_dir / "sequences.csv"
    ill_path = metadata_dir / "sequences_illumination.csv"
    occ_path = metadata_dir / "gait_cycles_occlusion.csv"
    qual_path = metadata_dir / "gait_cycles_quality.csv"
    amb_path = metadata_dir / "confirmed_ambiguous_identity_pairs.csv"

    for p in [seq_path, ill_path, occ_path, qual_path, amb_path]:
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
    amb_df = pd.read_csv(amb_path)

    occ_df["occlusion_label"] = occ_df["occlusion_label"].fillna("desconocida")

    merged = seq_df.merge(ill_df, on="sequence_id", how="inner", validate="one_to_one")
    merged = merged.merge(occ_df, on="sequence_id", how="inner", validate="one_to_one")
    merged = merged.merge(qual_df, on="sequence_id", how="inner", validate="one_to_one")

    assert len(merged) == 10672, f"Expected 10672 sequences, got {len(merged)}"
    assert merged["sequence_id"].is_unique, "Duplicate sequence_id detected after merge"

    merged["frames_dir"] = merged["rgb_dir"].astype(str).str.replace("\\", "/", regex=False)
    merged["person_id"] = merged["person_global_id"].astype(int)
    merged["scene"] = merged["scene_id"].astype(str)

    if not args.skip_image_check:
        sample_dirs = merged["frames_dir"].iloc[::50].tolist()
        for rel_dir in sample_dirs:
            full_dir = images_root / rel_dir
            if not full_dir.is_dir():
                raise FileNotFoundError(f"RGB directory missing: {full_dir}")
            if not any(full_dir.glob("*.jpg")):
                raise FileNotFoundError(f"No .jpg frames found in: {full_dir}")

    pid_stats = (
        merged.groupby("person_id")
        .agg(n_seqs=("sequence_id", "count"), n_cams=("cam_id", "nunique"))
        .reset_index()
    )
    good_sil_df = merged[merged["sil_quality_label"] != "no_confiable"]
    good_cam_counts = good_sil_df.groupby("person_id")["cam_id"].nunique().to_dict()
    cam_counts = pid_stats.set_index("person_id")["n_cams"].to_dict()

    amb_edges = [
        (int(r["person_global_id_x"]), int(r["person_global_id_y"]))
        for _, r in amb_df.iterrows()
    ]

    blocks, single_cam_pids, multi_edges, atomic_pairs = build_pair_aware_blocks(
        pid_stats=pid_stats,
        amb_edges=amb_edges,
        num_folds=args.num_folds,
        seed=args.seed,
    )

    pid_to_block = {p: -1 for p in single_cam_pids}
    for b_idx, blk in enumerate(blocks):
        for p in blk:
            pid_to_block[p] = b_idx
    merged["fold_block"] = merged["person_id"].map(pid_to_block)

    all_pids = set(int(p) for p in merged["person_id"].unique())
    multi_pids = {p for p in all_pids if p not in set(single_cam_pids)}

    # Global disjointness across test blocks
    all_test_pids: set[int] = set()
    for b_idx, blk in enumerate(blocks):
        bset = set(blk)
        assert all_test_pids.isdisjoint(bset), f"Overlap detected between test block {b_idx} and prior blocks!"
        all_test_pids.update(bset)
    assert all_test_pids == multi_pids, "Union of test blocks does not match all multi-camera PIDs!"

    def complete_pairs_in(pid_set: set[int]) -> list[list[int]]:
        return sorted(
            [min(u, v), max(u, v)]
            for (u, v) in multi_edges
            if u in pid_set and v in pid_set
        )

    folds_manifest = {}
    for k in range(args.num_folds):
        val_idx = (k + 1) % args.num_folds
        test_pids = set(blocks[k])
        val_pids = set(blocks[val_idx])
        train_pids = set(single_cam_pids)
        for b_idx in range(args.num_folds):
            if b_idx not in (k, val_idx):
                train_pids.update(blocks[b_idx])

        # Mandatory safety assertions per fold
        assert train_pids.isdisjoint(val_pids), f"Fold {k}: train/val overlap"
        assert train_pids.isdisjoint(test_pids), f"Fold {k}: train/test overlap"
        assert val_pids.isdisjoint(test_pids), f"Fold {k}: val/test overlap"
        assert (train_pids | val_pids | test_pids) == all_pids, f"Fold {k}: missing PIDs"

        assert all(cam_counts[p] >= 2 for p in val_pids), f"Fold {k}: single-cam PID in val!"
        assert all(cam_counts[p] >= 2 for p in test_pids), f"Fold {k}: single-cam PID in test!"
        assert all(good_cam_counts.get(p, 0) >= 2 for p in val_pids), (
            f"Fold {k}: PID in val loses >=2 cam coverage when filtering unreliable silhouettes!"
        )
        assert all(good_cam_counts.get(p, 0) >= 2 for p in test_pids), (
            f"Fold {k}: PID in test loses >=2 cam coverage when filtering unreliable silhouettes!"
        )

        test_complete_pairs = complete_pairs_in(test_pids)
        val_complete_pairs = complete_pairs_in(val_pids)
        train_complete_pairs = complete_pairs_in(train_pids)
        if args.num_folds == 5:
            assert len(test_complete_pairs) >= 5, (
                f"Fold {k}: expected >= 5 complete ambiguous pairs in test, got {len(test_complete_pairs)}"
            )

        def map_split(pid: int) -> str:
            if pid in train_pids:
                return "train"
            if pid in val_pids:
                return "val"
            return "test"

        fold_df = merged.copy()
        fold_df["split"] = fold_df["person_id"].map(map_split)
        fold_csv = out_dir / f"fold_{k}.csv"
        fold_df.to_csv(fold_csv, index=False, encoding="utf-8")

        split_info = {}
        for s_name in ["train", "val", "test"]:
            sub = fold_df[fold_df["split"] == s_name]
            pid_seq_counts = sub["person_id"].value_counts()
            max_pid = int(pid_seq_counts.index[0])
            max_seqs = int(pid_seq_counts.iloc[0])
            split_info[s_name] = {
                "num_sequences": int(len(sub)),
                "num_pids": int(sub["person_id"].nunique()),
                "pids": sorted(int(p) for p in sub["person_id"].unique()),
                "max_pid": max_pid,
                "max_pid_sequences": max_seqs,
                "max_pid_share": round(max_seqs / max(1, len(sub)), 4),
                "illum_distribution": {str(k_): int(v_) for k_, v_ in sub["illum_label"].value_counts().items()},
                "occlusion_distribution": {
                    str(k_): int(v_) for k_, v_ in sub["occlusion_label"].value_counts().items()
                },
                "reliable_sil_sequences": int((sub["sil_quality_label"] != "no_confiable").sum()),
            }

        folds_manifest[f"fold_{k}"] = {
            "fold_index": k,
            "test_block": k,
            "val_block": val_idx,
            "csv_path": str(fold_csv).replace("\\", "/"),
            "csv_sha256": sha256_file(fold_csv),
            "disjoint_check": {
                "train_val_overlap": 0,
                "train_test_overlap": 0,
                "val_test_overlap": 0,
            },
            "ambiguous_pairs": {
                "train_complete_count": len(train_complete_pairs),
                "val_complete_count": len(val_complete_pairs),
                "val_complete_pairs": val_complete_pairs,
                "test_complete_count": len(test_complete_pairs),
                "test_complete_pairs": test_complete_pairs,
            },
            "splits": split_info,
        }

    manifest = {
        "protocol": f"{args.num_folds}-fold pair-aware open-set cross-validation",
        "metadata_dir": str(metadata_dir).replace("\\", "/"),
        "images_root": str(images_root).replace("\\", "/"),
        "image_spot_check_executed": not args.skip_image_check,
        "seed": args.seed,
        "num_folds": args.num_folds,
        "total_sequences": int(len(merged)),
        "total_pids": int(len(all_pids)),
        "multi_cam_pids": len(multi_pids),
        "single_cam_pids_forced_to_train": single_cam_pids,
        "atomic_disjoint_pairs_used": [[u, v] for u, v in atomic_pairs],
        "blocks": {f"block_{i}": blocks[i] for i in range(args.num_folds)},
        "oe3_silhouette_multicam_preserved": True,
        "input_sha256": {
            "sequences.csv": sha256_file(seq_path),
            "sequences_illumination.csv": sha256_file(ill_path),
            "gait_cycles_occlusion.csv": sha256_file(occ_path),
            "gait_cycles_quality.csv": sha256_file(qual_path),
            "confirmed_ambiguous_identity_pairs.csv": sha256_file(amb_path),
        },
        "folds": folds_manifest,
    }

    manifest_path = out_dir / "sgreid_5fold_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Generated {args.num_folds}-fold pair-aware open-set splits in {out_dir}")
    print(f"Manifest saved to {manifest_path}")
    for k in range(args.num_folds):
        fm = folds_manifest[f"fold_{k}"]
        tr = fm["splits"]["train"]
        va = fm["splits"]["val"]
        te = fm["splits"]["test"]
        amb_t = fm["ambiguous_pairs"]["test_complete_count"]
        print(
            f"  Fold {k}: train={tr['num_pids']} PIDs ({tr['num_sequences']} seqs), "
            f"val={va['num_pids']} PIDs ({va['num_sequences']} seqs), "
            f"test={te['num_pids']} PIDs ({te['num_sequences']} seqs, "
            f"max_share={te['max_pid_share']*100:.1f}%, complete_ambig_pairs={amb_t})"
        )


if __name__ == "__main__":
    main()
