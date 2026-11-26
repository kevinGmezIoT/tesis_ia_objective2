import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def canonical_model_key(exp_dir_name: str, eval_data: dict) -> str:
    name = eval_data.get("experiment_name") or exp_dir_name
    name = re.sub(r"_fold\d+$", "", name)
    return name


def load_fold_experiments(exp_root: str):
    fold_dirs = sorted(
        [
            d
            for d in glob.glob(os.path.join(exp_root, "fold_*"))
            if os.path.isdir(d) and re.search(r"fold_(\d+)$", d)
        ],
        key=lambda p: int(re.search(r"fold_(\d+)$", p).group(1)),
    )

    folds = {}
    for fdir in fold_dirs:
        fold_idx = int(re.search(r"fold_(\d+)$", fdir).group(1))
        fold_models = {}
        for sub in sorted(os.listdir(fdir)):
            mdir = os.path.join(fdir, sub)
            if not os.path.isdir(mdir):
                continue
            eval_files = glob.glob(os.path.join(mdir, "categorized_eval_*.json"))
            if not eval_files:
                continue
            with open(eval_files[0], "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            npz_files = glob.glob(os.path.join(mdir, "per_query_ap_*.npz"))
            per_query = None
            if npz_files:
                arr = np.load(npz_files[0])
                per_query = {
                    "sequence_id": arr["sequence_id"].tolist(),
                    "ap": arr["ap"].astype(np.float64).tolist(),
                    "person_id": arr["person_id"].astype(np.int64).tolist()
                    if "person_id" in arr
                    else None,
                }

            model_key = canonical_model_key(sub, eval_data)
            fold_models[model_key] = {
                "dir": mdir,
                "eval": eval_data,
                "per_query": per_query,
            }
        if fold_models:
            folds[fold_idx] = fold_models
    return folds


def mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0, "values": []}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0,
        "n": int(len(arr)),
        "values": [float(v) for v in arr],
    }


def extract_per_pid_ap(exp_entry: dict) -> dict[int, tuple[float, int]]:
    ev = exp_entry["eval"]
    if isinstance(ev.get("per_pid_ap"), dict) and ev["per_pid_ap"]:
        return {
            int(pid): (float(info["ap"]), int(info.get("n_queries", 0)))
            for pid, info in ev["per_pid_ap"].items()
        }
    pq = exp_entry.get("per_query")
    if pq and pq.get("person_id") is not None:
        by_pid = {}
        for pid, ap in zip(pq["person_id"], pq["ap"]):
            by_pid.setdefault(int(pid), []).append(float(ap))
        return {pid: (float(np.mean(vals)), len(vals)) for pid, vals in by_pid.items()}
    return {}


def compute_paired_stats(a_vals: np.ndarray, b_vals: np.ndarray, label_a: str, label_b: str) -> dict | None:
    if len(a_vals) < 2:
        return None
    diff = a_vals - b_vals
    t_stat, t_p = stats.ttest_rel(a_vals, b_vals)
    if np.allclose(diff, 0.0):
        w_stat, w_p = 0.0, 1.0
    else:
        w_stat, w_p = stats.wilcoxon(a_vals, b_vals)
    return {
        "comparison": f"{label_a} vs {label_b}",
        "n": int(len(a_vals)),
        "mean_a": float(np.mean(a_vals)),
        "mean_b": float(np.mean(b_vals)),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)) if len(diff) >= 2 else 0.0,
        "wins_a_over_b": int(np.sum(diff > 0)),
        "ties": int(np.sum(np.isclose(diff, 0.0))),
        "wins_b_over_a": int(np.sum(diff < 0)),
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate 5-fold open-set OE2 results with OOF statistical tests.")
    parser.add_argument("--exp-root", type=str, default="outputs/oe2_sgreid_5fold")
    args = parser.parse_args()

    folds = load_fold_experiments(args.exp_root)
    if not folds:
        print(f"No completed fold evaluations found in {args.exp_root}")
        return

    fold_indices = sorted(folds.keys())
    all_model_keys = sorted({m for f in fold_indices for m in folds[f].keys()})
    # Put CNN baselines first, then ViT-Base, ViT-Small, and others
    def model_sort_key(m: str):
        ml = m.lower()
        if "cnn" in ml or "resnet" in ml:
            return (0, ml)
        if "vit_base" in ml:
            return (1, ml)
        if "vit_small_sgreid" in ml:
            return (2, ml)
        if "vit_small" in ml:
            return (3, ml)
        return (4, ml)

    all_model_keys = sorted(all_model_keys, key=model_sort_key)

    models_summary = {}
    oof_pid_maps: dict[str, dict[int, tuple[int, float, int]]] = {m: {} for m in all_model_keys}
    oof_query_maps: dict[str, dict[str, float]] = {m: {} for m in all_model_keys}
    csv_rows = []

    for m in all_model_keys:
        per_fold_entries = [(f, folds[f][m]) for f in fold_indices if m in folds[f]]
        micro_maps = []
        macro_maps = []
        r1s, r5s, r10s, riics, lats = [], [], [], [], []
        illum_vals = {k: [] for k in ["baja", "media_baja", "media_alta", "alta"]}
        occl_vals = {k: [] for k in ["baja", "parcial", "alta"]}
        amb_full_map, amb_full_r1 = [], []
        amb_only_map, amb_only_r1 = [], []
        amb_conf_rate, amb_conf_share = [], []
        arch_val, backbone_val, params_val = None, None, None

        for f_idx, entry in per_fold_entries:
            ev = entry["eval"]
            ov = ev["illumination"]["OVERALL"]
            arch_val = ev.get("arch", arch_val)
            backbone_val = ev.get("backbone", backbone_val)
            params_val = ev.get("num_params", params_val)

            micro_maps.append(float(ov["mAP"]))
            pid_ap_map = extract_per_pid_ap(entry)
            if "macro_mAP" in ov:
                macro_val = float(ov["macro_mAP"])
            elif pid_ap_map:
                macro_val = float(np.mean([v[0] for v in pid_ap_map.values()]))
            else:
                macro_val = float(ov["mAP"])
            macro_maps.append(macro_val)

            r1s.append(float(ov["rank1"]))
            r5s.append(float(ov["rank5"]))
            r10s.append(float(ov["rank10"]))
            riics.append(float(ov["ratio_inter_intra"]))
            lats.append(float(ev.get("avg_latency_ms_per_seq", 0.0)))

            for cat in illum_vals:
                if cat in ev["illumination"] and isinstance(ev["illumination"][cat], dict):
                    illum_vals[cat].append(float(ev["illumination"][cat]["mAP"]))
            for cat in occl_vals:
                if cat in ev["occlusion"] and isinstance(ev["occlusion"][cat], dict):
                    occl_vals[cat].append(float(ev["occlusion"][cat]["mAP"]))

            amb = ev.get("ambiguous_pairs", {})
            s_full = amb.get("subset_both_in_test_vs_full_gallery")
            if isinstance(s_full, dict):
                amb_full_map.append(float(s_full["mAP"]))
                amb_full_r1.append(float(s_full["rank1"]))
            s_only = amb.get("subset_both_in_test_only")
            if isinstance(s_only, dict):
                amb_only_map.append(float(s_only["mAP"]))
                amb_only_r1.append(float(s_only["rank1"]))
            conf = amb.get("ambiguous_partner_rank1_confusion")
            if isinstance(conf, dict):
                amb_conf_rate.append(float(conf["partner_confusion_rate_over_queries"]))
                if conf.get("partner_confusion_share_of_errors") is not None:
                    amb_conf_share.append(float(conf["partner_confusion_share_of_errors"]))

            # Accumulate OOF per-PID AP (each multi-camera PID must appear at most once in test)
            for pid, (ap_val, n_q) in pid_ap_map.items():
                if pid in oof_pid_maps[m]:
                    raise ValueError(
                        f"PID {pid} appears in multiple test folds for model {m} "
                        f"(folds {oof_pid_maps[m][pid][0]} and {f_idx}) — not disjoint OOF!"
                    )
                oof_pid_maps[m][pid] = (f_idx, float(ap_val), int(n_q))

            # Accumulate OOF per-query AP
            pq = entry.get("per_query")
            if pq is not None:
                for sid, ap_val in zip(pq["sequence_id"], pq["ap"]):
                    sid_str = str(sid)
                    if sid_str in oof_query_maps[m]:
                        raise ValueError(f"Duplicate sequence_id {sid_str} across test folds for {m}!")
                    oof_query_maps[m][sid_str] = float(ap_val)

        m_summary = {
            "arch": arch_val,
            "backbone": backbone_val,
            "num_params": params_val,
            "folds_completed": [f for f, _ in per_fold_entries],
            "overall": {
                "mAP_micro": mean_std(micro_maps),
                "mAP_macro": mean_std(macro_maps),
                "rank1": mean_std(r1s),
                "rank5": mean_std(r5s),
                "rank10": mean_std(r10s),
                "ratio_inter_intra": mean_std(riics),
                "avg_latency_ms_per_seq": mean_std(lats),
            },
            "illumination_mAP": {k: mean_std(v) for k, v in illum_vals.items()},
            "occlusion_mAP": {k: mean_std(v) for k, v in occl_vals.items()},
            "ambiguous_pairs": {
                "both_in_test_vs_full_gallery_mAP": mean_std(amb_full_map),
                "both_in_test_vs_full_gallery_rank1": mean_std(amb_full_r1),
                "both_in_test_only_mAP": mean_std(amb_only_map),
                "both_in_test_only_rank1": mean_std(amb_only_r1),
                "partner_confusion_rate_over_queries": mean_std(amb_conf_rate),
                "partner_confusion_share_of_errors": mean_std(amb_conf_share),
            },
            "oof_num_pids": len(oof_pid_maps[m]),
            "oof_num_queries": len(oof_query_maps[m]),
        }
        models_summary[m] = m_summary

        csv_rows.append(
            {
                "model": m,
                "arch": arch_val,
                "backbone": backbone_val,
                "num_params": params_val,
                "n_folds": len(per_fold_entries),
                "oof_pids": len(oof_pid_maps[m]),
                "mAP_micro_mean": m_summary["overall"]["mAP_micro"]["mean"],
                "mAP_micro_std": m_summary["overall"]["mAP_micro"]["std"],
                "mAP_macro_mean": m_summary["overall"]["mAP_macro"]["mean"],
                "mAP_macro_std": m_summary["overall"]["mAP_macro"]["std"],
                "rank1_mean": m_summary["overall"]["rank1"]["mean"],
                "rank1_std": m_summary["overall"]["rank1"]["std"],
                "rank5_mean": m_summary["overall"]["rank5"]["mean"],
                "rank5_std": m_summary["overall"]["rank5"]["std"],
                "rank10_mean": m_summary["overall"]["rank10"]["mean"],
                "rank10_std": m_summary["overall"]["rank10"]["std"],
                "riic_mean": m_summary["overall"]["ratio_inter_intra"]["mean"],
                "riic_std": m_summary["overall"]["ratio_inter_intra"]["std"],
                "latency_ms_mean": m_summary["overall"]["avg_latency_ms_per_seq"]["mean"],
                "ambig_full_mAP_mean": m_summary["ambiguous_pairs"]["both_in_test_vs_full_gallery_mAP"]["mean"],
                "ambig_full_mAP_std": m_summary["ambiguous_pairs"]["both_in_test_vs_full_gallery_mAP"]["std"],
                "ambig_full_rank1_mean": m_summary["ambiguous_pairs"]["both_in_test_vs_full_gallery_rank1"]["mean"],
                "ambig_full_rank1_std": m_summary["ambiguous_pairs"]["both_in_test_vs_full_gallery_rank1"]["std"],
                "partner_confusion_rate_mean": m_summary["ambiguous_pairs"]["partner_confusion_rate_over_queries"]["mean"],
                "partner_confusion_rate_std": m_summary["ambiguous_pairs"]["partner_confusion_rate_over_queries"]["std"],
            }
        )

    # Paired statistical tests across all unordered pairs (ordered with stronger/newer vs baseline)
    oof_pid_tests = []
    oof_query_tests = []
    pair_list = []
    for i in range(len(all_model_keys)):
        for j in range(i + 1, len(all_model_keys)):
            # Order (higher_index_model vs lower_index_model) so CNN baseline is B
            pair_list.append((all_model_keys[j], all_model_keys[i]))

    for mod_a, mod_b in pair_list:
        common_pids = sorted(set(oof_pid_maps[mod_a].keys()) & set(oof_pid_maps[mod_b].keys()))
        if len(common_pids) >= 2:
            a_vals = np.array([oof_pid_maps[mod_a][p][1] for p in common_pids], dtype=np.float64)
            b_vals = np.array([oof_pid_maps[mod_b][p][1] for p in common_pids], dtype=np.float64)
            res = compute_paired_stats(a_vals, b_vals, mod_a, mod_b)
            if res is not None:
                oof_pid_tests.append(res)

        common_q = sorted(set(oof_query_maps[mod_a].keys()) & set(oof_query_maps[mod_b].keys()))
        if len(common_q) >= 2:
            qa = np.array([oof_query_maps[mod_a][q] for q in common_q], dtype=np.float64)
            qb = np.array([oof_query_maps[mod_b][q] for q in common_q], dtype=np.float64)
            res_q = compute_paired_stats(qa, qb, mod_a, mod_b)
            if res_q is not None:
                oof_query_tests.append(res_q)

    os.makedirs(args.exp_root, exist_ok=True)
    summary_json_path = os.path.join(args.exp_root, "summary_5fold.json")
    summary_payload = {
        "exp_root": os.path.abspath(args.exp_root).replace("\\", "/"),
        "folds_found": fold_indices,
        "models": models_summary,
        "paired_tests_oof_identity": oof_pid_tests,
        "paired_tests_oof_query": oof_query_tests,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    summary_csv_path = os.path.join(args.exp_root, "summary_5fold_table.csv")
    pd.DataFrame(csv_rows).to_csv(summary_csv_path, index=False, encoding="utf-8")

    # Export OOF per-PID AP table
    all_oof_pids = sorted({p for m in all_model_keys for p in oof_pid_maps[m].keys()})
    pid_rows = []
    for pid in all_oof_pids:
        row = {"person_id": pid}
        fold_found = None
        n_q_found = None
        for m in all_model_keys:
            if pid in oof_pid_maps[m]:
                f_idx, ap_val, n_q = oof_pid_maps[m][pid]
                fold_found = f_idx if fold_found is None else fold_found
                n_q_found = n_q if n_q_found is None else n_q_found
                row[f"ap_{m}"] = ap_val
            else:
                row[f"ap_{m}"] = np.nan
        row["test_fold"] = fold_found
        row["n_queries"] = n_q_found
        pid_rows.append(row)
    per_pid_csv_path = os.path.join(args.exp_root, "per_pid_oof_5fold.csv")
    pd.DataFrame(pid_rows).to_csv(per_pid_csv_path, index=False, encoding="utf-8")

    # Plot 1: Overall 5-fold metrics (with std error bars)
    metric_specs = [
        ("mAP_micro", "mAP (micro)"),
        ("mAP_macro", "mAP (macro)"),
        ("rank1", "Rank-1"),
        ("rank5", "Rank-5"),
    ]
    x = np.arange(len(metric_specs))
    width = 0.8 / max(1, len(all_model_keys))
    plt.figure(figsize=(10.5, 5.2))
    for idx, m in enumerate(all_model_keys):
        ov = models_summary[m]["overall"]
        means = [ov[k]["mean"] or 0.0 for k, _ in metric_specs]
        stds = [ov[k]["std"] or 0.0 for k, _ in metric_specs]
        offset = (idx - (len(all_model_keys) - 1) / 2.0) * width
        bars = plt.bar(x + offset, means, width, yerr=stds, capsize=3, label=m)
        for b, val in zip(bars, means):
            plt.text(b.get_x() + b.get_width() / 2.0, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, [lbl for _, lbl in metric_specs])
    plt.ylim(0.0, 1.12)
    plt.ylabel("Score")
    plt.title(f"5-Fold Open-Set Comparison (mean ± std across {len(fold_indices)} folds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.exp_root, "comparison_5fold_metrics.png"), dpi=150)
    plt.close()

    # Plot 2: Ambiguous pairs performance and partner confusion rate
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x_m = np.arange(len(all_model_keys))
    amb_map_means = [
        models_summary[m]["ambiguous_pairs"]["both_in_test_vs_full_gallery_mAP"]["mean"] or 0.0
        for m in all_model_keys
    ]
    amb_map_stds = [
        models_summary[m]["ambiguous_pairs"]["both_in_test_vs_full_gallery_mAP"]["std"] or 0.0
        for m in all_model_keys
    ]
    amb_r1_means = [
        models_summary[m]["ambiguous_pairs"]["both_in_test_vs_full_gallery_rank1"]["mean"] or 0.0
        for m in all_model_keys
    ]
    amb_r1_stds = [
        models_summary[m]["ambiguous_pairs"]["both_in_test_vs_full_gallery_rank1"]["std"] or 0.0
        for m in all_model_keys
    ]
    w2 = 0.35
    axes[0].bar(x_m - w2 / 2, amb_map_means, w2, yerr=amb_map_stds, capsize=3, label="mAP")
    axes[0].bar(x_m + w2 / 2, amb_r1_means, w2, yerr=amb_r1_stds, capsize=3, label="Rank-1")
    axes[0].set_xticks(x_m)
    axes[0].set_xticklabels(all_model_keys, rotation=15, ha="right")
    axes[0].set_ylim(0.0, 1.12)
    axes[0].set_title("Complete Ambiguous Pairs (Both in Test)")
    axes[0].set_ylabel("Score")
    axes[0].legend()

    conf_means = [
        models_summary[m]["ambiguous_pairs"]["partner_confusion_rate_over_queries"]["mean"] or 0.0
        for m in all_model_keys
    ]
    conf_stds = [
        models_summary[m]["ambiguous_pairs"]["partner_confusion_rate_over_queries"]["std"] or 0.0
        for m in all_model_keys
    ]
    bars_c = axes[1].bar(x_m, conf_means, 0.5, yerr=conf_stds, capsize=3)
    for b, val in zip(bars_c, conf_means):
        axes[1].text(b.get_x() + b.get_width() / 2.0, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x_m)
    axes[1].set_xticklabels(all_model_keys, rotation=15, ha="right")
    axes[1].set_title("Confirmed Ambiguous Partner Rank-1 Confusion Rate (Lower is Better)")
    axes[1].set_ylabel("Fraction of queries confused with ambiguous partner")
    plt.tight_layout()
    plt.savefig(os.path.join(args.exp_root, "comparison_5fold_ambiguous.png"), dpi=150)
    plt.close()

    print(f"5-fold aggregation saved to {args.exp_root}:")
    print(f"  - {summary_json_path}")
    print(f"  - {summary_csv_path}")
    print(f"  - {per_pid_csv_path}")
    print(json.dumps({"oof_identity_tests": oof_pid_tests}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
