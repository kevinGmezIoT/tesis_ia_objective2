import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_experiment(exp_dir):
    history_path = os.path.join(exp_dir, "history.json")
    if not os.path.exists(history_path):
        return None

    eval_files = glob.glob(os.path.join(exp_dir, "categorized_eval_*.json"))
    if not eval_files:
        return None
    eval_path = eval_files[0]

    npz_files = glob.glob(os.path.join(exp_dir, "per_query_ap_*.npz"))
    npz_path = npz_files[0] if npz_files else None

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    per_query = None
    if npz_path and os.path.exists(npz_path):
        data = np.load(npz_path)
        per_query = {
            "sequence_id": data["sequence_id"],
            "ap": data["ap"],
        }

    return {
        "name": eval_data.get("experiment_name", os.path.basename(exp_dir)),
        "dir": exp_dir,
        "history": history,
        "eval": eval_data,
        "per_query": per_query,
    }


def paired_stat_test(exp_a, exp_b):
    if exp_a["per_query"] is None or exp_b["per_query"] is None:
        return None

    map_a = dict(zip(exp_a["per_query"]["sequence_id"].tolist(), exp_a["per_query"]["ap"].tolist()))
    map_b = dict(zip(exp_b["per_query"]["sequence_id"].tolist(), exp_b["per_query"]["ap"].tolist()))
    common_ids = sorted(set(map_a.keys()) & set(map_b.keys()))
    if len(common_ids) < 10:
        return None

    a_vals = np.array([map_a[sid] for sid in common_ids], dtype=np.float64)
    b_vals = np.array([map_b[sid] for sid in common_ids], dtype=np.float64)
    diff = a_vals - b_vals

    t_stat, t_p = stats.ttest_rel(a_vals, b_vals)
    if np.allclose(diff, 0.0):
        w_stat, w_p = 0.0, 1.0
    else:
        w_stat, w_p = stats.wilcoxon(a_vals, b_vals)

    return {
        "comparison": f"{exp_a['name']} vs {exp_b['name']}",
        "num_common_queries": len(common_ids),
        "mean_ap_a": float(np.mean(a_vals)),
        "mean_ap_b": float(np.mean(b_vals)),
        "mean_diff": float(np.mean(diff)),
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare OE2 experiments and compute paired statistical tests.")
    parser.add_argument("--exp-root", type=str, default="outputs/oe2_sgreid")
    parser.add_argument("--exp-dirs", nargs="*", default=None, help="Explicit list of experiment directories.")
    args = parser.parse_args()

    if args.exp_dirs:
        candidate_dirs = args.exp_dirs
    else:
        if not os.path.isdir(args.exp_root):
            print(f"Experiment root not found: {args.exp_root}")
            return
        candidate_dirs = [
            os.path.join(args.exp_root, d)
            for d in sorted(os.listdir(args.exp_root))
            if os.path.isdir(os.path.join(args.exp_root, d))
        ]

    exps = []
    for d in candidate_dirs:
        loaded = load_experiment(d)
        if loaded is not None:
            exps.append(loaded)

    if not exps:
        print("No valid completed experiments found.")
        return

    out_dir = args.exp_root if not args.exp_dirs else os.path.dirname(candidate_dirs[0])
    os.makedirs(out_dir, exist_ok=True)

    # 1. History comparison plot (train_loss, val_mAP, val_rank1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for exp in exps:
        h = exp["history"]
        label = exp["name"]
        axes[0].plot(h["train_loss"], label=label)
        if "val_mAP" in h:
            axes[1].plot(h["val_mAP"], label=label)
        if "val_rank1" in h:
            axes[2].plot(h["val_rank1"], label=label)

    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Validation mAP (Open-Set)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].legend()

    axes[2].set_title("Validation Rank-1 (Open-Set)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Rank-1")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "history_comparison.png"), dpi=150)
    plt.close()

    # 2. Overall test metrics comparison
    metrics_keys = ["mAP", "rank1", "rank5", "ratio_inter_intra"]
    metric_labels = ["mAP", "Rank-1", "Rank-5", "RIIC"]
    x = np.arange(len(metrics_keys))
    width = 0.8 / len(exps)

    plt.figure(figsize=(10, 5.5))
    for i, exp in enumerate(exps):
        ov = exp["eval"]["illumination"]["OVERALL"]
        vals = [ov[k] for k in metrics_keys]
        offset = (i - (len(exps) - 1) / 2.0) * width
        bars = plt.bar(x + offset, vals, width, label=exp["name"])
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2.0, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, metric_labels)
    plt.ylabel("Score")
    plt.title("Test Metrics Comparison (Open-Set SGREID)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_comparison.png"), dpi=150)
    plt.close()

    # 3. Categorized comparison (Illumination + Occlusion mAP)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    illum_cats = [c for c in ["baja", "media_baja", "media_alta", "alta"] if c in exps[0]["eval"]["illumination"]]
    if illum_cats:
        xi = np.arange(len(illum_cats))
        wi = 0.8 / len(exps)
        for i, exp in enumerate(exps):
            vals = [exp["eval"]["illumination"].get(c, {}).get("mAP", 0.0) for c in illum_cats]
            axes[0].bar(xi + (i - (len(exps) - 1) / 2.0) * wi, vals, wi, label=exp["name"])
        axes[0].set_xticks(xi)
        axes[0].set_xticklabels(illum_cats)
        axes[0].set_title("mAP by Illumination Level")
        axes[0].set_ylabel("mAP")
        axes[0].legend()

    occl_cats = [c for c in ["baja", "parcial", "alta"] if c in exps[0]["eval"]["occlusion"]]
    if occl_cats:
        xo = np.arange(len(occl_cats))
        wo = 0.8 / len(exps)
        for i, exp in enumerate(exps):
            vals = [exp["eval"]["occlusion"].get(c, {}).get("mAP", 0.0) for c in occl_cats]
            axes[1].bar(xo + (i - (len(exps) - 1) / 2.0) * wo, vals, wo, label=exp["name"])
        axes[1].set_xticks(xo)
        axes[1].set_xticklabels(occl_cats)
        axes[1].set_title("mAP by Occlusion Level")
        axes[1].set_ylabel("mAP")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "categorized_comparison.png"), dpi=150)
    plt.close()

    # 4. Statistical comparison & efficiency summary
    cnn_exp = next((e for e in exps if e["eval"].get("arch") == "cnn"), None)
    paired_tests = []
    if cnn_exp is not None:
        for exp in exps:
            if exp is cnn_exp:
                continue
            res = paired_stat_test(exp, cnn_exp)
            if res is not None:
                paired_tests.append(res)

    summary_table = []
    for exp in exps:
        ev = exp["eval"]
        ov = ev["illumination"]["OVERALL"]
        summary_table.append({
            "experiment": exp["name"],
            "arch": ev.get("arch"),
            "backbone": ev.get("backbone"),
            "embed_dim": ev.get("embed_dim"),
            "num_params": ev.get("num_params"),
            "avg_latency_ms_per_seq": ev.get("avg_latency_ms_per_seq"),
            "mAP": ov["mAP"],
            "rank1": ov["rank1"],
            "rank5": ov["rank5"],
            "rank10": ov["rank10"],
            "ratio_inter_intra": ov["ratio_inter_intra"],
        })

    stat_out = {
        "models": summary_table,
        "paired_tests_vs_cnn": paired_tests,
    }

    stat_path = os.path.join(out_dir, "statistical_comparison.json")
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump(stat_out, f, indent=4, ensure_ascii=False)

    print(f"Comparison plots and {stat_path} saved to {out_dir}")
    print(json.dumps(stat_out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
