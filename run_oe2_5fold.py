import argparse
import os
import subprocess
import sys
from pathlib import Path


MODEL_REGISTRY = {
    "cnn_resnet50": {
        "train_script": "src/train_cnn.py",
        "config": "src/configs/cnn_config.yaml",
        "arch": "cnn",
        "exp_name": "cnn_resnet50_sgreid",
    },
    "vit_base": {
        "train_script": "src/train_vit.py",
        "config": "src/configs/vit_config.yaml",
        "arch": "vit",
        "exp_name": "vit_base_sgreid",
    },
    "vit_small": {
        "train_script": "src/train_vit.py",
        "config": "src/configs/vit_small_config.yaml",
        "arch": "vit",
        "exp_name": "vit_small_sgreid",
    },
    "vit_small_in1k": {
        "train_script": "src/train_vit.py",
        "config": "src/configs/vit_small_in1k_config.yaml",
        "arch": "vit",
        "exp_name": "vit_small_in1k_sgreid",
    },
}


def run_cmd(cmd: list[str], env: dict[str, str], cwd: Path):
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, cwd=str(cwd), check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run resumable 5-fold pair-aware open-set OE2 experiments on SGREID."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="D:/tesis_dev/result1/dataset/output",
        help="Root directory containing rgb/ image folders.",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="processed/splits_5fold",
        help="Directory containing fold_0.csv..fold_4.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/oe2_sgreid_5fold",
        help="Root output directory for 5-fold runs and aggregated summaries.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Fold indices to execute.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cnn_resnet50", "vit_base", "vit_small"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Models to run per fold.",
    )
    parser.add_argument(
        "--include-in1k-control",
        action="store_true",
        help="Append vit_small_in1k (ImageNet-1k pretrained control) to the model list.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override (e.g., smoke test).")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed.")
    parser.add_argument("--force", action="store_true", help="Re-run completed models even if eval JSON exists.")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only evaluate existing best.pth checkpoints, then aggregate.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    splits_dir = Path(args.splits_dir)
    if not splits_dir.is_absolute():
        splits_dir = repo_root / splits_dir
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    models = list(args.models)
    if args.include_in1k_control and "vit_small_in1k" not in models:
        models.append("vit_small_in1k")

    env = os.environ.copy()
    src_dir = str(repo_root / "src")
    env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    py = sys.executable

    # Ensure 5-fold split CSVs exist
    missing_folds = [k for k in args.folds if not (splits_dir / f"fold_{k}.csv").exists()]
    if missing_folds:
        print(f"Missing fold CSVs for folds {missing_folds}; generating 5-fold splits...")
        prep_cmd = [
            py,
            "src/preprocess/prepare_sgreid_5fold.py",
            "--images-root",
            args.data_root,
            "--output-dir",
            str(splits_dir),
            "--seed",
            str(args.seed),
        ]
        if not Path(args.data_root).is_dir():
            prep_cmd.append("--skip-image-check")
        run_cmd(prep_cmd, env=env, cwd=repo_root)

    for k in args.folds:
        fold_csv = splits_dir / f"fold_{k}.csv"
        if not fold_csv.exists():
            raise FileNotFoundError(f"Fold CSV not found: {fold_csv}")

        fold_out_root = output_root / f"fold_{k}"
        fold_out_root.mkdir(parents=True, exist_ok=True)

        print(f"\n==================== FOLD {k} ====================")
        for m_key in models:
            spec = MODEL_REGISTRY[m_key]
            exp_name = spec["exp_name"]
            arch = spec["arch"]
            exp_dir = fold_out_root / exp_name
            ckpt_path = exp_dir / "checkpoints" / "best.pth"
            eval_json = exp_dir / f"categorized_eval_{arch}.json"

            if eval_json.exists() and not args.force:
                print(f"[Fold {k} | {m_key}] Already evaluated ({eval_json}); skipping.")
                continue

            if not args.eval_only:
                if ckpt_path.exists() and not args.force:
                    print(f"[Fold {k} | {m_key}] Found checkpoint ({ckpt_path}); skipping training.")
                else:
                    train_cmd = [
                        py,
                        spec["train_script"],
                        "--config",
                        spec["config"],
                        "--seed",
                        str(args.seed),
                        "--exp-name",
                        exp_name,
                        "--output-root",
                        str(fold_out_root),
                        "--index-csv",
                        str(fold_csv),
                        "--data-root",
                        args.data_root,
                    ]
                    if args.epochs is not None:
                        train_cmd.extend(["--epochs", str(args.epochs)])
                    run_cmd(train_cmd, env=env, cwd=repo_root)

            if not ckpt_path.exists():
                raise FileNotFoundError(f"[Fold {k} | {m_key}] Missing checkpoint: {ckpt_path}")

            eval_cmd = [
                py,
                "src/evaluate_categorized.py",
                "--config",
                spec["config"],
                "--checkpoint",
                str(ckpt_path),
                "--arch",
                arch,
                "--input-csv",
                str(fold_csv),
                "--data-root",
                args.data_root,
                "--exp-name",
                exp_name,
                "--split",
                "test",
                "--output-dir",
                str(exp_dir),
            ]
            run_cmd(eval_cmd, env=env, cwd=repo_root)

        run_cmd(
            [py, "src/compare_results.py", "--exp-root", str(fold_out_root)],
            env=env,
            cwd=repo_root,
        )

    run_cmd(
        [py, "src/aggregate_5fold.py", "--exp-root", str(output_root)],
        env=env,
        cwd=repo_root,
    )


if __name__ == "__main__":
    main()
