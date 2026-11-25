import argparse
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test.")
    parser.add_argument("--index-csv", required=True, help="Path to input sequences CSV")
    parser.add_argument("--output-csv", required=True, help="Path to save the new CSV with splits")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of IDs for training")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of IDs for validation")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Ratio of IDs for testing")
    parser.add_argument("--use-standard-split", action="store_true", help="Only split train into train/val")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["open_set", "closed_set"],
        default="open_set",
        help="open_set splits by disjoint person_id; closed_set splits sequences within each person_id.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Normalize ratios if they don't sum to 1
    total = args.train_ratio + args.val_ratio + args.test_ratio
    train_r = args.train_ratio / total
    val_r = args.val_ratio / total
    
    random.seed(args.seed)
    
    df = pd.read_csv(args.index_csv)
    
    # Standardize column names
    rename_dict = {}
    if 'person_global_id' in df.columns:
        rename_dict['person_global_id'] = 'person_id'
    if 'rgb_dir' in df.columns:
        rename_dict['rgb_dir'] = 'frames_dir'
    if rename_dict:
        df = df.rename(columns=rename_dict)

    pids = sorted(df['person_id'].unique().tolist())
    random.shuffle(pids)
    
    print(f"Total IDs: {len(pids)} | Mode: {args.mode}")

    if args.mode == "open_set":
        n_total = len(pids)
        n_train = max(1, int(round(n_total * train_r)))
        n_val = max(1, int(round(n_total * val_r))) if n_total >= 3 else 0
        if n_train + n_val >= n_total and n_total >= 3:
            n_train = n_total - max(1, n_val) - 1
        train_pids = set(pids[:n_train])
        val_pids = set(pids[n_train : n_train + n_val])
        test_pids = set(pids[n_train + n_val :])

        assert train_pids.isdisjoint(val_pids)
        assert train_pids.isdisjoint(test_pids)
        assert val_pids.isdisjoint(test_pids)

        df["split"] = df["person_id"].map(
            lambda p: "train" if p in train_pids else ("val" if p in val_pids else "test")
        )
    else:
        pbar = tqdm(pids, desc="Splitting per ID (closed_set)")
        for pid in pbar:
            pid_df = df[df['person_id'] == pid]
            seq_ids = pid_df['sequence_id'].tolist()
            random.shuffle(seq_ids)
            n = len(seq_ids)

            n_train = max(1, int(n * train_r)) if n > 1 else 1
            n_val = max(0, int(n * val_r))

            if n == 1:
                train_s = seq_ids
                val_s = []
                test_s = []
            elif n == 2:
                train_s = [seq_ids[0]]
                test_s = [seq_ids[1]]
                val_s = []
            else:
                train_s = seq_ids[:n_train]
                val_s = seq_ids[n_train:n_train+n_val]
                test_s = seq_ids[n_train+n_val:]
                if not test_s and n > 2:
                    test_s = [train_s.pop()]

            df.loc[df['sequence_id'].isin(train_s), 'split'] = 'train'
            df.loc[df['sequence_id'].isin(val_s), 'split'] = 'val'
            df.loc[df['sequence_id'].isin(test_s), 'split'] = 'test'

    df.to_csv(args.output_csv, index=False)
    print(f"Saved split metadata to {args.output_csv}")

if __name__ == "__main__":
    main()
