import os
import glob
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
import pandas as pd
import numpy as np
from collections import defaultdict


class SequenceDataset(Dataset):
    """Generic RGB sequence dataset for video-based person Re-ID."""

    def __init__(
        self,
        index_csv,
        root_dir,
        transform=None,
        sequence_len=16,
        split="train",
        is_train=False,
        return_metadata=False,
    ):
        self.df = pd.read_csv(index_csv)

        rename_dict = {}
        if "person_global_id" in self.df.columns and "person_id" not in self.df.columns:
            rename_dict["person_global_id"] = "person_id"
        if "rgb_dir" in self.df.columns and "frames_dir" not in self.df.columns:
            rename_dict["rgb_dir"] = "frames_dir"
        if "scene_id" in self.df.columns and "scene" not in self.df.columns:
            rename_dict["scene_id"] = "scene"
        if rename_dict:
            self.df = self.df.rename(columns=rename_dict)

        if split is not None:
            if "split" in self.df.columns:
                self.df = self.df[self.df["split"] == split].reset_index(drop=True)
            else:
                print(f"Warning: 'split' column not found in {index_csv}. Using all data.")

        self.df["frames_dir"] = self.df["frames_dir"].astype(str).str.replace("\\", "/", regex=False)

        self.root_dir = root_dir
        self.transform = transform
        self.sequence_len = sequence_len
        self.is_train = is_train
        self.return_metadata = return_metadata

        self.pids = sorted(self.df["person_id"].unique())
        self.pid2label = {pid: i for i, pid in enumerate(self.pids)}
        self.num_classes = len(self.pids)

    def __len__(self):
        return len(self.df)

    def _sample_indices(self, num_frames: int):
        if num_frames == self.sequence_len:
            return np.arange(num_frames, dtype=int)
        if self.is_train and num_frames > self.sequence_len:
            # Stratified random temporal sampling across uniform segments
            edges = np.linspace(0, num_frames, self.sequence_len + 1, dtype=int)
            indices = []
            for i in range(self.sequence_len):
                lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
                indices.append(np.random.randint(lo, hi))
            return np.asarray(indices, dtype=int)
        return np.linspace(0, num_frames - 1, self.sequence_len).round().astype(int)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames_dir = row["frames_dir"]
        pid = int(row["person_id"])
        label = self.pid2label[pid]

        full_frames_dir = os.path.join(self.root_dir, frames_dir)
        frame_paths = sorted(glob.glob(os.path.join(full_frames_dir, "*.jpg")))

        num_frames = len(frame_paths)
        if num_frames == 0:
            raise ValueError(f"No frames found in {full_frames_dir}")

        indices = self._sample_indices(num_frames)
        frame_paths = [frame_paths[i] for i in indices]

        frames = []
        for p in frame_paths:
            with Image.open(p) as im:
                img = im.convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames)

        if self.return_metadata:
            metadata = {
                "sequence_id": str(row.get("sequence_id", "")),
                "person_id": pid,
                "cam_id": str(row.get("cam_id", "")),
                "scene": str(row.get("scene", "")),
                "illum_label": str(row.get("illum_label", "unknown")),
                "occlusion_label": str(row.get("occlusion_label", "unknown")),
            }
            return frames, label, metadata

        return frames, label


class RandomIdentitySampler(Sampler):
    """
    Randomly sample P identities, then for each identity sample K instances.
    Supports max_instances_per_pid to prevent high-count identities from dominating each epoch.
    """

    def __init__(self, data_source, batch_size, num_instances, max_instances_per_pid=32):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.max_instances_per_pid = max_instances_per_pid
        self.index_dic = defaultdict(list)

        for index, row in data_source.df.iterrows():
            pid = row["person_id"]
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())

        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if self.max_instances_per_pid is not None:
                num = min(num, self.max_instances_per_pid)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = list(self.index_dic[pid])
            np.random.shuffle(idxs)
            if self.max_instances_per_pid is not None and len(idxs) > self.max_instances_per_pid:
                idxs = idxs[: self.max_instances_per_pid]
            if len(idxs) < self.num_instances:
                idxs = list(np.random.choice(idxs, size=self.num_instances, replace=True))
            batch_idxs_dict[pid] = list(idxs)

        avai_pids = list(self.index_dic.keys())
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                for _ in range(self.num_instances):
                    final_idxs.append(batch_idxs_dict[pid].pop(0))
                if len(batch_idxs_dict[pid]) < self.num_instances:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length
