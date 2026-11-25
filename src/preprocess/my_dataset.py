import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np


class MyDataset(Dataset):
    def __init__(self, index_csv, root_dir, transform=None, sequence_len=16, split="test"):
        """
        Dataset for categorized Re-ID evaluation with metadata support.
        """
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

        self.pids = sorted(self.df["person_id"].unique())
        self.pid2label = {pid: i for i, pid in enumerate(self.pids)}
        self.num_classes = len(self.pids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames_dir = row["frames_dir"]
        pid = int(row["person_id"])
        label = self.pid2label[pid]

        metadata = {
            "sequence_id": str(row.get("sequence_id", "")),
            "person_id": pid,
            "cam_id": str(row.get("cam_id", "")),
            "scene": str(row.get("scene", "")),
            "illum_label": str(row.get("illum_label", "unknown")),
            "occlusion_label": str(row.get("occlusion_label", "unknown")),
        }

        full_frames_dir = os.path.join(self.root_dir, frames_dir)
        frame_paths = sorted(glob.glob(os.path.join(full_frames_dir, "*.jpg")))

        num_frames = len(frame_paths)
        if num_frames == 0:
            raise ValueError(f"No frames found in {full_frames_dir}")

        if num_frames != self.sequence_len:
            indices = np.linspace(0, num_frames - 1, self.sequence_len).round().astype(int)
            frame_paths = [frame_paths[i] for i in indices]

        frames = []
        for p in frame_paths:
            with Image.open(p) as im:
                img = im.convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames)

        return frames, label, metadata
