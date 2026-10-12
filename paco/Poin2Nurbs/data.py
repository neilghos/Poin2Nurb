import os
import random
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


def ensure_train_val_splits(data_root, val_count=1000, seed=42):
    """
    Deterministically creates non-overlapping train_clean.txt and val.txt from train.txt.
    Guarantees:
      - train_clean INTERSECT val = EMPTY SET
      - val INTERSECT test = EMPTY SET
    """
    train_clean_file = os.path.join(data_root, "train_clean.txt")
    val_file = os.path.join(data_root, "val.txt")

    if os.path.exists(train_clean_file) and os.path.exists(val_file):
        return train_clean_file, val_file

    train_file = os.path.join(data_root, "train.txt")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Source train file not found at {train_file}")

    with open(train_file, "r") as f:
        all_train = [line.strip() for line in f if line.strip()]

    rng = random.Random(seed)
    shuffled = list(all_train)
    rng.shuffle(shuffled)

    n_val = min(val_count, len(all_train))
    val_items = sorted(shuffled[:n_val])
    train_items = sorted(shuffled[n_val:])

    with open(val_file, "w") as f:
        f.write("\n".join(val_items) + "\n")

    with open(train_clean_file, "w") as f:
        f.write("\n".join(train_items) + "\n")

    # Leakage check: verify disjoint sets
    assert len(set(val_items).intersection(set(train_items))) == 0, "Data leakage: train and val overlap!"

    test_file = os.path.join(data_root, "test.txt")
    if os.path.exists(test_file):
        with open(test_file, "r") as f:
            test_items = set(line.strip() for line in f if line.strip())
        assert len(set(val_items).intersection(test_items)) == 0, "Data leakage: val and test overlap!"

    print(f"Created leak-free splits: {len(train_items)} train shapes, {len(val_items)} val shapes (seed {seed})")
    return train_clean_file, val_file


class ABCDataset(Dataset):
    """
    Direct HDF5 ABC Benchmark Dataset matching PaCo's protocol.
    Supports leak-free 'train', 'val', 'train_full', and 'test' splits.
    """
    def __init__(self, data_root=None, split="train", val_count=1000, seed=42,
                 mode="mixed", match_paco_input=False):
        if mode not in ("mixed", "easy", "medium", "hard"):
            raise ValueError(f"Unknown ABC input mode: {mode}")
        if mode != "mixed" and split != "test":
            raise ValueError("Easy, medium, and hard inputs are test-only")
        if data_root is None:
            # Check relative to this script or current working directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                data_root,
                os.path.abspath(os.path.join(script_dir, "../../abc")),
                os.path.abspath(os.path.join(script_dir, "../abc")),
                os.path.abspath("abc"),
                os.path.abspath(os.path.join(script_dir, "../../data/abc")),
                os.path.abspath(os.path.join(script_dir, "../data/abc")),
                os.path.abspath("data/abc"),
            ]
            data_root = next((c for c in candidates if c and os.path.exists(c)), "abc")
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"ABC data root not found: {data_root}")

        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.match_paco_input = match_paco_input

        if split in ["train", "val"]:
            train_clean_file, val_file = ensure_train_val_splits(data_root, val_count=val_count, seed=seed)
            split_file = train_clean_file if split == "train" else val_file
        elif split == "train_full":
            split_file = os.path.join(data_root, "train.txt")
        else:
            split_file = os.path.join(data_root, f"{split}.txt")

        with open(split_file, "r") as f:
            self.model_ids = [line.strip().replace(".npy", "") for line in f if line.strip()]

        # 24 camera viewpoints for train; fixed canonical view 0 for val and test
        self.num_renderings = 24 if split in ["train", "train_full"] else 1
        
        # Check whether data is stored as unified H5 or individual NPY files
        self.use_h5 = mode == "mixed" and os.path.exists(os.path.join(self.data_root, "gt_all.h5"))
        if self.use_h5 and match_paco_input:
            raise ValueError("PaCo input preprocessing requires the original NPY scans")
        self.pc_dir = os.path.join(self.data_root, "pc" if mode == "mixed" else f"eval/pc_{mode}")
        self.gt_dir = os.path.join(self.data_root, "gt")

        # Lazy file handles for H5 if used
        self.pc_file = None
        self.gt_file = None

    def _open_h5(self):
        if self.pc_file is None:
            self.pc_file = h5py.File(os.path.join(self.data_root, "pc_all.h5"), "r")
            self.gt_file = h5py.File(os.path.join(self.data_root, "gt_all.h5"), "r")

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, idx):
        model_id = self.model_ids[idx]

        # Random camera viewpoint for multi-view train; view 0 for val and test
        render_idx = random.randint(0, self.num_renderings - 1) if self.num_renderings > 1 else 0

        if self.use_h5:
            self._open_h5()
            pc_key = f"{model_id}_{render_idx:02d}"
            pc = torch.from_numpy(self.pc_file[pc_key][:]).float()
            gt_raw = torch.from_numpy(self.gt_file[model_id][:]).float()
            gt_xyz = gt_raw[:, :3]
            gt_normals = gt_raw[:, 3:6]
        else:
            pc_path = os.path.join(self.pc_dir, f"{model_id}_{render_idx:02d}.npy")
            pc_raw = np.load(pc_path)
            pts = pc_raw[:, :3]
            n_pts = pts.shape[0]

            # The benchmark path reproduces PaCo's UpSamplePoints indexing on XYZ.
            if self.match_paco_input:
                rng = np.random if self.split in ["train", "train_full"] else np.random.RandomState(idx)
                if self.split not in ["train", "train_full"]:
                    # PaCo's Compose consumes one uniform draw before each
                    # transform (UpSamplePlanes, then UpSamplePoints). Keep
                    # those draws so the subsequent permutation is identical.
                    rng.uniform(0, 1)
                    rng.uniform(0, 1)
                need = 2048 - n_pts
                if need < 0:
                    pts = pts[rng.permutation(2048)]
                else:
                    while n_pts <= need:
                        pts = np.tile(pts, (2, 1))
                        need -= n_pts
                        n_pts *= 2
                    pts = np.concatenate((pts, pts[rng.permutation(need)]), axis=0)
            elif n_pts >= 2048:
                if self.split in ["train", "train_full"]:
                    choice = np.random.choice(n_pts, 2048, replace=False)
                else:
                    rng = np.random.RandomState(idx)
                    choice = rng.choice(n_pts, 2048, replace=False)
                pts = pts[choice]
            else:
                need = 2048 - n_pts
                if self.split in ["train", "train_full"]:
                    choice = np.random.choice(n_pts, need, replace=True)
                else:
                    rng = np.random.RandomState(idx)
                    choice = rng.choice(n_pts, need, replace=True)
                pts = np.concatenate([pts, pts[choice]], axis=0)

            pc = torch.from_numpy(pts).float()

            gt_path = os.path.join(self.gt_dir, f"{model_id}.npy")
            gt_raw = np.load(gt_path)
            gt_xyz = torch.from_numpy(gt_raw[:, :3]).float()
            gt_normals = torch.from_numpy(gt_raw[:, 3:6]).float()

        return {
            "model_id": model_id,
            "pc": pc,                  # (2048, 3)
            "gt": gt_xyz,              # (8192, 3)
            "gt_normals": gt_normals,  # (8192, 3)
        }


if __name__ == "__main__":
    # 1. Instantiate dataset and loader
    dataset = ABCDataset(split="train")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    # 2. Fetch one batch
    batch = next(iter(loader))

    pc_batch = batch["pc"]      # shape: (B, 2048, 3)
    gt_batch = batch["gt"]      # shape: (B, 8192, 3)
    ids = batch["model_id"]

    print("Batch loaded successfully:")
    print("  Input scan (pc) shape :", pc_batch.shape, pc_batch.dtype)
    print("  Ground truth (gt) shape:", gt_batch.shape, gt_batch.dtype)
    print("  Sample IDs in batch   :", ids[:3])
