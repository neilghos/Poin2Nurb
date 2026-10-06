import os
import random
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class ABCDataset(Dataset):
    """
    Direct HDF5 ABC Benchmark Dataset matching PaCo's protocol.
    """
    def __init__(self, data_root=None, split="train"):
        if data_root is None or not os.path.exists(data_root):
            # Check relative to this script or current working directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                data_root,
                os.path.abspath(os.path.join(script_dir, "../../data/abc")),
                os.path.abspath(os.path.join(script_dir, "../data/abc")),
                os.path.abspath("data/abc"),
                "/data/Nurbs Geometric Nueral Net with self training data generation/data/abc",
            ]
            data_root = next((c for c in candidates if c and os.path.exists(c)), "data/abc")

        self.data_root = data_root
        self.split = split
        split_file = os.path.join(data_root, f"{split}.txt")
        
        with open(split_file, "r") as f:
            self.model_ids = [line.strip().replace(".npy", "") for line in f if line.strip()]

        # 24 renderings for train, 1 fixed rendering for test (PaCo standard)
        self.num_renderings = 24 if split == "train" else 1
        
        # Lazy file handles for safe multi-worker DataLoader support
        self.pc_file = None
        self.gt_file = None

    def _open_h5(self):
        if self.pc_file is None:
            self.pc_file = h5py.File(os.path.join(self.data_root, "pc_all.h5"), "r")
            self.gt_file = h5py.File(os.path.join(self.data_root, "gt_all.h5"), "r")

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, idx):
        self._open_h5()
        model_id = self.model_ids[idx]

        # Random camera viewpoint for train; view 0 for test
        render_idx = random.randint(0, self.num_renderings - 1) if self.split == "train" else 0
        pc_key = f"{model_id}_{render_idx:02d}"

        # Incomplete scan: shape (2048, 3)
        pc = torch.from_numpy(self.pc_file[pc_key][:]).float()

        gt_raw = torch.from_numpy(self.gt_file[model_id][:]).float()
        gt_xyz = gt_raw[:, :3]
        gt_normals = gt_raw[:, 3:6]

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
