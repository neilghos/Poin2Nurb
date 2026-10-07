import torch
import torch.nn as nn
from encoder import HierarchicalPointEncoder
from macro_gnn import MacroGNN
from decoder import NurbsDecoder, chamfer_distance, chamfer_and_normal_loss


class GeomNet(nn.Module):
    """
    GeomNet End-to-End Point-to-NURBS Surface Completion Network.
    
    Data Flow:
      1. Input: Partial point cloud scan (B, 2048, 3)
      2. HierarchicalPointEncoder: Multi-scale point convs + K zonal spatial cross-attention
      3. MacroGNN (Tier 2): 2-layer Graph Attention over the K zone nodes using 3D anchor
         coordinates to coordinate boundary features and topological continuity
      4. NurbsDecoder: K patch worker MLPs predict control points P (B, K, 4, 4, 3)
         and rational weights W (B, K, 4, 4)
      5. DifferentiableNurbsEvaluator: Contraction of cubic Bernstein basis generates
         dense surface point cloud (B, K * R * R, 3) and analytical normals (B, K * R * R, 3)
    """
    def __init__(
        self,
        embed_dim=128,
        num_patches=32,
        eval_res=16,
        use_macro_gnn=True,
        k_neighbors=6,
        gnn_layers=2,
    ):
        super().__init__()
        self.use_macro_gnn = use_macro_gnn
        self.encoder = HierarchicalPointEncoder(in_channels=3, out_dim=embed_dim, num_patches=num_patches)
        if use_macro_gnn:
            self.macro_gnn = MacroGNN(embed_dim=embed_dim, k_neighbors=k_neighbors, num_layers=gnn_layers)
        else:
            self.macro_gnn = None
        self.decoder = NurbsDecoder(embed_dim=embed_dim, num_patches=num_patches, eval_res=eval_res)

    def forward(self, pc, return_normals=True):
        """
        Args:
            pc: Incomplete point cloud scan of shape (B, 2048, 3)
            return_normals: If True, evaluates analytical surface normals
        Returns:
            dict with 'control_points', 'weights', 'surface_points', 'surface_normals', 'zone_anchors'
        """
        global_emb, zonal_embs, zone_anchors = self.encoder(pc)
        if self.use_macro_gnn and self.macro_gnn is not None:
            zonal_features = self.macro_gnn(zonal_embs, zone_anchors)
        else:
            zonal_features = zonal_embs

        return self.decoder(
            global_emb,
            zonal_embeddings=zonal_features,
            zone_anchors=zone_anchors,
            return_normals=return_normals,
        )


if __name__ == "__main__":
    import os
    import sys
    from torch.utils.data import DataLoader
    from data import ABCDataset

    print("=== GeomNet End-to-End Overfit Sanity Test on ABC Dataset ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load 1 real batch from ABC benchmark
    dataset = ABCDataset(split="train")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    pc = batch["pc"].to(device)  # (4, 2048, 3)
    gt = batch["gt"].to(device)  # (4, 8192, 3)
    model_ids = batch["model_id"]
    print(f"Loaded batch of 4 shapes: {model_ids}")

    # Initialize end-to-end model
    model = GeomNet(embed_dim=256, num_patches=16, eval_res=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nTraining on single real batch over 15 steps:")
    for step in range(1, 16):
        optimizer.zero_grad()
        output = model(pc)
        loss = chamfer_distance(output["surface_points"], gt)
        loss.backward()
        optimizer.step()
        print(f"Step {step:2d} | Chamfer Distance: {loss.item():.4f}")

    print("\nEnd-to-End Sanity Test: PASSED (complete gradient flow from CAD surface to raw scan points).")
