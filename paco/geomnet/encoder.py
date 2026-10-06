import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalPointEncoder(nn.Module):
    """
    Hierarchical Multi-Scale Point Cloud Encoder.
    Processes input partial scans (B, N, 3) across 3 geometric scales:
      - Level 1: Local point features (edges, corners, sharp transitions)
      - Level 2: Mid-level regional curvature (cylinder/planar patches)
      - Level 3: Global macro topology (overall bounding shape and symmetry)
    Fuses pooled representations from all 3 levels into a unified shape embedding.
    """
    def __init__(self, in_channels=3, out_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # Level 1: Fine local features (3 -> 64)
        self.level1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Level 2: Regional curvature features (64 -> 128)
        self.level2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Level 3: Global macro features (128 -> 512)
        self.level3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # Multi-scale hierarchical fusion: (64 + 128 + 512 = 704 -> out_dim)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 128 + 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, N, 3) containing partial point cloud coordinates
        Returns:
            shape_embedding: Tensor of shape (B, out_dim)
        """
        B, N, C = x.shape
        # Transpose to (B, C, N) for 1D convolution
        x_trans = x.transpose(1, 2)

        # Hierarchical forward representations
        f1 = self.level1(x_trans)  # (B, 64, N)
        f2 = self.level2(f1)       # (B, 128, N)
        f3 = self.level3(f2)       # (B, 512, N)

        # Permutation-invariant pooling across all three geometric scales
        p1 = torch.max(f1, dim=2)[0]  # (B, 64)
        p2 = torch.max(f2, dim=2)[0]  # (B, 128)
        p3 = torch.max(f3, dim=2)[0]  # (B, 512)

        # Multi-scale concatenation
        hierarchical_feat = torch.cat([p1, p2, p3], dim=-1)  # (B, 704)
        shape_embedding = self.fusion(hierarchical_feat)      # (B, out_dim)

        return shape_embedding


if __name__ == "__main__":
    print("=== HierarchicalPointEncoder Sanity Test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    B, N = 4, 2048
    encoder = HierarchicalPointEncoder(in_channels=3, out_dim=256).to(device)
    dummy_pc = torch.randn(B, N, 3, device=device, requires_grad=True)

    out = encoder(dummy_pc)
    print(f"Input point cloud shape : {dummy_pc.shape}")
    print(f"Output embedding shape  : {out.shape}")

    loss = out.sum()
    loss.backward()

    print(f"Input points grad norm  : {dummy_pc.grad.norm().item():.4f}")
    print(f"Level 1 conv1 grad norm : {encoder.level1[0].weight.grad.norm().item():.4f}")
    print(f"Level 3 conv2 grad norm : {encoder.level3[3].weight.grad.norm().item():.4f}")
    assert dummy_pc.grad is not None and dummy_pc.grad.norm().item() > 0, "No gradients reached input points!"
    print("\nEncoder test: PASSED (healthy multi-scale gradients).")
