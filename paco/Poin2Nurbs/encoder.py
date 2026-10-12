import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedEdgeConv(nn.Module):
    """
    Local Geometric Graph Convolution (EdgeConv).
    For each point p_i, finds k nearest neighbors and computes:
      edge_feat = [p_i, p_j - p_i] -> Conv2d -> MaxPool over neighbors
    Explicitly encodes local 3D tangent planes, surface curvature, and sharp CAD creases.
    """
    def __init__(self, in_channels=3, out_channels=64, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        # x: (B, C, N)
        B, C, N = x.shape
        x_pts = x.transpose(1, 2).contiguous()  # (B, N, C)

        # Pairwise distance matrix: (B, N, N)
        dist = torch.cdist(x_pts, x_pts)
        knn_idx = torch.topk(dist, k=self.k, dim=-1, largest=False)[1]  # (B, N, k)

        # Vectorized gather: gathers only N * k neighbors (32,768 points)
        idx_flat = knn_idx.reshape(B, N * self.k).unsqueeze(-1).expand(-1, -1, C)
        neighbors = torch.gather(x_pts, 1, idx_flat).reshape(B, N, self.k, C)
        pts_expanded = x_pts.unsqueeze(2).expand(-1, -1, self.k, -1)

        # Relative edge vector: [p_i, p_j - p_i]
        edge_feat = torch.cat([pts_expanded, neighbors - pts_expanded], dim=-1)
        edge_feat = edge_feat.permute(0, 3, 1, 2).contiguous()  # (B, 2*C, N, k)

        feat = self.mlp(edge_feat)
        out = feat.max(dim=-1)[0]  # (B, out_channels, N)
        return out


class HierarchicalPointEncoder(nn.Module):
    """
    Zonal Hierarchical Point Cloud Encoder.
    Processes input partial scans (B, N, 3) across 3 tiers:
      1. Point-Level Multi-Scale Convs:
         - Level 1: Curvature & Crease-Aware EdgeConv (Local relative displacements, 64-dim)
         - Level 2: Regional features (128-dim)
         - Level 3: Global macro features (256-dim)
      2. Top Tier: Global Macro Context (B, out_dim) via multi-scale pooling
      3. Middle Tier: K Zonal Encoders (B, K, out_dim) via spatial distance-biased cross-attention
         anchored around learnable 3D zone centers
    """
    def __init__(self, in_channels=3, out_dim=128, num_patches=32, k_neighbors=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.num_patches = num_patches

        # Level 1: Curvature & Crease-Aware EdgeConv (3 -> 64)
        self.level1 = OptimizedEdgeConv(in_channels=in_channels, out_channels=64, k=k_neighbors)

        # Level 2: Regional features (64 -> 128)
        self.level2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Level 3: Global macro features (128 -> 256)
        self.level3 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Multi-scale global hierarchical fusion: (64 + 128 + 256 = 448 -> out_dim)
        feat_dim = 64 + 128 + 256
        self.global_fusion = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, out_dim),
        )

        # Initialize K zone anchors spread across the canonical CAD volume [-0.35, 0.35]^3 via spherical Fibonacci lattice
        indices = torch.arange(0, num_patches, dtype=torch.float) + 0.5
        phi = torch.arccos(1.0 - 2.0 * indices / num_patches)
        theta = math.pi * (1.0 + 5.0 ** 0.5) * indices
        r = 0.35
        x_a = r * torch.sin(phi) * torch.cos(theta)
        y_a = r * torch.sin(phi) * torch.sin(theta)
        z_a = r * torch.cos(phi)
        anchors = torch.stack([x_a, y_a, z_a], dim=-1)
        self.zone_anchors = nn.Parameter(anchors)

        # Learnable Zone Query Tokens (K, out_dim)
        self.zone_queries = nn.Parameter(torch.randn(num_patches, out_dim) * 0.02)

        # Condition Zonal Queries on Global Macro Context (Top-down guidance from Global box to Zonal boxes)
        self.zonal_query_mlp = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        # Distance-biased spatial cross-attention projections
        self.to_q = nn.Linear(out_dim, out_dim)
        self.to_k = nn.Linear(feat_dim, out_dim)
        self.to_v = nn.Linear(feat_dim, out_dim)
        self.scale = 1.0 / math.sqrt(out_dim)

        # Learnable spatial distance penalty parameter gamma > 0
        self.raw_gamma = nn.Parameter(torch.tensor([5.0]))

        # Final Zonal feature fusion
        self.zonal_fusion = nn.Sequential(
            nn.Linear(out_dim * 2 + 3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, N, 3) containing partial point cloud coordinates
        Returns:
            global_embedding: (B, out_dim)
            zonal_embeddings: (B, K, out_dim)
            zone_anchors: (B, K, 3)
        """
        B, N, C = x.shape
        x_trans = x.transpose(1, 2)

        # Hierarchical forward representations
        f1 = self.level1(x_trans)  # (B, 64, N)
        f2 = self.level2(f1)       # (B, 128, N)
        f3 = self.level3(f2)       # (B, 512, N)

        # 1. Global Macro Context (Top Tier)
        p1 = torch.max(f1, dim=2)[0]  # (B, 64)
        p2 = torch.max(f2, dim=2)[0]  # (B, 128)
        p3 = torch.max(f3, dim=2)[0]  # (B, 512)
        global_feat = torch.cat([p1, p2, p3], dim=-1)         # (B, 704)
        global_embedding = self.global_fusion(global_feat)    # (B, out_dim)

        # 2. Zonal Spatial Cross-Attention (Middle Tier)
        point_feats = torch.cat([f1, f2, f3], dim=1).transpose(1, 2)  # (B, N, 704)

        # Condition zone queries on global context
        K = self.num_patches
        z_queries = self.zone_queries.unsqueeze(0).expand(B, -1, -1)
        g_expand = global_embedding.unsqueeze(1).expand(-1, K, -1)
        cond_queries = self.zonal_query_mlp(torch.cat([z_queries, g_expand], dim=-1))  # (B, K, out_dim)

        # Pairwise distance squared between zone anchors and partial scan points: (B, K, N)
        anchors_expanded = self.zone_anchors.unsqueeze(0).expand(B, -1, -1)  # (B, K, 3)
        dist_sq = torch.cdist(anchors_expanded, x) ** 2

        Q = self.to_q(cond_queries)       # (B, K, out_dim)
        Key = self.to_k(point_feats)      # (B, N, out_dim)
        Val = self.to_v(point_feats)      # (B, N, out_dim)

        gamma = F.softplus(self.raw_gamma)
        attn_logits = torch.bmm(Q, Key.transpose(1, 2)) * self.scale - gamma * dist_sq  # (B, K, N)
        attn_weights = F.softmax(attn_logits, dim=-1)                                    # (B, K, N)
        zonal_context = torch.bmm(attn_weights, Val)                                     # (B, K, out_dim)

        # Fuse zonal context, query, and anchor 3D position
        zonal_input = torch.cat([zonal_context, cond_queries, anchors_expanded], dim=-1)
        zonal_embeddings = self.zonal_fusion(zonal_input)  # (B, K, out_dim)

        return global_embedding, zonal_embeddings, anchors_expanded


if __name__ == "__main__":
    print("=== HierarchicalPointEncoder Sanity Test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    B, N = 4, 2048
    encoder = HierarchicalPointEncoder(in_channels=3, out_dim=256).to(device)
    dummy_pc = torch.randn(B, N, 3, device=device, requires_grad=True)

    g_emb, z_embs, anchors = encoder(dummy_pc)
    print(f"Input point cloud shape  : {dummy_pc.shape}")
    print(f"Global embedding shape   : {g_emb.shape}")
    print(f"Zonal embedding shape    : {z_embs.shape}")
    print(f"Zone anchors shape       : {anchors.shape}")

    loss = g_emb.sum() + z_embs.sum()
    loss.backward()

    print(f"Input points grad norm   : {dummy_pc.grad.norm().item():.4f}")
    print(f"Level 1 EdgeConv mlp norm: {encoder.level1.mlp[0].weight.grad.norm().item():.4f}")
    print(f"Level 3 conv2 grad norm  : {encoder.level3[3].weight.grad.norm().item():.4f}")
    assert dummy_pc.grad is not None and dummy_pc.grad.norm().item() > 0, "No gradients reached input points!"
    print("\nEncoder test: PASSED (healthy multi-scale gradients).")
