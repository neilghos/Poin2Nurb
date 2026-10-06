import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableNurbsEvaluator(nn.Module):
    """
    Differentiable B-Spline / NURBS Surface Patch Evaluator.
    Evaluates degree-3 (bicubic) patches with 4x4 control point grids
    across an (R x R) evaluation parameter grid using tensor contraction.
    """
    def __init__(self, num_samples=16):
        super().__init__()
        self.num_samples = num_samples

        # Analytical cubic Bernstein basis on [0, 1]
        t = torch.linspace(0.0, 1.0, num_samples)
        b0 = (1.0 - t) ** 3
        b1 = 3.0 * t * ((1.0 - t) ** 2)
        b2 = 3.0 * (t ** 2) * (1.0 - t)
        b3 = t ** 3
        basis = torch.stack([b0, b1, b2, b3], dim=-1)  # shape: (R, 4)
        self.register_buffer("basis", basis)

    def forward(self, control_points, weights=None):
        """
        Args:
            control_points: Tensor of shape (B, K, 4, 4, 3)
            weights: Optional tensor of shape (B, K, 4, 4)
        Returns:
            surface_points: Tensor of shape (B, K * R * R, 3)
        """
        B, K, _, _, _ = control_points.shape
        M = self.basis  # (R, 4)

        if weights is None:
            # Standard integral B-spline patch
            S = torch.einsum("ru, bkuvc, sv -> bkrsc", M, control_points, M)
        else:
            # Rational NURBS patch: S = sum(w * P) / sum(w)
            P_w = control_points * weights.unsqueeze(-1)
            S_num = torch.einsum("ru, bkuvc, sv -> bkrsc", M, P_w, M)
            S_den = torch.einsum("ru, bkuv, sv -> bkrs", M, weights, M).unsqueeze(-1)
            S = S_num / (S_den + 1e-7)

        # Flatten patches into a single dense surface point cloud
        return S.reshape(B, K * self.num_samples * self.num_samples, 3)


class NurbsDecoder(nn.Module):
    """
    Geometric Worker Decoder:
    Given a global shape embedding, K patch workers predict
    local control point grids P and rational weights W.
    """
    def __init__(self, embed_dim=256, num_patches=16, eval_res=16):
        super().__init__()
        self.num_patches = num_patches
        self.eval_res = eval_res
        self.evaluator = DifferentiableNurbsEvaluator(num_samples=eval_res)

        # Learnable worker query slots: (K, embed_dim)
        self.patch_queries = nn.Parameter(torch.randn(num_patches, embed_dim) * 0.02)

        # Ground-level worker network
        self.worker_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Controller output heads:
        # Control points: 4 x 4 x 3 = 48 values per patch
        self.cp_head = nn.Linear(256, 4 * 4 * 3)
        # Rational weights: 4 x 4 = 16 values per patch (positive)
        self.weight_head = nn.Linear(256, 4 * 4)

    def forward(self, shape_embedding):
        """
        Args:
            shape_embedding: Tensor of shape (B, embed_dim)
        Returns:
            dict containing:
              - 'control_points': (B, K, 4, 4, 3)
              - 'weights': (B, K, 4, 4)
              - 'surface_points': (B, K * R * R, 3)
        """
        B = shape_embedding.shape[0]

        # Expand patch queries across batch: (B, K, embed_dim)
        queries = self.patch_queries.unsqueeze(0).expand(B, -1, -1)

        # Condition workers on global shape embedding: (B, K, 2 * embed_dim)
        emb_expanded = shape_embedding.unsqueeze(1).expand(-1, self.num_patches, -1)
        worker_input = torch.cat([queries, emb_expanded], dim=-1)

        feat = self.worker_mlp(worker_input)

        # Predict controllers
        cp = self.cp_head(feat).reshape(B, self.num_patches, 4, 4, 3)
        w = F.softplus(self.weight_head(feat)).reshape(B, self.num_patches, 4, 4) + 0.1

        # Evaluate continuous parametric surfaces
        surface_points = self.evaluator(cp, w)

        return {
            "control_points": cp,
            "weights": w,
            "surface_points": surface_points,
        }


def chamfer_distance(p1, p2, chunk_size=1024):
    """
    Memory-efficient Symmetric Chamfer Distance with automatic chunking.
    Prevents OOM spikes on large point clouds.
    Args:
        p1: Tensor of shape (B, N, 3)
        p2: Tensor of shape (B, M, 3)
    Returns:
        Scalar loss
    """
    B, N, _ = p1.shape
    M = p2.shape[1]

    if N * M <= 8_000_000:
        dists = torch.cdist(p1, p2)
        return dists.min(dim=2)[0].mean() + dists.min(dim=1)[0].mean()

    min_d1_list = []
    for i in range(0, N, chunk_size):
        p1_chunk = p1[:, i : i + chunk_size, :]
        dists = torch.cdist(p1_chunk, p2)
        min_d1_list.append(dists.min(dim=2)[0])
    min_d1 = torch.cat(min_d1_list, dim=1)

    min_d2_list = []
    for j in range(0, M, chunk_size):
        p2_chunk = p2[:, j : j + chunk_size, :]
        dists = torch.cdist(p2_chunk, p1)
        min_d2_list.append(dists.min(dim=2)[0])
    min_d2 = torch.cat(min_d2_list, dim=1)

    return min_d1.mean() + min_d2.mean()


