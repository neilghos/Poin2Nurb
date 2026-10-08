import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableNurbsEvaluator(nn.Module):
    """
    Differentiable B-Spline / NURBS Surface Patch Evaluator.
    Evaluates degree-d patches with (d+1) x (d+1) control point grids
    across an (R x R) evaluation parameter grid using tensor contraction.
    """
    def __init__(self, num_samples=16, degree=3):
        super().__init__()
        self.num_samples = num_samples
        self.degree = degree
        G = degree + 1

        # Analytical Bernstein basis on [0, 1] and analytical first derivatives
        t = torch.linspace(0.0, 1.0, num_samples, dtype=torch.float32)
        basis_list = []
        dbasis_list = []

        for i in range(G):
            coeff = float(math.comb(degree, i))
            term_t = torch.where((t == 0) & (i == 0), torch.ones_like(t), t ** i)
            term_1_t = torch.where((t == 1) & (degree - i == 0), torch.ones_like(t), (1.0 - t) ** (degree - i))
            b = coeff * term_t * term_1_t
            basis_list.append(b)

            if i == 0:
                db = -float(degree) * ((1.0 - t) ** (degree - 1))
            elif i == degree:
                db = float(degree) * (t ** (degree - 1))
            else:
                t1 = i * torch.where((t == 0) & (i - 1 == 0), torch.ones_like(t), t ** (i - 1)) * ((1.0 - t) ** (degree - i))
                t2 = (degree - i) * (t ** i) * torch.where((t == 1) & (degree - i - 1 == 0), torch.ones_like(t), (1.0 - t) ** (degree - i - 1))
                db = coeff * (t1 - t2)
            dbasis_list.append(db)

        basis = torch.stack(basis_list, dim=-1)   # shape: (R, G)
        self.register_buffer("basis", basis, persistent=False)

        d_basis = torch.stack(dbasis_list, dim=-1) # shape: (R, G)
        self.register_buffer("d_basis", d_basis, persistent=False)

    def forward(self, control_points, weights=None, return_normals=True):
        """
        Args:
            control_points: Tensor of shape (B, K, 4, 4, 3)
            weights: Optional tensor of shape (B, K, 4, 4)
            return_normals: If True, also computes exact analytical surface normals
        Returns:
            surface_points: (B, K * R * R, 3)
            surface_normals (optional): (B, K * R * R, 3)
        """
        B, K, _, _, _ = control_points.shape
        M = self.basis      # (R, 4)
        dM = self.d_basis   # (R, 4)

        if weights is None:
            # Standard integral B-spline patch
            S = torch.einsum("ru, bkuvc, sv -> bkrsc", M, control_points, M)
            surface_points = S.reshape(B, K * self.num_samples * self.num_samples, 3)

            if return_normals:
                Tu = torch.einsum("ru, bkuvc, sv -> bkrsc", dM, control_points, M)
                Tv = torch.einsum("ru, bkuvc, sv -> bkrsc", M, control_points, dM)
                n_unnorm = torch.cross(Tu, Tv, dim=-1)
                normals = F.normalize(n_unnorm, dim=-1, eps=1e-7)
                surface_normals = normals.reshape(B, K * self.num_samples * self.num_samples, 3)
                return surface_points, surface_normals
            return surface_points
        else:
            # Rational NURBS patch: S = sum(w * P) / sum(w)
            P_w = control_points * weights.unsqueeze(-1)
            S_num = torch.einsum("ru, bkuvc, sv -> bkrsc", M, P_w, M)
            S_den = torch.einsum("ru, bkuv, sv -> bkrs", M, weights, M).unsqueeze(-1)
            S = S_num / (S_den + 1e-7)
            surface_points = S.reshape(B, K * self.num_samples * self.num_samples, 3)

            if return_normals:
                # Analytical partial derivatives via quotient rule
                d_Nu = torch.einsum("ru, bkuvc, sv -> bkrsc", dM, P_w, M)
                d_Du = torch.einsum("ru, bkuv, sv -> bkrs", dM, weights, M).unsqueeze(-1)

                d_Nv = torch.einsum("ru, bkuvc, sv -> bkrsc", M, P_w, dM)
                d_Dv = torch.einsum("ru, bkuv, sv -> bkrs", M, weights, dM).unsqueeze(-1)

                Tu = d_Nu * S_den - S_num * d_Du
                Tv = d_Nv * S_den - S_num * d_Dv

                n_unnorm = torch.cross(Tu, Tv, dim=-1)
                normals = F.normalize(n_unnorm, dim=-1, eps=1e-7)
                surface_normals = normals.reshape(B, K * self.num_samples * self.num_samples, 3)
                return surface_points, surface_normals
            return surface_points


class ResidualMLPBlock(nn.Module):
    """
    Residual MLP Block with LayerNorm for coordinate regression.
    Preserves gradient flow and stabilizes continuous spline outputs.
    """
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        res = x
        out = self.act(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return self.act(res + out)


class NurbsDecoder(nn.Module):
    """
    Geometric Worker Decoder:
    Given a global shape embedding, K zonal embeddings, and 3D zone anchors,
    K patch workers predict local control point grids P and rational weights W.
    """
    def __init__(self, embed_dim=128, num_patches=32, eval_res=16, patch_degree=3):
        super().__init__()
        self.num_patches = num_patches
        self.eval_res = eval_res
        self.patch_degree = patch_degree
        self.grid_size = patch_degree + 1
        G = self.grid_size
        self.evaluator = DifferentiableNurbsEvaluator(num_samples=eval_res, degree=patch_degree)

        # Learnable worker query slots: (K, embed_dim)
        self.patch_queries = nn.Parameter(torch.randn(num_patches, embed_dim) * 0.02)

        # Ground-level worker network: takes [patch_query, zonal_feature, global_embedding, zone_anchor]
        in_dim = embed_dim * 3 + 3
        hidden_dim = max(256, embed_dim * 2)

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res1 = ResidualMLPBlock(hidden_dim)
        self.res2 = ResidualMLPBlock(hidden_dim)

        # Controller output heads:
        # Control points: G x G x 3 values per patch
        self.cp_head = nn.Linear(hidden_dim, G * G * 3)
        # Rational weights: G x G values per patch (positive)
        self.weight_head = nn.Linear(hidden_dim, G * G)

    def forward(self, shape_embedding, zonal_embeddings=None, zone_anchors=None, return_normals=True):
        """
        Args:
            shape_embedding: Tensor of shape (B, embed_dim) from Global Macro Encoder
            zonal_embeddings: Optional tensor of shape (B, K, embed_dim) from K Zonal Encoders
            zone_anchors: Optional tensor of shape (B, K, 3) 3D zone centers
            return_normals: If True, evaluates analytical surface normals
        Returns:
            dict containing:
              - 'control_points': (B, K, G, G, 3)
              - 'weights': (B, K, G, G)
              - 'surface_points': (B, K * R * R, 3)
              - 'surface_normals' (optional): (B, K * R * R, 3)
        """
        B = shape_embedding.shape[0]
        G = self.grid_size

        # Expand patch queries across batch: (B, K, embed_dim)
        queries = self.patch_queries.unsqueeze(0).expand(B, -1, -1)

        # Condition workers on global shape embedding: (B, K, embed_dim)
        emb_expanded = shape_embedding.unsqueeze(1).expand(-1, self.num_patches, -1)

        z_feat = zonal_embeddings if zonal_embeddings is not None else emb_expanded
        anchors = zone_anchors if zone_anchors is not None else torch.zeros(B, self.num_patches, 3, device=queries.device, dtype=queries.dtype)

        worker_input = torch.cat([queries, z_feat, emb_expanded, anchors], dim=-1)

        feat = self.in_proj(worker_input)
        feat = self.res1(feat)
        feat = self.res2(feat)

        # Predict controllers
        cp_delta = self.cp_head(feat).reshape(B, self.num_patches, G, G, 3)
        if zone_anchors is not None:
            # Anchor patches around their physical 3D zone centers
            cp = cp_delta + zone_anchors.unsqueeze(2).unsqueeze(3)
        else:
            cp = cp_delta

        w = F.softplus(self.weight_head(feat)).reshape(B, self.num_patches, G, G) + 0.1

        # Evaluate continuous parametric surfaces
        if return_normals:
            surface_points, surface_normals = self.evaluator(cp, w, return_normals=True)
            return {
                "control_points": cp,
                "weights": w,
                "surface_points": surface_points,
                "surface_normals": surface_normals,
                "zone_anchors": zone_anchors,
            }
        else:
            surface_points = self.evaluator(cp, w, return_normals=False)
            return {
                "control_points": cp,
                "weights": w,
                "surface_points": surface_points,
                "zone_anchors": zone_anchors,
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

    if N * M <= 70_000_000:
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


def control_point_laplacian_loss(cp):
    """
    Penalizes second-order discrete differences across the 4x4 control grid.
    Acts as a 2D membrane stiffness prior, preventing pillowing and control point flaring.
    cp: shape (B, K, 4, 4, 3)
    """
    diff_u = cp[:, :, 2:, :, :] - 2.0 * cp[:, :, 1:-1, :, :] + cp[:, :, :-2, :, :]
    diff_v = cp[:, :, :, 2:, :] - 2.0 * cp[:, :, :, 1:-1, :] + cp[:, :, :, :-2, :]
    return diff_u.pow(2).mean() + diff_v.pow(2).mean()


def chamfer_and_normal_loss(
    p_pred,
    n_pred,
    p_gt,
    n_gt,
    lambda_normal=0.1,
    cp=None,
    lambda_laplacian=0.01,
    topk_ratio=0.05,
    lambda_topk=0.5,
):
    """
    Joint Chamfer Distance, Top-k Outlier Loss, Analytical Normal Alignment,
    and 2D Control Grid Laplacian Regularizer.
    
    Args:
        p_pred: Predicted surface points (B, N, 3)
        n_pred: Analytical unit surface normals (B, N, 3)
        p_gt: Ground truth surface points (B, M, 3)
        n_gt: Ground truth surface normals (B, M, 3)
        lambda_normal: Weight for symmetric normal alignment loss
        cp: Control points (B, K, 4, 4, 3) for Laplacian stiffness regularization
        lambda_laplacian: Weight for 2D control point Laplacian penalty
        topk_ratio: Fraction of worst outlier points to penalize in Direction 1 (Pred->GT)
        lambda_topk: Weight for top-k outlier penalty
    Returns:
        total_loss, cd_mean, normal_loss, nc_metric
    """
    dists = torch.cdist(p_pred, p_gt)  # (B, N, M)
    min_d1, idx1 = dists.min(dim=2)    # (B, N) - Pred -> GT
    min_d2, idx2 = dists.min(dim=1)    # (B, M) - GT -> Pred

    cd_mean = min_d1.mean() + min_d2.mean()

    # Forward normal alignment: for each p_pred point, compare its normal with matched gt normal
    n_gt_matched = torch.gather(n_gt, 1, idx1.unsqueeze(-1).expand(-1, -1, 3))
    cos1 = torch.abs((n_pred * n_gt_matched).sum(dim=-1))
    normal_loss1 = (1.0 - cos1).mean()

    # Backward normal alignment: for each p_gt point, compare its normal with matched pred normal
    n_pred_matched = torch.gather(n_pred, 1, idx2.unsqueeze(-1).expand(-1, -1, 3))
    cos2 = torch.abs((n_gt * n_pred_matched).sum(dim=-1))
    normal_loss2 = (1.0 - cos2).mean()

    normal_loss = 0.5 * (normal_loss1 + normal_loss2)
    nc_metric = 0.5 * (cos1.mean() + cos2.mean())

    total_loss = cd_mean + lambda_normal * normal_loss

    # Top-k hard outlier penalty on Direction 1 (Pred -> GT) to crush boundary whiskers
    if topk_ratio > 0.0 and lambda_topk > 0.0:
        k = max(1, int(topk_ratio * min_d1.shape[1]))
        topk_d1 = torch.topk(min_d1, k=k, dim=1)[0].mean()
        total_loss = total_loss + lambda_topk * topk_d1

    # 2D Control grid Laplacian stiffness to kill pillowing
    if cp is not None and lambda_laplacian > 0.0:
        lap_loss = control_point_laplacian_loss(cp)
        total_loss = total_loss + lambda_laplacian * lap_loss

    return total_loss, cd_mean, normal_loss, nc_metric




