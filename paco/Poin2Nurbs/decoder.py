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

    def forward(self, control_points, weights=None, return_normals=True, return_tangents=False):
        """
        Args:
            control_points: Tensor of shape (B, K, G, G, 3)
            weights: Optional tensor of shape (B, K, G, G)
            return_normals: If True, also computes exact analytical surface normals
            return_tangents: If True, returns (S, Tu, Tv) with unflattened shape (B, K, R, R, 3)
        Returns:
            surface_points: (B, K * R * R, 3)
            surface_normals (optional): (B, K * R * R, 3)
            or (S, Tu, Tv) if return_tangents=True
        """
        B, K, _, _, _ = control_points.shape
        M = self.basis      # (R, G)
        dM = self.d_basis   # (R, G)

        if weights is None:
            # Standard integral B-spline patch
            S = torch.einsum("ru, bkuvc, sv -> bkrsc", M, control_points, M)

            if return_tangents:
                Tu = torch.einsum("ru, bkuvc, sv -> bkrsc", dM, control_points, M)
                Tv = torch.einsum("ru, bkuvc, sv -> bkrsc", M, control_points, dM)
                return S, Tu, Tv

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

            if return_tangents or return_normals:
                # Analytical partial derivatives via exact quotient rule
                d_Nu = torch.einsum("ru, bkuvc, sv -> bkrsc", dM, P_w, M)
                d_Du = torch.einsum("ru, bkuv, sv -> bkrs", dM, weights, M).unsqueeze(-1)

                d_Nv = torch.einsum("ru, bkuvc, sv -> bkrsc", M, P_w, dM)
                d_Dv = torch.einsum("ru, bkuv, sv -> bkrs", M, weights, dM).unsqueeze(-1)

                S_den_sq = (S_den + 1e-7).pow(2)
                Tu = (d_Nu * S_den - S_num * d_Du) / S_den_sq
                Tv = (d_Nv * S_den - S_num * d_Dv) / S_den_sq

                if return_tangents:
                    return S, Tu, Tv

                surface_points = S.reshape(B, K * self.num_samples * self.num_samples, 3)
                n_unnorm = torch.cross(Tu, Tv, dim=-1)
                normals = F.normalize(n_unnorm, dim=-1, eps=1e-7)
                surface_normals = normals.reshape(B, K * self.num_samples * self.num_samples, 3)
                return surface_points, surface_normals

            surface_points = S.reshape(B, K * self.num_samples * self.num_samples, 3)
            return surface_points


class MultiDegreeNurbsEvaluator(nn.Module):
    """
    Hierarchical Multi-Degree Bernstein Evaluator (4x4 + 5x5 + 6x6).
    Evaluates:
      - Base cubic patch (degree 3, 4x4 control grid, rational)
      - Mid quartic residual patch (degree 4, 5x5 control grid, integral)
      - Fine quintic residual patch (degree 5, 6x6 control grid, integral)
    
    Composite surface points and exact analytical tangent vectors via linearity of derivatives:
      S  = S_3  + Delta S_4  + Delta S_5
      Tu = Tu_3 + Delta Tu_4 + Delta Tu_5
      Tv = Tv_3 + Delta Tv_4 + Delta Tv_5
      Normal = normalize(Tu x Tv)
    """
    def __init__(self, num_samples=16, degrees=(3, 4, 5)):
        super().__init__()
        self.num_samples = num_samples
        self.degrees = tuple(degrees)
        self.evaluators = nn.ModuleDict({
            f"deg_{d}": DifferentiableNurbsEvaluator(num_samples=num_samples, degree=d)
            for d in self.degrees
        })

    def set_eval_res(self, num_samples):
        self.num_samples = num_samples
        device = next(self.parameters()).device if list(self.parameters()) else next(self.buffers()).device
        for d in self.degrees:
            self.evaluators[f"deg_{d}"] = DifferentiableNurbsEvaluator(num_samples=num_samples, degree=d).to(device)

    def forward(self, cp_dict, weights_dict=None, return_normals=True):
        if isinstance(cp_dict, (list, tuple)):
            cp3, cp4, cp5 = cp_dict[0], cp_dict[1], cp_dict[2]
        else:
            cp3 = cp_dict["cp3"]
            cp4 = cp_dict["cp4"]
            cp5 = cp_dict["cp5"]

        w3 = weights_dict.get("w3") if isinstance(weights_dict, dict) else weights_dict

        B, K = cp3.shape[0], cp3.shape[1]

        # 1. Base cubic patch (degree 3)
        S3, Tu3, Tv3 = self.evaluators["deg_3"](cp3, weights=w3, return_tangents=True)

        # 2. Mid quartic residual (degree 4)
        S4, Tu4, Tv4 = self.evaluators["deg_4"](cp4, weights=None, return_tangents=True)

        # 3. Fine quintic residual (degree 5)
        S5, Tu5, Tv5 = self.evaluators["deg_5"](cp5, weights=None, return_tangents=True)

        # Hierarchical sum: coarse base + mid residual + fine residual
        S = S3 + S4 + S5
        surface_points = S.reshape(B, K * self.num_samples * self.num_samples, 3)

        if return_normals:
            Tu = Tu3 + Tu4 + Tu5
            Tv = Tv3 + Tv4 + Tv5
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
    Supports:
      1. Single-degree Biquintic/Bicubic NURBS patch decoder (multi_degree=False)
      2. Hierarchical Multi-Degree Bernstein Decoder (multi_degree=True):
         4x4 Base + 5x5 Residual + 6x6 Residual
    """
    def __init__(
        self,
        embed_dim=128,
        num_patches=32,
        eval_res=16,
        patch_degree=5,
        multi_degree=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.eval_res = eval_res
        self.patch_degree = patch_degree
        self.multi_degree = multi_degree

        # Learnable worker query slots: (K, embed_dim)
        self.patch_queries = nn.Parameter(torch.randn(num_patches, embed_dim) * 0.02)

        in_dim = embed_dim * 3 + 3
        hidden_dim = max(256, embed_dim * 2)

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res1 = ResidualMLPBlock(hidden_dim)
        self.res2 = ResidualMLPBlock(hidden_dim)

        if self.multi_degree:
            # 4x4 Base patch heads (degree 3)
            self.cp_head_3 = nn.Linear(hidden_dim, 4 * 4 * 3)
            self.weight_head_3 = nn.Linear(hidden_dim, 4 * 4)

            # 5x5 Mid-frequency residual head (degree 4)
            self.cp_head_4 = nn.Linear(hidden_dim, 5 * 5 * 3)
            nn.init.normal_(self.cp_head_4.weight, std=1e-3)
            nn.init.zeros_(self.cp_head_4.bias)

            # 6x6 High-frequency residual head (degree 5)
            self.cp_head_5 = nn.Linear(hidden_dim, 6 * 6 * 3)
            nn.init.normal_(self.cp_head_5.weight, std=1e-3)
            nn.init.zeros_(self.cp_head_5.bias)

            self.evaluator = MultiDegreeNurbsEvaluator(num_samples=eval_res, degrees=(3, 4, 5))
        else:
            self.grid_size = patch_degree + 1
            G = self.grid_size
            self.evaluator = DifferentiableNurbsEvaluator(num_samples=eval_res, degree=patch_degree)
            self.cp_head = nn.Linear(hidden_dim, G * G * 3)
            self.weight_head = nn.Linear(hidden_dim, G * G)

    def set_eval_res(self, res):
        self.eval_res = res
        if self.multi_degree:
            self.evaluator.set_eval_res(res)
        else:
            device = next(self.parameters()).device
            self.evaluator = DifferentiableNurbsEvaluator(num_samples=res, degree=self.patch_degree).to(device)

    def forward(self, shape_embedding, zonal_embeddings=None, zone_anchors=None, return_normals=True):
        """
        Args:
            shape_embedding: Tensor of shape (B, embed_dim) from Global Macro Encoder
            zonal_embeddings: Optional tensor of shape (B, K, embed_dim) from K Zonal Encoders
            zone_anchors: Optional tensor of shape (B, K, 3) 3D zone centers
            return_normals: If True, evaluates analytical surface normals
        Returns:
            dict with control_points, weights, surface_points, surface_normals, zone_anchors
        """
        B = shape_embedding.shape[0]

        queries = self.patch_queries.unsqueeze(0).expand(B, -1, -1)
        emb_expanded = shape_embedding.unsqueeze(1).expand(-1, self.num_patches, -1)
        z_feat = zonal_embeddings if zonal_embeddings is not None else emb_expanded
        anchors = zone_anchors if zone_anchors is not None else torch.zeros(B, self.num_patches, 3, device=queries.device, dtype=queries.dtype)

        worker_input = torch.cat([queries, z_feat, emb_expanded, anchors], dim=-1)

        feat = self.in_proj(worker_input)
        feat = self.res1(feat)
        feat = self.res2(feat)

        if self.multi_degree:
            # 1. Base 4x4 (Degree 3) anchored around 3D zone centers
            cp3_delta = self.cp_head_3(feat).reshape(B, self.num_patches, 4, 4, 3)
            if zone_anchors is not None:
                cp3 = cp3_delta + zone_anchors.unsqueeze(2).unsqueeze(3)
            else:
                cp3 = cp3_delta
            w3 = F.softplus(self.weight_head_3(feat)).reshape(B, self.num_patches, 4, 4) + 0.1

            # 2. Mid 5x5 (Degree 4) residual offset
            cp4 = self.cp_head_4(feat).reshape(B, self.num_patches, 5, 5, 3)

            # 3. Fine 6x6 (Degree 5) residual offset
            cp5 = self.cp_head_5(feat).reshape(B, self.num_patches, 6, 6, 3)

            cp_all = [cp3, cp4, cp5]

            if return_normals:
                surface_points, surface_normals = self.evaluator(
                    cp_all, weights_dict=w3, return_normals=True
                )
                return {
                    "control_points": cp_all,
                    "weights": w3,
                    "surface_points": surface_points,
                    "surface_normals": surface_normals,
                    "zone_anchors": zone_anchors,
                }
            else:
                surface_points = self.evaluator(
                    cp_all, weights_dict=w3, return_normals=False
                )
                return {
                    "control_points": cp_all,
                    "weights": w3,
                    "surface_points": surface_points,
                    "zone_anchors": zone_anchors,
                }
        else:
            G = self.grid_size
            cp_delta = self.cp_head(feat).reshape(B, self.num_patches, G, G, 3)
            if zone_anchors is not None:
                cp = cp_delta + zone_anchors.unsqueeze(2).unsqueeze(3)
            else:
                cp = cp_delta
            w = F.softplus(self.weight_head(feat)).reshape(B, self.num_patches, G, G) + 0.1

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
    Penalizes second-order discrete differences across the control grid.
    Acts as a 2D membrane stiffness prior, preventing pillowing and control point flaring.
    Supports a single tensor (B, K, G, G, 3) or a list/tuple of tensors [cp3, cp4, cp5].
    """
    if isinstance(cp, (list, tuple)):
        loss = 0.0
        for p in cp:
            diff_u = p[:, :, 2:, :, :] - 2.0 * p[:, :, 1:-1, :, :] + p[:, :, :-2, :, :]
            diff_v = p[:, :, :, 2:, :] - 2.0 * p[:, :, :, 1:-1, :] + p[:, :, :, :-2, :]
            loss = loss + diff_u.pow(2).mean() + diff_v.pow(2).mean()
        return loss / max(len(cp), 1)
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
    topk_ratio_cov=None,
    lambda_topk_cov=None,
):
    """
    Joint Chamfer Distance, Symmetric Top-k Precision & Coverage Outlier Loss,
    Analytical Normal Alignment, and 2D Control Grid Laplacian Regularizer.
    
    Args:
        p_pred: Predicted surface points (B, N, 3)
        n_pred: Analytical unit surface normals (B, N, 3)
        p_gt: Ground truth surface points (B, M, 3)
        n_gt: Ground truth surface normals (B, M, 3)
        lambda_normal: Weight for symmetric normal alignment loss
        cp: Control points (B, K, G, G, 3) or list [cp3, cp4, cp5] for Laplacian stiffness
        lambda_laplacian: Weight for 2D control point Laplacian penalty
        topk_ratio: Fraction of worst outlier points in Direction 1 (Pred->GT: precision)
        lambda_topk: Weight for Direction 1 top-k precision penalty (crushes stray whiskers)
        topk_ratio_cov: Fraction of worst uncovered points in Direction 2 (GT->Pred: recall)
        lambda_topk_cov: Weight for Direction 2 top-k coverage penalty (forces wings/crossbars)
    Returns:
        total_loss, cd_mean, normal_loss, nc_metric
    """
    dists = torch.cdist(p_pred, p_gt)  # (B, N, M)
    min_d1, idx1 = dists.min(dim=2)    # (B, N) - Pred -> GT (Precision)
    min_d2, idx2 = dists.min(dim=1)    # (B, M) - GT -> Pred (Recall / Coverage)

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

    # Direction 1: Top-k hard outlier penalty (Pred -> GT) to crush boundary whiskers & stray points
    if topk_ratio > 0.0 and lambda_topk > 0.0:
        k1 = max(1, int(topk_ratio * min_d1.shape[1]))
        topk_d1 = torch.topk(min_d1, k=k1, dim=1)[0].mean()
        total_loss = total_loss + lambda_topk * topk_d1

    # Direction 2: Top-k hard coverage penalty (GT -> Pred) to force completion of unobserved CAD wings/crossbars
    effective_cov_ratio = topk_ratio_cov if topk_ratio_cov is not None else topk_ratio
    effective_cov_weight = lambda_topk_cov if lambda_topk_cov is not None else lambda_topk
    if effective_cov_ratio > 0.0 and effective_cov_weight > 0.0:
        k2 = max(1, int(effective_cov_ratio * min_d2.shape[1]))
        topk_d2 = torch.topk(min_d2, k=k2, dim=1)[0].mean()
        total_loss = total_loss + effective_cov_weight * topk_d2

    # 2D Control grid Laplacian stiffness to kill pillowing
    if cp is not None and lambda_laplacian > 0.0:
        lap_loss = control_point_laplacian_loss(cp)
        total_loss = total_loss + lambda_laplacian * lap_loss
        if isinstance(cp, (list, tuple)) and len(cp) >= 3:
            # Gentle L2 residual damping to keep high-order residual grids near zero on planar CAD faces
            res_damping = 0.005 * (cp[1].pow(2).mean() + cp[2].pow(2).mean())
            total_loss = total_loss + res_damping

    return total_loss, cd_mean, normal_loss, nc_metric




