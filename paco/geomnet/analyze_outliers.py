import os
import sys
import json
import argparse
import numpy as np
from scipy import spatial
import torch
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import ABCDataset
from model import GeomNet
from decoder import DifferentiableNurbsEvaluator


def compute_detailed_metrics(pred_pts, gt_pts, pred_normals=None, gt_normals=None):
    """
    Computes fine-grained directional metrics and point-wise distance residuals.
    """
    tree_pred = spatial.cKDTree(pred_pts)
    tree_gt = spatial.cKDTree(gt_pts)

    # Direction 1: Predicted -> GT (precision / stray points)
    d_pred_to_gt, corr_gt_ids = tree_gt.query(pred_pts, 1)

    # Direction 2: GT -> Predicted (coverage / recall / missed holes)
    d_gt_to_pred, corr_pred_ids = tree_pred.query(gt_pts, 1)

    mean_p2g = float(d_pred_to_gt.mean())
    mean_g2p = float(d_gt_to_pred.mean())
    cd = mean_p2g + mean_g2p
    cd_x100 = cd * 100.0

    hd_p2g = float(d_pred_to_gt.max())
    hd_g2p = float(d_gt_to_pred.max())
    hd = max(hd_p2g, hd_g2p)
    hd_x100 = hd * 100.0

    nc = None
    if gt_normals is not None and pred_normals is not None:
        dot1 = np.abs(np.sum(pred_normals * gt_normals[corr_gt_ids], axis=1)).mean()
        dot2 = np.abs(np.sum(gt_normals * pred_normals[corr_pred_ids], axis=1)).mean()
        nc = float((dot1 + dot2) / 2.0)

    return {
        "cd_x100": cd_x100,
        "mean_p2g_x100": mean_p2g * 100.0,
        "mean_g2p_x100": mean_g2p * 100.0,
        "hd_x100": hd_x100,
        "hd_p2g_x100": hd_p2g * 100.0,
        "hd_g2p_x100": hd_g2p * 100.0,
        "nc": nc,
        "d_pred_to_gt": d_pred_to_gt,
        "d_gt_to_pred": d_gt_to_pred,
    }


def save_diagnostic_visualization(
    pc, pred, gt, d_pred_to_gt, patch_ids, model_id, rank, rank_type, metrics, output_dir="visualizations/outliers"
):
    """
    Saves a 4-panel diagnostic Plotly HTML:
      Panel 1: Input partial point cloud
      Panel 2: Predicted NURBS surface colored by pointwise Chamfer error heatmap
      Panel 3: Predicted NURBS surface colored by 32 discrete patches
      Panel 4: Ground Truth CAD model
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{rank_type}_{rank:02d}_{model_id}_cd_{metrics['cd_x100']:.2f}_hd_{metrics['hd_x100']:.2f}.html"
    filepath = os.path.join(output_dir, filename)

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            f"Input Scan ({pc.shape[0]} pts)",
            "Pointwise Error Heatmap (Pred->GT)",
            "Discrete NURBS Patches (32 patches)",
            f"Ground Truth CAD ({gt.shape[0]} pts)",
        ],
        horizontal_spacing=0.015,
    )

    # Panel 1: Input partial scan
    fig.add_trace(go.Scatter3d(
        x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
        mode="markers",
        marker=dict(size=2.0, color="rgb(230, 70, 50)", opacity=0.85),
        name="Input Scan",
    ), row=1, col=1)

    # Panel 2: Predicted NURBS surface with Chamfer error heatmap
    # Cap error visualization at 95th percentile or 0.05 for high-contrast viewing
    err_vals = d_pred_to_gt * 100.0
    vmax = min(max(float(np.percentile(err_vals, 95)), 3.0), 15.0)
    fig.add_trace(go.Scatter3d(
        x=pred[:, 0], y=pred[:, 1], z=pred[:, 2],
        mode="markers",
        marker=dict(
            size=1.5,
            color=err_vals,
            colorscale="Plasma",
            cmin=0.0,
            cmax=vmax,
            colorbar=dict(title="Error x100", len=0.6, x=0.48),
            opacity=0.9,
        ),
        name="Error Heatmap",
    ), row=1, col=2)

    # Panel 3: Discrete Patches
    fig.add_trace(go.Scatter3d(
        x=pred[:, 0], y=pred[:, 1], z=pred[:, 2],
        mode="markers",
        marker=dict(
            size=1.5,
            color=patch_ids,
            colorscale="Turbo",
            opacity=0.9,
        ),
        name="Patches",
    ), row=1, col=3)

    # Panel 4: Ground Truth CAD model
    fig.add_trace(go.Scatter3d(
        x=gt[:, 0], y=gt[:, 1], z=gt[:, 2],
        mode="markers",
        marker=dict(size=1.5, color="rgb(40, 180, 100)", opacity=0.75),
        name="Ground Truth",
    ), row=1, col=4)

    eye = dict(x=0, y=1.5, z=2)
    scene_layout = dict(
        xaxis=dict(visible=False, range=[-0.6, 0.6]),
        yaxis=dict(visible=False, range=[-0.6, 0.6]),
        zaxis=dict(visible=False, range=[-0.6, 0.6]),
        aspectmode="cube",
        camera=dict(eye=eye, up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)),
    )

    nc_str = f"{metrics['nc']:.4f}" if metrics['nc'] is not None else "N/A"
    title_text = (
        f"GeomNet Diagnostic [{rank_type.upper()} Rank {rank}] Shape: {model_id} | "
        f"CD x 100: {metrics['cd_x100']:.2f} (P2G: {metrics['mean_p2g_x100']:.2f}, G2P: {metrics['mean_g2p_x100']:.2f}) | "
        f"HD x 100: {metrics['hd_x100']:.2f} (P2G: {metrics['hd_p2g_x100']:.2f}, G2P: {metrics['hd_g2p_x100']:.2f}) | "
        f"NC: {nc_str}"
    )

    fig.update_layout(
        title=title_text,
        margin=dict(t=50, b=20, l=10, r=10),
        scene=scene_layout,
        scene2=scene_layout,
        scene3=scene_layout,
        scene4=scene_layout,
        height=550, width=1700,
    )

    fig.write_html(filepath)
    return filepath


def run_analysis(
    checkpoint_path="checkpoints/geomnet_256.pth",
    eval_points=10000,
    top_k_viz=10,
    output_dir="visualizations/outliers",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(">>> GEOMNET COMPREHENSIVE OUTLIER & ERROR PATHOLOGY AUDIT <<<")
    print(f"Device          : {device}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Eval Points     : {eval_points} pts/shape (CVPR 2025 Table 1 aligned)")
    print("=" * 75)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    k_patches = ckpt.get("num_patches", 32)
    deg = ckpt.get("patch_degree", 5)
    e_dim = ckpt.get("embed_dim", 256)
    use_gnn = ckpt.get("use_macro_gnn", True)

    req_res = int(np.ceil(np.sqrt(eval_points / k_patches)))
    actual_pts = k_patches * req_res * req_res

    model = GeomNet(
        embed_dim=e_dim,
        num_patches=k_patches,
        eval_res=req_res,
        use_macro_gnn=use_gnn,
        patch_degree=deg,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_dataset = ABCDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Loaded {len(test_dataset)} test shapes from test.txt.")

    records = []
    print("\nRunning full forward pass and metric computation across all shapes...")

    pts_per_patch = req_res * req_res
    patch_ids_full = np.repeat(np.arange(k_patches), pts_per_patch)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            pc_t = batch["pc"].to(device)
            gt_t = batch["gt"].to(device)
            gt_n = batch["gt_normals"][0].cpu().numpy() if "gt_normals" in batch else None
            m_id = batch["model_id"][0]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(pc_t, return_normals=True)

            pred_pts = out["surface_points"][0].float().cpu().numpy()
            pred_n = out["surface_normals"][0].float().cpu().numpy() if "surface_normals" in out else None

            # Subsample to exact eval_points if needed
            if eval_points and len(pred_pts) > eval_points:
                sample_idx = np.linspace(0, len(pred_pts) - 1, eval_points, dtype=int)
                pred_pts_sub = pred_pts[sample_idx]
                pred_n_sub = pred_n[sample_idx] if pred_n is not None else None
                patch_ids = patch_ids_full[sample_idx]
            else:
                pred_pts_sub = pred_pts
                pred_n_sub = pred_n
                patch_ids = patch_ids_full

            gt_pts = gt_t[0].cpu().numpy()
            pc_pts = pc_t[0].cpu().numpy()

            metrics = compute_detailed_metrics(pred_pts_sub, gt_pts, pred_normals=pred_n_sub, gt_normals=gt_n)

            rec = {
                "index": i,
                "model_id": m_id,
                "cd_x100": metrics["cd_x100"],
                "mean_p2g_x100": metrics["mean_p2g_x100"],
                "mean_g2p_x100": metrics["mean_g2p_x100"],
                "hd_x100": metrics["hd_x100"],
                "hd_p2g_x100": metrics["hd_p2g_x100"],
                "hd_g2p_x100": metrics["hd_g2p_x100"],
                "nc": metrics["nc"],
                # Cache arrays for top outliers
                "pc": pc_pts,
                "pred": pred_pts_sub,
                "gt": gt_pts,
                "d_pred_to_gt": metrics["d_pred_to_gt"],
                "patch_ids": patch_ids,
            }
            records.append(rec)

            if (i + 1) % 250 == 0 or (i + 1) == len(test_dataset):
                print(f"  Processed [{i + 1:4d} / {len(test_dataset):4d}] shapes...")

    # Sort and analyze
    cds = np.array([r["cd_x100"] for r in records])
    hds = np.array([r["hd_x100"] for r in records])
    ncs = np.array([r["nc"] for r in records if r["nc"] is not None])
    p2gs = np.array([r["mean_p2g_x100"] for r in records])
    g2ps = np.array([r["mean_g2p_x100"] for r in records])

    total_n = len(cds)
    mean_cd = float(np.mean(cds))
    median_cd = float(np.median(cds))
    mean_hd = float(np.mean(hds))
    median_hd = float(np.median(hds))
    mean_nc = float(np.mean(ncs))

    print("\n" + "=" * 75)
    print(">>> STATISTICAL DISTRIBUTION BREAKDOWN <<<")
    print(f"Total Test Shapes: {total_n}")
    print(f"Chamfer Distance (CD x 100):")
    print(f"  Mean   : {mean_cd:.2f}")
    print(f"  Median : {median_cd:.2f}")
    print(f"  p10    : {np.percentile(cds, 10):.2f}")
    print(f"  p25    : {np.percentile(cds, 25):.2f}")
    print(f"  p75    : {np.percentile(cds, 75):.2f}")
    print(f"  p90    : {np.percentile(cds, 90):.2f}")
    print(f"  p95    : {np.percentile(cds, 95):.2f}")
    print(f"  p98    : {np.percentile(cds, 98):.2f}")
    print(f"  p99    : {np.percentile(cds, 99):.2f}")
    print(f"  Max    : {np.max(cds):.2f}")

    print(f"\nHausdorff Distance (HD x 100):")
    print(f"  Mean   : {mean_hd:.2f}")
    print(f"  Median : {median_hd:.2f}")
    print(f"  p90    : {np.percentile(hds, 90):.2f}")
    print(f"  p95    : {np.percentile(hds, 95):.2f}")
    print(f"  p99    : {np.percentile(hds, 99):.2f}")
    print(f"  Max    : {np.max(hds):.2f}")

    print(f"\nNormal Consistency (NC):")
    print(f"  Mean   : {mean_nc:.4f}")
    print(f"  Median : {np.median(ncs):.4f}")

    # Bracket counts
    print("\n--- Error Bracket Distribution ---")
    b1 = np.sum(cds < 1.50)
    b2 = np.sum((cds >= 1.50) & (cds < 2.00))
    b3 = np.sum((cds >= 2.00) & (cds < 2.50))
    b4 = np.sum((cds >= 2.50) & (cds < 3.50))
    b5 = np.sum((cds >= 3.50) & (cds < 5.00))
    b6 = np.sum(cds >= 5.00)

    print(f"  CD < 1.50      (Superb)          : {b1:4d} shapes ({b1 / total_n * 100:.1f}%)")
    print(f"  1.50 <= CD < 2.00 (Excellent)    : {b2:4d} shapes ({b2 / total_n * 100:.1f}%)")
    print(f"  2.00 <= CD < 2.50 (Good)         : {b3:4d} shapes ({b3 / total_n * 100:.1f}%)")
    print(f"  2.50 <= CD < 3.50 (Moderate)     : {b4:4d} shapes ({b4 / total_n * 100:.1f}%)")
    print(f"  3.50 <= CD < 5.00 (Elevated)     : {b5:4d} shapes ({b5 / total_n * 100:.1f}%)")
    print(f"  CD >= 5.00      (Severe Outliers): {b6:4d} shapes ({b6 / total_n * 100:.1f}%)")

    # Trimmed Mean Analysis
    print("\n--- Outlier Impact: Trimmed Mean Analysis ---")
    sorted_cds = np.sort(cds)
    trim_1pct = int(total_n * 0.01)
    trim_2pct = int(total_n * 0.02)
    trim_5pct = int(total_n * 0.05)

    mean_trim1 = np.mean(sorted_cds[:-trim_1pct])
    mean_trim2 = np.mean(sorted_cds[:-trim_2pct])
    mean_trim5 = np.mean(sorted_cds[:-trim_5pct])

    print(f"  Full Test Set (2,375 shapes)       : Mean CD = {mean_cd:.2f}")
    print(f"  Excluding Top 1% ({trim_1pct:2d} worst shapes) : Mean CD = {mean_trim1:.2f} (a {- (mean_cd - mean_trim1):.2f} drop)")
    print(f"  Excluding Top 2% ({trim_2pct:2d} worst shapes) : Mean CD = {mean_trim2:.2f} (a {- (mean_cd - mean_trim2):.2f} drop)")
    print(f"  Excluding Top 5% ({trim_5pct:2d} worst shapes) : Mean CD = {mean_trim5:.2f} (a {- (mean_cd - mean_trim5):.2f} drop)")

    # Directional Analysis
    print("\n--- Directional Bias Analysis ---")
    mean_p2g = np.mean(p2gs)
    mean_g2p = np.mean(g2ps)
    print(f"  Pred -> GT (Precision / stray points) : {mean_p2g:.2f} ({mean_p2g / mean_cd * 100:.1f}% of CD)")
    print(f"  GT -> Pred (Recall / missed coverage) : {mean_g2p:.2f} ({mean_g2p / mean_cd * 100:.1f}% of CD)")

    # Identify Worst CD Outliers
    records_by_cd = sorted(records, key=lambda x: x["cd_x100"], reverse=True)
    records_by_hd = sorted(records, key=lambda x: x["hd_x100"], reverse=True)

    print("\n" + "=" * 75)
    print(f">>> TOP 15 WORST CHAMFER DISTANCE OUTLIERS <<<")
    print(f"{'Rank':<5} {'Model ID':<12} {'CDx100':<8} {'P2G x100':<10} {'G2P x100':<10} {'HDx100':<8} {'NC':<8}")
    print("-" * 65)
    for rank, r in enumerate(records_by_cd[:15], 1):
        nc_val = f"{r['nc']:.4f}" if r["nc"] is not None else "N/A"
        print(f"{rank:<5} {r['model_id']:<12} {r['cd_x100']:<8.2f} {r['mean_p2g_x100']:<10.2f} {r['mean_g2p_x100']:<10.2f} {r['hd_x100']:<8.2f} {nc_val:<8}")

    print("\n" + "=" * 75)
    print(f">>> TOP 15 WORST HAUSDORFF DISTANCE OUTLIERS <<<")
    print(f"{'Rank':<5} {'Model ID':<12} {'HDx100':<8} {'HD_P2G':<10} {'HD_G2P':<10} {'CDx100':<8} {'NC':<8}")
    print("-" * 65)
    for rank, r in enumerate(records_by_hd[:15], 1):
        nc_val = f"{r['nc']:.4f}" if r["nc"] is not None else "N/A"
        print(f"{rank:<5} {r['model_id']:<12} {r['hd_x100']:<8.2f} {r['hd_p2g_x100']:<10.2f} {r['hd_g2p_x100']:<10.2f} {r['cd_x100']:<8.2f} {nc_val:<8}")

    # Generate Visualizations
    print("\n" + "=" * 75)
    print(f">>> GENERATING 3D DIAGNOSTIC VISUALIZATIONS IN {output_dir} <<<")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Top CD Outliers
    print("\nGenerating Top Worst Chamfer Outliers...")
    for rank, r in enumerate(records_by_cd[:top_k_viz], 1):
        path = save_diagnostic_visualization(
            r["pc"], r["pred"], r["gt"], r["d_pred_to_gt"], r["patch_ids"],
            r["model_id"], rank, "cd_outlier", r, output_dir=output_dir
        )
        print(f"  [Saved CD Outlier #{rank:02d}] {path}")

    # 2. Top HD Outliers
    print("\nGenerating Top Worst Hausdorff Outliers...")
    for rank, r in enumerate(records_by_hd[:top_k_viz], 1):
        path = save_diagnostic_visualization(
            r["pc"], r["pred"], r["gt"], r["d_pred_to_gt"], r["patch_ids"],
            r["model_id"], rank, "hd_outlier", r, output_dir=output_dir
        )
        print(f"  [Saved HD Outlier #{rank:02d}] {path}")

    # 3. Median Representative Shapes
    print("\nGenerating Typical Median Shapes for Comparison...")
    median_idx = int(len(records_by_cd) * 0.5)
    for rank, r in enumerate(records_by_cd[median_idx : median_idx + 3], 1):
        path = save_diagnostic_visualization(
            r["pc"], r["pred"], r["gt"], r["d_pred_to_gt"], r["patch_ids"],
            r["model_id"], rank, "median_shape", r, output_dir=output_dir
        )
        print(f"  [Saved Median #{rank:02d}] {path}")

    # Save summary JSON (excluding heavy point cloud arrays)
    summary_path = os.path.join(output_dir, "outlier_analysis_summary.json")
    clean_records = []
    for r in records_by_cd:
        clean_records.append({
            "model_id": r["model_id"],
            "cd_x100": r["cd_x100"],
            "mean_p2g_x100": r["mean_p2g_x100"],
            "mean_g2p_x100": r["mean_g2p_x100"],
            "hd_x100": r["hd_x100"],
            "hd_p2g_x100": r["hd_p2g_x100"],
            "hd_g2p_x100": r["hd_g2p_x100"],
            "nc": r["nc"],
        })

    with open(summary_path, "w") as f:
        json.dump({
            "stats": {
                "total_models": total_n,
                "mean_cd_x100": mean_cd,
                "median_cd_x100": median_cd,
                "mean_hd_x100": mean_hd,
                "median_hd_x100": median_hd,
                "mean_nc": mean_nc,
                "trimmed_mean_1pct": float(mean_trim1),
                "trimmed_mean_2pct": float(mean_trim2),
                "trimmed_mean_5pct": float(mean_trim5),
                "brackets": {
                    "sub_1.50": int(b1),
                    "1.50_to_2.00": int(b2),
                    "2.00_to_2.50": int(b3),
                    "2.50_to_3.50": int(b4),
                    "3.50_to_5.00": int(b5),
                    "above_5.00": int(b6),
                }
            },
            "top_15_worst_cd": clean_records[:15],
            "top_15_worst_hd": sorted(clean_records, key=lambda x: x["hd_x100"], reverse=True)[:15],
        }, f, indent=2)

    print(f"\n[Saved Full Analysis JSON] {summary_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeomNet Outlier & Error Pathology Analyzer")
    parser.add_argument("--ckpt", type=str, default="checkpoints/geomnet_256.pth", help="Checkpoint path")
    parser.add_argument("--points", type=int, default=10000, help="Surface points to evaluate (default 10000)")
    parser.add_argument("--viz", type=int, default=10, help="Number of outlier shapes to visualize (default 10)")
    parser.add_argument("--outdir", type=str, default="visualizations/outliers", help="Output directory")
    args = parser.parse_args()

    run_analysis(checkpoint_path=args.ckpt, eval_points=args.points, top_k_viz=args.viz, output_dir=args.outdir)
