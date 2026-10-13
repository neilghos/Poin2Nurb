import os
import sys
import argparse
import csv
import gc
import hashlib
import json
import numpy as np
import scipy
import trimesh
import torch
from torch.utils.data import DataLoader

# Add paco root to sys.path so we can import PaCo's native utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
ABC_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "abc"))
if PACO_ROOT not in sys.path:
    sys.path.insert(0, PACO_ROOT)

from data import ABCDataset
from model import GeomNet
from utils.evaluator import mesh_comparison


def build_patch_faces(num_patches=32, grid_res=16):
    """
    Precomputes triangle faces for K parametric patches evaluated on an (R x R) grid.
    Each grid quad is split along its diagonal into two triangles.
    Returns: (K * (R-1) * (R-1) * 2, 3) ndarray of vertex indices.
    """
    faces = []
    R = grid_res
    for k in range(num_patches):
        offset = k * (R * R)
        for r in range(R - 1):
            for c in range(R - 1):
                v00 = offset + r * R + c
                v01 = offset + r * R + (c + 1)
                v10 = offset + (r + 1) * R + c
                v11 = offset + (r + 1) * R + (c + 1)
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])
    return np.array(faces, dtype=np.int32)


def save_interactive_visualization(pc, pred, gt, model_id, output_path="visualizations"):
    """
    Generates an interactive 3D comparison plot using Plotly.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"recon_{model_id}.html")

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            f"Input Scan ({pc.shape[0]} pts)",
            f"GeomNet NURBS Surface ({pred.shape[0]} pts)",
            f"Ground Truth ({gt.shape[0]} pts)"
        ],
        horizontal_spacing=0.02
    )

    fig.add_trace(go.Scatter3d(
        x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
        mode="markers",
        marker=dict(size=2, color="rgb(230, 80, 50)", opacity=0.85),
        name="Input Scan"
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=pred[:, 0], y=pred[:, 1], z=pred[:, 2],
        mode="markers",
        marker=dict(size=1.5, color="rgb(0, 180, 220)", opacity=0.85),
        name="GeomNet NURBS"
    ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=gt[:, 0], y=gt[:, 1], z=gt[:, 2],
        mode="markers",
        marker=dict(size=1.5, color="rgb(80, 200, 120)", opacity=0.7),
        name="Ground Truth"
    ), row=1, col=3)

    eye = dict(x=0, y=1.5, z=2)
    scene_layout = dict(
        xaxis=dict(visible=False, range=[-0.6, 0.6]),
        yaxis=dict(visible=False, range=[-0.6, 0.6]),
        zaxis=dict(visible=False, range=[-0.6, 0.6]),
        aspectmode="cube",
        camera=dict(eye=eye, up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0))
    )

    fig.update_layout(
        title=f"GeomNet CAD Surface Reconstruction — Shape {model_id}",
        margin=dict(t=50, b=20, l=10, r=10),
        scene=scene_layout, scene2=scene_layout, scene3=scene_layout,
        height=550, width=1300
    )

    fig.write_html(file_path)
    return file_path


def score_meshes(recon_dir, ref_dir, csv_report, subset_file, num_workers,
                 num_samples, total_models, metadata, test_models=None):
    """Run PaCo's released mesh evaluator and write one mode's result."""
    print(f"\nRunning PaCo mesh_comparison with {num_workers} parallel workers (10,000 points/mesh)...")
    results = mesh_comparison(
        new_meshes_dir_abs=recon_dir,
        ref_meshes_dir_abs=ref_dir,
        num_processes=num_workers,
        report_name=csv_report,
        samples_per_model=num_samples,
        dataset_file_abs=subset_file,
    )
    if results is None:
        raise RuntimeError("PaCo evaluator returned no results")
    with open(csv_report, newline="") as f:
        rows = list(csv.DictReader(f))
    failed = [r for r in rows if float(r["Normal consistency(0: no input)"]) == 0.0]
    if len(rows) != total_models or len(results) + len(failed) != total_models:
        raise RuntimeError(f"Incomplete evaluation: {len(rows)} CSV rows and {len(results)} valid meshes for {total_models} test IDs")
    if failed:
        raise RuntimeError(f"PaCo evaluator reported {len(failed)}/{total_models} failed meshes; unit-cube scoring is not implemented")
    if not all(np.isfinite(float(r[key])) for r in rows for key in
               ("Chamfer dist(1: no input)", "Hausdorff dist", "Normal consistency(0: no input)")):
        raise RuntimeError("Non-finite mesh metric in evaluator CSV")

    cd_vals = [float(r[5]) * 100.0 for r in results]
    hd_vals = [float(r[4]) * 100.0 for r in results]
    nc_vals = [float(r[6]) for r in results]
    mean_cd_x100 = float(np.mean(cd_vals))
    median_cd_x100 = float(np.median(cd_vals))
    mean_hd_x100 = float(np.mean(hd_vals))
    mean_nc = float(np.mean(nc_vals))
    metadata.update({
        "test_models": total_models,
        "cd_x100": mean_cd_x100,
        "hd_x100": mean_hd_x100,
        "nc": mean_nc,
        "scoring_failure_rate": 0.0,
        "failure_definition": "missing mesh or PaCo evaluator returned zero normal consistency; mesh watertightness is not checked",
        "trimesh_version": trimesh.__version__,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    })
    metadata_path = os.path.join(os.path.dirname(os.path.abspath(csv_report)), "metadata.json")
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")

    print("\n================ GeomNet mesh evaluation ================")
    if test_models is not None and total_models != test_models:
        print("  Status                   : exploratory subset; not comparable to full test set")
    print(f"  Models Evaluated         : {len(results)}")
    print(f"  Sampled Surface Points   : {num_samples} pts per mesh")
    print(f"  Chamfer Distance (CDx100): {mean_cd_x100:.2f} (Median: {median_cd_x100:.2f})")
    print(f"  Hausdorff Dist   (HDx100): {mean_hd_x100:.2f}")
    print(f"  Normal Consistency (NC)  : {mean_nc:.4f}")
    print(f"  Scoring failures         : 0/{total_models}")
    print(f"  Detailed Results CSV     : {csv_report}")
    print(f"  Run metadata             : {metadata_path}")
    print("=============================================================================\n")
    return {
        "models_evaluated": len(results),
        "mean_cd_x100": mean_cd_x100,
        "median_cd_x100": median_cd_x100,
        "mean_hd_x100": mean_hd_x100,
        "mean_nc": mean_nc,
        "scoring_failure_rate": 0.0,
    }


def run_evaluation(
    checkpoint_path="checkpoints/geomnet_256.pth",
    data_root=ABC_ROOT,
    mode="easy",
    recon_dir=None,
    csv_report=None,
    num_test_models=None,
    num_samples=10000,
    patch_res=16,
    num_workers=1,
    num_viz=0,
    score_existing=False,
    overwrite=False,
):
    """
    Evaluates GeomNet meshes using PaCo's mesh metric implementation:
      1. Evaluates GeomNet on test shapes and triangulates patches into OBJ meshes.
      2. Calls PaCo's native mesh_comparison on predicted vs reference CAD OBJs.
      3. Samples exactly 10,000 area-weighted surface points from both meshes.
      4. Computes CD x 100, HD x 100, NC, and checks reconstruction failures.
    """
    ref_dir = os.path.join(data_root, "eval", "ref_obj")
    if recon_dir is None:
        recon_dir = os.path.join(CURRENT_DIR, "output", "benchmarks", mode,
                                 os.path.splitext(os.path.basename(checkpoint_path))[0], "reconstructed_obj")
    if csv_report is None:
        csv_report = os.path.join(os.path.dirname(recon_dir), "evaluation.csv")
    if num_test_models is not None and num_test_models <= 0:
        raise ValueError("--samples must be positive")
    if num_samples != 10000:
        raise ValueError("The PaCo Table 1 protocol fixes sampling at 10,000 points")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=============================================================================")
    print("       GeomNet evaluation with PaCo mesh metrics")
    print("=============================================================================")
    print(f"Device               : {device}")
    print(f"Checkpoint           : {checkpoint_path}")
    print(f"ABC data             : {data_root} ({mode})")
    print(f"Reference CAD Meshes : {ref_dir}")
    print(f"Sampled Points       : {num_samples} per mesh")
    print(f"Patch Triangulation  : {patch_res}x{patch_res} grid per patch")
    print(f"CPU Workers          : {num_workers}")

    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"Reference mesh directory not found at {ref_dir}")

    # 1. Load Model
    resolved_ckpt = checkpoint_path
    if not os.path.exists(resolved_ckpt):
        alt = os.path.join(CURRENT_DIR, checkpoint_path)
        if os.path.exists(alt):
            resolved_ckpt = alt

    if os.path.isfile(resolved_ckpt):
        checkpoint_hash = hashlib.sha256()
        with open(resolved_ckpt, "rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                checkpoint_hash.update(chunk)
        ckpt = torch.load(resolved_ckpt, map_location=device)
        k_patches = ckpt.get("num_patches", 32)
        deg = ckpt.get("patch_degree", 5)
        e_dim = ckpt.get("embed_dim", 256)
        use_gnn = ckpt.get("use_macro_gnn", any("macro_gnn" in k for k in ckpt.get("model_state", {}).keys()))
        is_multi_degree = ckpt.get(
            "multi_degree",
            any("cp_head_3" in k or "evaluators.deg_3" in k for k in ckpt.get("model_state", {}).keys())
        )
        model = GeomNet(
            embed_dim=e_dim,
            num_patches=k_patches,
            eval_res=patch_res,
            use_macro_gnn=use_gnn,
            patch_degree=deg,
            multi_degree=is_multi_degree,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.decoder.set_eval_res(patch_res)
        deg_str = "Hierarchical Multi-Degree (4x4+5x5+6x6)" if is_multi_degree else f"degree {deg}"
        gnn_str = "with Macro GNN" if use_gnn else "without Macro GNN"
        cd_val = ckpt.get("val_cd_x100", ckpt.get("test_cd_x100", None))
        cd_str = f"Val CD: {cd_val:.2f}" if isinstance(cd_val, (int, float)) else "CD: ?"
        print(f"Loaded weights from epoch {ckpt.get('epoch', '?')} (K={k_patches}, {deg_str}, {gnn_str}, {cd_str})")
        print(f"Checkpoint SHA-256: {checkpoint_hash.hexdigest()}")
    else:
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint_path}")

    model.eval()

    if score_existing:
        if not os.path.isdir(recon_dir):
            raise FileNotFoundError(f"Existing reconstruction directory not found: {recon_dir}")
        with open(os.path.join(data_root, "test.txt")) as manifest_file:
            manifest_ids = [line.strip().removesuffix(".npy") for line in manifest_file
                            if line.strip()]
        total_models = len(manifest_ids) if num_test_models is None else min(num_test_models, len(manifest_ids))
        subset_file = os.path.join(recon_dir, "eval_subset.txt")
        subset_ids = manifest_ids[:total_models]
        existing_subset = []
        if os.path.isfile(subset_file):
            with open(subset_file) as subset_handle:
                existing_subset = [line.strip().removesuffix(".npy") for line in subset_handle
                                   if line.strip()]
        if existing_subset != subset_ids:
            with open(subset_file, "w") as subset_handle:
                subset_handle.write("\n".join(subset_ids) + "\n")
        metadata = {
            "checkpoint": os.path.abspath(resolved_ckpt),
            "checkpoint_sha256": checkpoint_hash.hexdigest(),
            "checkpoint_epoch": ckpt.get("epoch"),
            "data_root": os.path.abspath(data_root),
            "mode": mode,
            "surface_samples_per_mesh": num_samples,
            "patch_grid_resolution": patch_res,
            "score_existing": True,
        }
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return score_meshes(recon_dir, ref_dir, csv_report, subset_file,
                            num_workers, num_samples, total_models, metadata,
                            test_models=len(manifest_ids))

    # Precompute patch quad-to-triangle faces
    faces = build_patch_faces(num_patches=k_patches, grid_res=patch_res)

    # 2. Reconstruct OBJ Meshes from GeomNet
    os.makedirs(recon_dir, exist_ok=True)
    if any(name.endswith(".obj") for name in os.listdir(recon_dir)):
        if overwrite:
            for f in os.listdir(recon_dir):
                if f.endswith(".obj") or f.endswith(".txt"):
                    try:
                        os.remove(os.path.join(recon_dir, f))
                    except OSError:
                        pass
        else:
            raise FileExistsError(f"Reconstruction directory contains old OBJ files: {recon_dir}. Pass --overwrite to regenerate them, or choose a new --output_root.")
    os.makedirs(os.path.dirname(os.path.abspath(csv_report)), exist_ok=True)

    test_dataset = ABCDataset(data_root=data_root, split="test", mode=mode,
                              match_paco_input=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    total_models = len(test_dataset) if num_test_models is None else min(num_test_models, len(test_dataset))
    print(f"Exporting {total_models} reconstructed OBJ meshes to {recon_dir}...")

    viz_saved = 0
    exported_ids = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if num_test_models is not None and i >= num_test_models:
                break
            pc_t = batch["pc"].to(device)
            m_id = batch["model_id"][0]

            out = model(pc_t, return_normals=False)
            verts = out["surface_points"][0].cpu().numpy()  # (K * R * R, 3)
            if not np.isfinite(verts).all():
                raise RuntimeError(f"Non-finite predicted surface for {m_id}")

            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            obj_path = os.path.join(recon_dir, f"{m_id}.obj")
            mesh.export(obj_path)
            exported_ids.append(m_id)

            if viz_saved < num_viz:
                gt_pts = batch["gt"][0].cpu().numpy()
                pc_pts = pc_t[0].cpu().numpy()
                html_path = save_interactive_visualization(pc_pts, verts, gt_pts, m_id)
                print(f"  [Saved 3D Visualization] {html_path}")
                viz_saved += 1

            if (i + 1) % 250 == 0 or (i + 1) == total_models:
                print(f"  Exported {i + 1}/{total_models} OBJ meshes...")

    # Always pass the exact test manifest so missing and unrelated OBJs cannot be hidden.
    subset_file = os.path.join(recon_dir, "eval_subset.txt")
    with open(subset_file, "w") as f:
        f.write("\n".join(exported_ids) + "\n")

    metadata = {
        "checkpoint": os.path.abspath(resolved_ckpt),
        "checkpoint_sha256": checkpoint_hash.hexdigest(),
        "checkpoint_epoch": ckpt.get("epoch"),
        "data_root": os.path.abspath(data_root),
        "mode": mode,
        "test_models": total_models,
        "surface_samples_per_mesh": num_samples,
        "patch_grid_resolution": patch_res,
    }
    # Release the network and loader before PaCo starts worker processes. This
    # materially lowers the parent RSS and avoids forked model-memory copies.
    del model, test_loader, test_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score_meshes(recon_dir, ref_dir, csv_report, subset_file,
                        num_workers, num_samples, total_models, metadata,
                        test_models=total_models if num_test_models is not None else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeomNet evaluation using PaCo mesh metrics")
    parser.add_argument("--ckpt", type=str, default="checkpoints/geomnet_256.pth", help="Trained checkpoint path")
    parser.add_argument("--samples", type=int, default=None, help="Number of test shapes to evaluate (default: full test set)")
    parser.add_argument("--data_root", type=str, default=ABC_ROOT, help="ABC data root")
    parser.add_argument("--mode", choices=("all", "mixed", "easy", "medium", "hard"), required=True,
                        help="Evaluate one mode, or all three difficulty modes and aggregate them")
    parser.add_argument("--output_root", type=str, default="output/paper_eval",
                        help="Output root used by --mode all")
    parser.add_argument("--recon_dir", type=str, default=None, help="New, empty directory for predicted OBJ meshes")
    parser.add_argument("--csv_out", type=str, default=None, help="Output CSV report path")
    parser.add_argument("--workers", type=int, default=1, help="Number of CPU workers for parallel KDTree queries (1 minimizes RAM)")
    parser.add_argument("--viz", type=int, default=0, help="Number of interactive 3D visualizations to save")
    parser.add_argument("--patch_res", type=int, default=16, help="Grid resolution per patch for triangulation (default: 16)")
    parser.add_argument("--points", type=int, default=10000, help="Points sampled per mesh (default: 10,000 for CVPR Table 1)")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing reconstructed OBJ files")
    parser.add_argument("--score-existing", action="store_true",
                        help="Score existing OBJ files in --recon_dir without rerunning inference")
    args = parser.parse_args()

    if args.mode != "all":
        run_evaluation(
            checkpoint_path=args.ckpt,
            data_root=args.data_root,
            mode=args.mode,
            recon_dir=args.recon_dir,
            csv_report=args.csv_out,
            num_test_models=args.samples,
            num_samples=args.points,
            patch_res=args.patch_res,
            num_workers=args.workers,
        num_viz=args.viz,
        score_existing=args.score_existing,
        overwrite=args.overwrite,
        )
    else:
        if args.recon_dir is not None or args.csv_out is not None:
            raise ValueError("--mode all chooses mode-specific outputs under --output_root")
        mode_results = {}
        for mode in ("easy", "medium", "hard"):
            mode_root = os.path.join(args.output_root, mode)
            mode_results[mode] = run_evaluation(
                checkpoint_path=args.ckpt,
                data_root=args.data_root,
                mode=mode,
                recon_dir=os.path.join(mode_root, "reconstructed_obj"),
                csv_report=os.path.join(mode_root, "evaluation.csv"),
                num_test_models=args.samples,
                num_samples=args.points,
                patch_res=args.patch_res,
                num_workers=args.workers,
                num_viz=args.viz,
                score_existing=args.score_existing,
            overwrite=args.overwrite,
            )
        aggregate = {
            "cd_x100": float(np.mean([mode_results[m]["mean_cd_x100"] for m in mode_results])),
            "hd_x100": float(np.mean([mode_results[m]["mean_hd_x100"] for m in mode_results])),
            "nc": float(np.mean([mode_results[m]["mean_nc"] for m in mode_results])),
        }
        table_summary = {
            "protocol": "PaCo native mesh evaluator; equal mean of easy, medium, hard",
            "models_per_mode": len(ABCDataset(data_root=args.data_root, split="test", mode="easy")),
            "total_evaluations": sum(mode_results[m]["models_evaluated"] for m in mode_results),
            "surface_samples_per_mesh": args.points,
            "patch_grid_resolution": args.patch_res,
            "modes": mode_results,
            "aggregate": aggregate,
            "reported_paco_table1": {"cd_x100": 1.87, "hd_x100": 4.09,
                                      "nc": 0.943, "failure_rate_percent": 0.48},
        }
        summary_path = os.path.join(args.output_root, "table1_summary.json")
        os.makedirs(args.output_root, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(table_summary, f, indent=2)
            f.write("\n")
        print("\n================ Table 1-style GeomNet summary ================")
        print(json.dumps(table_summary, indent=2))
        print(f"Saved {summary_path}")
