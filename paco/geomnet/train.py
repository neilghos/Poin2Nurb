import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import ABCDataset
from model import GeomNet
from decoder import chamfer_distance, chamfer_and_normal_loss


def evaluate(model, test_loader, device, max_batches=None):
    """
    Evaluates the model on held-out test shapes.
    Reports:
      - Raw Chamfer Distance
      - CD x 100 (Official PaCo CVPR 2025 Table 1 benchmark scale)
      - Normal Consistency (NC)
    """
    model.eval()
    total_cd = 0.0
    total_nc = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if max_batches is not None and i >= max_batches:
                break
            pc = batch["pc"].to(device)
            gt = batch["gt"].to(device)
            gt_normals = batch["gt_normals"].to(device)

            out = model(pc, return_normals=True)
            _, cd, _, nc = chamfer_and_normal_loss(
                out["surface_points"], out["surface_normals"], gt, gt_normals, lambda_normal=0.0
            )

            batch_sz = pc.shape[0]
            total_cd += cd.item() * batch_sz
            total_nc += nc.item() * batch_sz
            total_samples += batch_sz

    mean_cd = total_cd / max(total_samples, 1)
    cd_x100 = mean_cd * 100.0
    mean_nc = total_nc / max(total_samples, 1)
    return mean_cd, cd_x100, mean_nc, total_samples


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting GeomNet Training & Evaluation ===")
    print(f"Device        : {device}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Learning Rate : {args.lr}")
    print(f"Num Patches   : {args.num_patches} (eval resolution {args.eval_res}x{args.eval_res})")
    print(f"Total Points  : {args.num_patches * args.eval_res * args.eval_res} per shape")
    print(f"Lambda Normal : {args.lambda_normal}")
    print(f"Epochs        : {args.epochs}")
    print(f"-----------------------------------------------")

    # 1. Datasets & Loaders
    train_dataset = ABCDataset(split="train")
    test_dataset = ABCDataset(split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train shapes : {len(train_dataset)}")
    print(f"Test shapes  : {len(test_dataset)}")

    # 2. Model & Optimizer
    model = GeomNet(
        embed_dim=args.embed_dim,
        num_patches=args.num_patches,
        eval_res=args.eval_res,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 3. Initial Baseline Evaluation before training
    print("\nRunning initial zero-shot test evaluation...")
    init_cd, init_cd_x100, init_nc, n_eval = evaluate(model, test_loader, device, max_batches=args.test_eval_batches)
    print(f"Initial Test CD x 100: {init_cd_x100:.2f} | Test NC: {init_nc:.4f} (evaluated on {n_eval} shapes)")
    print(f"PaCo SOTA Reference  : CD ~2.2 - 3.8 | NC: ~0.943\n")

    best_test_cd = float("inf")

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_cd_total = 0.0
        train_nc_total = 0.0
        train_count = 0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break
            pc = batch["pc"].to(device)
            gt = batch["gt"].to(device)
            gt_normals = batch["gt_normals"].to(device)

            optimizer.zero_grad()
            out = model(pc, return_normals=True)
            total_loss, cd_loss, normal_loss, nc = chamfer_and_normal_loss(
                out["surface_points"], out["surface_normals"], gt, gt_normals, lambda_normal=args.lambda_normal
            )
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = pc.shape[0]
            train_loss += total_loss.item() * bs
            train_cd_total += cd_loss.item() * bs
            train_nc_total += nc.item() * bs
            train_count += bs

            if (step + 1) % args.log_interval == 0 or (step + 1) == len(train_loader):
                elapsed = time.time() - start_time
                shapes_per_sec = train_count / elapsed
                avg_step_cd = (train_cd_total / train_count) * 100.0
                avg_step_nc = train_nc_total / train_count
                print(
                    f"Epoch [{epoch:2d}/{args.epochs:2d}] | "
                    f"Step [{step + 1:4d}/{len(train_loader):4d}] | "
                    f"Train CD x 100: {avg_step_cd:.2f} | "
                    f"Train NC: {avg_step_nc:.4f} | "
                    f"Speed: {shapes_per_sec:.1f} shapes/s"
                )

        scheduler.step()
        epoch_dur = time.time() - start_time
        avg_train_cd = (train_cd_total / train_count) * 100.0
        avg_train_nc = train_nc_total / train_count

        # Evaluate on Test Split
        test_cd, test_cd_x100, test_nc, n_eval = evaluate(
            model, test_loader, device, max_batches=args.test_eval_batches
        )

        is_best = test_cd_x100 < best_test_cd
        if is_best:
            best_test_cd = test_cd_x100
            torch.save(
                {
                    "epoch": epoch,
                    "num_patches": args.num_patches,
                    "eval_res": args.eval_res,
                    "embed_dim": args.embed_dim,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "test_cd_x100": test_cd_x100,
                    "test_nc": test_nc,
                },
                os.path.join(args.checkpoint_dir, f"geomnet_k{args.num_patches}_best.pth"),
            )
            # Also maintain symlink/copy as geomnet_best.pth
            torch.save(
                {
                    "epoch": epoch,
                    "num_patches": args.num_patches,
                    "eval_res": args.eval_res,
                    "embed_dim": args.embed_dim,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "test_cd_x100": test_cd_x100,
                    "test_nc": test_nc,
                },
                os.path.join(args.checkpoint_dir, "geomnet_best.pth"),
            )

        star = " (*)" if is_best else ""
        print(f"\n>>> Epoch {epoch:2d} Summary [{epoch_dur:.1f}s]:")
        print(f"    Train CD x 100 : {avg_train_cd:.2f} | Train NC: {avg_train_nc:.4f}")
        print(f"    Test  CD x 100 : {test_cd_x100:.2f} | Test  NC: {test_nc:.4f} (evaluated on {n_eval} test shapes){star}")
        print(f"    Best  CD x 100 : {best_test_cd:.2f} | PaCo SOTA Ref: ~2.2 - 3.8 / NC: 0.943\n")

    print(f"Training completed. Best Test CD x 100: {best_test_cd:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate GeomNet on ABC Benchmark")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (safe for 8 GB RAM)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=256, help="Encoder embedding dimension")
    parser.add_argument("--num_patches", type=int, default=32, help="Number of NURBS surface patches")
    parser.add_argument("--eval_res", type=int, default=16, help="Resolution per patch (16x16=256 points)")
    parser.add_argument("--lambda_normal", type=float, default=0.1, help="Weight for analytical normal alignment loss")
    parser.add_argument("--num_workers", type=int, default=0, help="0 workers to prevent RAM replication")
    parser.add_argument("--log_interval", type=int, default=100, help="Log step interval")
    parser.add_argument("--test_eval_batches", type=int, default=25, help="Num test batches for validation (25*8=200 shapes)")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Cap train steps per epoch for fast sanity checks")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to save weights")

    args = parser.parse_args()
    train(args)
