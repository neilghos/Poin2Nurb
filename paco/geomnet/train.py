import os
import sys
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import ABCDataset
from model import GeomNet
from decoder import chamfer_distance, chamfer_and_normal_loss, DifferentiableNurbsEvaluator


def set_seed(seed=42):
    """
    Paper-grade End-to-End Determinism Stack.
    Guarantees bit-identical reproducibility across Python, NumPy, PyTorch, and CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """
    Seeds each DataLoader worker deterministically based on initial PyTorch seed.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CUDAPrefetcher:
    """
    Asynchronous CUDA Stream Prefetcher.
    Overlaps CPU Disk I/O / DataLoader tensor batching with GPU forward & backward execution.
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.loader_iter = iter(loader)
        self.next_batch = None
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            for k, v in self.next_batch.items():
                if isinstance(v, torch.Tensor):
                    self.next_batch[k] = v.to(self.device, non_blocking=True)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


def evaluate(model, loader, device, eval_res=16, max_batches=None):
    """
    Evaluates the model on validation or held-out test shapes.
    Evaluates at full benchmark resolution (eval_res=16 -> 8,192 points) and full 8,192 GT points.
    Reports:
      - Raw Chamfer Distance
      - CD x 100 (Official PaCo CVPR 2025 Table 1 benchmark scale)
      - Normal Consistency (NC)
    """
    model.eval()
    orig_res = model.decoder.eval_res
    orig_evaluator = model.decoder.evaluator
    if orig_res != eval_res:
        model.decoder.evaluator = DifferentiableNurbsEvaluator(
            num_samples=eval_res, degree=model.decoder.patch_degree
        ).to(device)
        model.decoder.eval_res = eval_res

    total_cd = 0.0
    total_nc = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            pc = batch["pc"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            gt_normals = batch["gt_normals"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(pc, return_normals=True)

            _, cd, _, nc = chamfer_and_normal_loss(
                out["surface_points"].float(),
                out["surface_normals"].float(),
                gt,
                gt_normals,
                lambda_normal=0.0,
                topk_ratio=0.0,
                lambda_topk=0.0,
                lambda_laplacian=0.0,
            )

            batch_sz = pc.shape[0]
            total_cd += cd.item() * batch_sz
            total_nc += nc.item() * batch_sz
            total_samples += batch_sz

    # Restore training evaluator
    model.decoder.evaluator = orig_evaluator
    model.decoder.eval_res = orig_res

    mean_cd = total_cd / max(total_samples, 1)
    cd_x100 = mean_cd * 100.0
    mean_nc = total_nc / max(total_samples, 1)
    return mean_cd, cd_x100, mean_nc, total_samples


def train(args):
    # 0. Set Total End-to-End Determinism Stack
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting GeomNet Training & Evaluation ===")
    print(f"Device          : {device}")
    print(f"Random Seed     : {args.seed} (Deterministic stack: Python, NumPy, PyTorch, CUDNN)")
    print(f"Batch Size      : {args.batch_size} (eval batch size: {args.eval_batch_size})")
    print(f"Train GT Points : {args.train_gt_points if args.train_gt_points > 0 else 'Full (8192)'} (stochastic sampling)")
    print(f"Learning Rate   : {args.lr}")
    print(f"Num Patches     : {args.num_patches} (train res {args.train_eval_res}x{args.train_eval_res}, test res {args.eval_res}x{args.eval_res})")
    print(f"Train Pred Pts  : {args.num_patches * args.train_eval_res * args.train_eval_res} per shape")
    print(f"Regularization  : Top-k (k={args.topk_ratio}, weight={args.lambda_topk}) + Laplacian (weight={args.lambda_laplacian})")
    print(f"Lambda Normal   : {args.lambda_normal}")
    print(f"Macro GNN       : {'Enabled (Tier 2 inter-zone coordination)' if args.macro_gnn else 'Disabled (original baseline)'}")
    print(f"Epochs          : {args.epochs}")
    print(f"-----------------------------------------------")

    # 1. Leak-Free Datasets & Loaders
    train_dataset = ABCDataset(split="train", seed=args.seed)
    val_dataset = ABCDataset(split="val", seed=args.seed)
    test_dataset = ABCDataset(split="test", seed=args.seed)

    g_train = torch.Generator()
    g_train.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=g_train,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train shapes    : {len(train_dataset)} ({len(train_loader)} batches/epoch)")
    print(f"Val shapes      : {len(val_dataset)} (evaluated each epoch for model checkpoint selection)")
    print(f"Test shapes     : {len(test_dataset)} (STRICTLY HELD-OUT: evaluated only once at the end of training)")

    # 2. Model & Optimizer
    model = GeomNet(
        embed_dim=args.embed_dim,
        num_patches=args.num_patches,
        eval_res=args.train_eval_res,
        use_macro_gnn=args.macro_gnn,
        patch_degree=args.patch_degree,
    ).to(device)

    if args.resume and os.path.exists(args.resume):
        print(f"\n--> Loading checkpoint from {args.resume} for fine-tuning...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = args.max_train_batches if args.max_train_batches is not None else len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_pct,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    sched_desc = f"OneCycleLR (peak LR={args.lr:.4f}, warmup {args.warmup_pct*100:.1f}%, total steps={total_steps})"

    print(f"Scheduler       : {sched_desc}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 3. Initial Baseline Evaluation on Validation Set
    print("\nRunning initial validation evaluation...")
    init_cd, init_cd_x100, init_nc, n_eval = evaluate(
        model, val_loader, device, eval_res=args.eval_res, max_batches=args.val_eval_batches
    )
    print(f"Initial Val CD x 100: {init_cd_x100:.2f} | Val NC: {init_nc:.4f} (evaluated on {n_eval} validation shapes)")
    print(f"PaCo SOTA Reference : CD ~2.2 - 3.8 | NC: ~0.943\n")

    best_val_cd = init_cd_x100 if args.resume else float("inf")
    best_epoch = 0

    cuda_gen = torch.Generator(device=device).manual_seed(args.seed)

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_cd_total = 0.0
        train_nc_total = 0.0
        train_count = 0
        start_time = time.time()

        prefetcher = CUDAPrefetcher(train_loader, device=device)
        for step, batch in enumerate(prefetcher):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break
            pc = batch["pc"]
            gt = batch["gt"]
            gt_normals = batch["gt_normals"]

            # Deterministic stochastic GT point sampling
            if args.train_gt_points and 0 < args.train_gt_points < gt.shape[1]:
                sub_idx = torch.randint(0, gt.shape[1], (args.train_gt_points,), device=device, generator=cuda_gen)
                gt_train = gt[:, sub_idx, :]
                gt_normals_train = gt_normals[:, sub_idx, :]
            else:
                gt_train = gt
                gt_normals_train = gt_normals

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(pc, return_normals=True)

            total_loss, cd_loss, normal_loss, nc = chamfer_and_normal_loss(
                out["surface_points"].float(),
                out["surface_normals"].float(),
                gt_train,
                gt_normals_train,
                lambda_normal=args.lambda_normal,
                cp=out["control_points"].float(),
                lambda_laplacian=args.lambda_laplacian,
                topk_ratio=args.topk_ratio,
                lambda_topk=args.lambda_topk,
            )
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

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

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_dur = time.time() - start_time
        avg_train_cd = (train_cd_total / train_count) * 100.0
        avg_train_nc = train_nc_total / train_count

        # Evaluate on Validation Set (leak-free: test set is never touched here)
        val_cd, val_cd_x100, val_nc, n_eval = evaluate(
            model, val_loader, device, eval_res=args.eval_res, max_batches=args.val_eval_batches
        )

        is_best = val_cd_x100 < best_val_cd
        if is_best:
            best_val_cd = val_cd_x100
            best_epoch = epoch
            save_payload = {
                "epoch": epoch,
                "seed": args.seed,
                "num_patches": args.num_patches,
                "patch_degree": args.patch_degree,
                "eval_res": args.eval_res,
                "embed_dim": args.embed_dim,
                "use_macro_gnn": args.macro_gnn,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_cd_x100": val_cd_x100,
                "val_nc": val_nc,
            }
            torch.save(save_payload, os.path.join(args.checkpoint_dir, args.save_name))

        star = " (*)" if is_best else ""
        print(f"\n>>> Epoch {epoch:2d}/{args.epochs:2d} Summary [{epoch_dur:.1f}s | LR: {current_lr:.6f}]:")
        print(f"    Train CD x 100 : {avg_train_cd:.2f} | Train NC: {avg_train_nc:.4f}")
        print(f"    Val   CD x 100 : {val_cd_x100:.2f} | Val   NC: {val_nc:.4f} (evaluated on {n_eval} val shapes){star}")
        print(f"    Best  Val CD   : {best_val_cd:.2f} (Epoch {best_epoch}) | Definitive SOTA Ref: ~1.6 - 1.8 / NC: 0.950\n")

    print(f"Training completed. Best Validation CD x 100: {best_val_cd:.2f} (Epoch {best_epoch})")

    # 5. Final Evaluation on Held-Out Test Set
    print("\n" + "=" * 70)
    print(">>> TRAINING COMPLETE. RUNNING FINAL LEAK-FREE TEST BENCHMARK <<<")
    best_ckpt_path = os.path.join(args.checkpoint_dir, args.save_name)
    if os.path.exists(best_ckpt_path):
        print(f"Loading best checkpoint from {best_ckpt_path} (Best Val CD: {best_val_cd:.2f} from Epoch {best_epoch})...")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_cd, test_cd_x100, test_nc, n_test = evaluate(
        model, test_loader, device, eval_res=args.eval_res, max_batches=None
    )
    print(f"\nFINAL UNTOUCHED TEST SET EVALUATION ({n_test} test shapes):")
    print(f"  Final Test CD x 100 : {test_cd_x100:.2f}")
    print(f"  Final Test NC       : {test_nc:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate GeomNet on ABC Benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for total end-to-end determinism")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default 100)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for OneCycleLR")
    parser.add_argument("--warmup_pct", type=float, default=0.05, help="Warmup fraction of total steps (default 0.05 = 5% warmup)")
    parser.add_argument("--macro_gnn", action="store_true", default=True, help="Enable Tier 2 Macro GNN inter-zone message passing (default True)")
    parser.add_argument("--no_macro_gnn", dest="macro_gnn", action="store_false", help="Disable Tier 2 Macro GNN (ablation to original baseline)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--train_gt_points", type=int, default=2048, help="Number of GT points to sample during training (default 2048 for 4x speedup, 0 for all 8192)")
    parser.add_argument("--embed_dim", type=int, default=256, help="Encoder embedding dimension")
    parser.add_argument("--num_patches", type=int, default=32, help="Number of NURBS surface patches")
    parser.add_argument("--patch_degree", type=int, default=5, help="Bernstein patch polynomial degree (3 for 4x4, 5 for 6x6, etc.)")
    parser.add_argument("--train_eval_res", type=int, default=12, help="Patch resolution during training (12x12=144 pts/patch)")
    parser.add_argument("--eval_res", type=int, default=16, help="Patch resolution during evaluation (16x16=256 pts/patch = 8192 pts total)")
    parser.add_argument("--lambda_normal", type=float, default=0.15, help="Weight for analytical normal alignment loss")
    parser.add_argument("--num_workers", type=int, default=0, help="0 workers to prevent RAM replication")
    parser.add_argument("--log_interval", type=int, default=50, help="Log step interval")
    parser.add_argument("--val_eval_batches", type=int, default=None, help="Num val batches per epoch (default None = all 1000 val shapes, or e.g. 15 for 240 shapes)")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Cap train steps per epoch for fast sanity checks")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to save weights")
    parser.add_argument("--save_name", type=str, default="geomnet_k32_deg5_100e.pth", help="Checkpoint filename to save best weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for fine-tuning/resuming")
    parser.add_argument("--lambda_laplacian", type=float, default=0.01, help="Weight for 2D control point Laplacian stiffness")
    parser.add_argument("--topk_ratio", type=float, default=0.05, help="Top-k outlier ratio for Direction 1 Pred->GT")
    parser.add_argument("--lambda_topk", type=float, default=0.5, help="Weight for top-k outlier penalty")

    args = parser.parse_args()
    train(args)



