#!/usr/bin/env python3
"""bs2026 Phase B (case 3): ResNet50 fine-tuning on bird domain (full_y).

Design (per Phase A3 真因対処):
  - Start from ImageNet pretrain (ResNet50_Weights.IMAGENET1K_V2).
  - Replace fc with new linear layer over all unique labels in the
    hierarchy CSV (e.g. 887+1 = bird species + 'nonbird').
  - Fine-tune ONLY:
      * fc          (new head)         lr 1e-4
      * layer4      (last residual blk) lr 1e-5
    Other layers are frozen (requires_grad=False).
  - DDP across 4× H200 NVL on hpciaiss1.
  - AMP (autocast bf16 on H200) for speed.
  - Val split 10%, early-stop on val top-1 (patience=3 epochs).

Output:
  <output-dir>/
    best.pt          (state_dict of full ResNet50; load into tvm.resnet50())
    last.pt          (state_dict at the end)
    finetune_log.csv (per-epoch metrics)
    meta.json        (hyperparams, class list)

Launch:
  torchrun --nproc_per_node=4 scripts/finetune_resnet50_v5b.py \\
      --hierarchy-csv data/hierarchy_labels_v5b_full_y.csv \\
      --output-dir models/resnet50_ft_v5b \\
      --epochs 20 --batch-size 128

Single-GPU fallback (no torchrun): the script auto-detects WORLD_SIZE.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.models as tvm
from torchvision import transforms


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_dist():
    if is_distributed():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_dist():
    if is_distributed():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def log(rank, msg):
    if is_main(rank):
        print(msg, file=sys.stderr, flush=True)


class BirdDataset(Dataset):
    def __init__(self, rows, transform, label_to_idx):
        self.rows = rows
        self.transform = transform
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = r["image_path"]
        try:
            with Image.open(path) as img:
                t = self.transform(img.convert("RGB"))
            return t, self.label_to_idx[r["label"]], True
        except Exception:
            return torch.zeros(3, 224, 224), 0, False


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def split_train_val(rows, val_fraction, seed):
    rng = np.random.default_rng(seed)
    n = len(rows)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_fraction)
    val_idx = sorted(idx[:n_val].tolist())
    train_idx = sorted(idx[n_val:].tolist())
    val_rows = [rows[i] for i in val_idx]
    train_rows = [rows[i] for i in train_idx]
    return train_rows, val_rows


def build_model_finetune(n_classes):
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    # Freeze all
    for p in m.parameters():
        p.requires_grad = False
    # Unfreeze layer4 + replace fc
    for p in m.layer4.parameters():
        p.requires_grad = True
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    for p in m.fc.parameters():
        p.requires_grad = True
    return m


def all_reduce_mean(tensor, world_size):
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


def all_reduce_sum(tensor):
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


@torch.no_grad()
def evaluate(model, loader, device, world_size):
    model.eval()
    correct = torch.zeros(1, device=device)
    total = torch.zeros(1, device=device)
    loss_sum = torch.zeros(1, device=device)
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y, ok in loader:
        ok_mask = ok.to(device)
        if ok_mask.sum() == 0:
            continue
        x = x[ok].to(device, non_blocking=True)
        y = y[ok].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = crit(logits, y)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().float()
        total += y.numel()
        loss_sum += loss
    correct = all_reduce_sum(correct)
    total = all_reduce_sum(total)
    loss_sum = all_reduce_sum(loss_sum)
    if total.item() == 0:
        return 0.0, float("inf")
    acc = (correct / total).item()
    avg_loss = (loss_sum / total).item()
    return acc, avg_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hierarchy-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128,
                    help="Per-GPU batch size")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-layer4", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=3,
                    help="Early-stop patience on val top-1 (epochs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-nonbird", action="store_true",
                    help="If set, drop rows with label='nonbird'")
    ap.add_argument("--log-every", type=int, default=50,
                    help="Log every N steps within an epoch (rank 0 only)")
    args = ap.parse_args()

    rank, world_size, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")

    out_dir = Path(args.output_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    log(rank, f"DDP: rank={rank}/{world_size} local_rank={local_rank} "
              f"device={device}")

    # Load CSV (all ranks; rows are identical)
    with open(args.hierarchy_csv) as f:
        rows = list(csv.DictReader(f))
    if args.exclude_nonbird:
        rows = [r for r in rows if r["label"] != "nonbird"]
    log(rank, f"loaded {len(rows)} rows from {args.hierarchy_csv}")

    # Class index (sorted for determinism)
    labels_sorted = sorted({r["label"] for r in rows})
    label_to_idx = {l: i for i, l in enumerate(labels_sorted)}
    n_classes = len(labels_sorted)
    log(rank, f"n_classes={n_classes}")

    # Train/val split with same seed across ranks
    train_rows, val_rows = split_train_val(
        rows, args.val_fraction, args.seed)
    log(rank, f"train={len(train_rows)} val={len(val_rows)}")

    train_tf, val_tf = build_transforms()
    train_ds = BirdDataset(train_rows, train_tf, label_to_idx)
    val_ds = BirdDataset(val_rows, val_tf, label_to_idx)

    if is_distributed():
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed)
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank,
            shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0), drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0))

    # Model
    model = build_model_finetune(n_classes).to(device)
    if is_distributed():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(rank, f"trainable params: {n_trainable:,}")

    # Optimizer with param groups
    if is_distributed():
        m_for_params = model.module
    else:
        m_for_params = model
    opt = torch.optim.AdamW([
        {"params": m_for_params.fc.parameters(), "lr": args.lr_head},
        {"params": m_for_params.layer4.parameters(), "lr": args.lr_layer4},
    ], weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_epoch = -1
    patience_left = args.patience
    log_path = out_dir / "finetune_log.csv"
    if is_main(rank):
        with log_path.open("w", newline="") as lf:
            csv.writer(lf).writerow(
                ["epoch", "train_loss", "train_acc",
                 "val_loss", "val_acc", "elapsed_sec", "lr_head", "lr_layer4"])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        run_loss = torch.zeros(1, device=device)
        run_correct = torch.zeros(1, device=device)
        run_total = torch.zeros(1, device=device)
        for step, (x, y, ok) in enumerate(train_loader):
            ok_count = int(ok.sum().item())
            if ok_count == 0:
                continue
            x = x[ok].to(device, non_blocking=True)
            y = y[ok].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.detach() * y.numel()
            pred = logits.argmax(dim=1)
            run_correct += (pred == y).sum().float()
            run_total += y.numel()
            if is_main(rank) and (step + 1) % args.log_every == 0:
                el = time.time() - t0
                rate = (run_total.item() / el) if el > 0 else 0.0
                log(rank,
                    f"  ep{epoch} step{step+1}/{len(train_loader)} "
                    f"loss={loss.item():.4f} {rate:.0f} img/s")
        run_loss = all_reduce_sum(run_loss)
        run_correct = all_reduce_sum(run_correct)
        run_total = all_reduce_sum(run_total)
        train_loss = (run_loss / run_total).item() if run_total > 0 else 0.0
        train_acc = (run_correct / run_total).item() if run_total > 0 else 0.0

        val_acc, val_loss = evaluate(model, val_loader, device, world_size)
        elapsed = time.time() - t0
        log(rank, f"ep{epoch}/{args.epochs}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
                  f"{elapsed:.0f}s")
        if is_main(rank):
            with log_path.open("a", newline="") as lf:
                csv.writer(lf).writerow(
                    [epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                     f"{val_loss:.6f}", f"{val_acc:.6f}",
                     f"{elapsed:.1f}",
                     f"{args.lr_head:.2e}", f"{args.lr_layer4:.2e}"])

        # Early stop / best checkpoint
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_left = args.patience
            if is_main(rank):
                state = (model.module if is_distributed() else model).state_dict()
                torch.save(state, out_dir / "best.pt")
                log(rank, f"  ✓ new best val_acc={val_acc:.4f}, saved best.pt")
        else:
            patience_left -= 1
            log(rank, f"  no improvement (patience {patience_left}/{args.patience})")

        # Broadcast stop signal across ranks
        if is_distributed():
            stop_t = torch.tensor([1 if patience_left <= 0 else 0],
                                  device=device)
            dist.broadcast(stop_t, src=0)
            if int(stop_t.item()) == 1:
                log(rank, "early stop triggered")
                break
        else:
            if patience_left <= 0:
                log(rank, "early stop triggered")
                break

    # Save last
    if is_main(rank):
        state = (model.module if is_distributed() else model).state_dict()
        torch.save(state, out_dir / "last.pt")
        meta = {
            "args": vars(args),
            "n_classes": n_classes,
            "labels_sorted": labels_sorted,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "world_size": world_size,
        }
        with (out_dir / "meta.json").open("w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        log(rank, f"Saved last.pt and meta.json. best_val_acc={best_val_acc:.4f} "
                  f"at ep{best_epoch}")

    cleanup_dist()


if __name__ == "__main__":
    main()
