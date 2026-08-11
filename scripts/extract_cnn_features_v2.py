#!/usr/bin/env python3
"""bs2026 Phase 4b v2: DataLoader 並列版 CNN 特徴量抽出.

vs v1 (extract_cnn_features.py):
  - DataLoader + num_workers で transform を CPU 並列化
  - GPU が連続的に batch を処理 → 100x 高速化
  - 失敗画像 (PIL/transform エラー) は X から除外し、対応する rows も除外

I/O は pin_memory + non_blocking で GPU 転送を効率化.

CLI:
  --hierarchy-csv PATH         required
  --output PATH                required (.npz)
  --model {mobilenet_v3_small,resnet50}  default: resnet50
  --batch-size INT             default: 128
  --num-workers INT            default: 8 (hpciaiss1 の 48 cores 中 8)
  --device {cpu,cuda,auto}     default: auto
  --target-nonbird INT         default: -1 (= len(bird))
  --seed INT                   default: 42
  --progress-every INT         default: 5000
  --downsample-to INT          default: 0 (off; 0 reproduces the original
                               pipeline byte-for-byte)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
from torchvision import transforms


class HierarchyDataset(Dataset):
    """Yields (idx, tensor, ok_flag). ok_flag=False のときは tensor=zeros."""
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path = self.rows[idx]["image_path"]
        try:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")
                t = self.transform(img_rgb)
            return idx, t, True
        except Exception:
            return idx, torch.zeros(3, 224, 224), False


def _load_checkpoint(model, checkpoint_path):
    """Load a state_dict into `model` (in-place), tolerating common formats.

    Filters out keys whose shape mismatches the target model (e.g., fc with
    a different num_classes after fine-tuning), since extract_features
    discards fc anyway.
    """
    sd = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    # Strip DDP prefix
    sd = {k[len("module."):] if k.startswith("module.") else k: v
          for k, v in sd.items()}
    own = model.state_dict()
    skipped_shape = []
    sd_ok = {}
    for k, v in sd.items():
        if k in own and hasattr(v, "shape") and own[k].shape == v.shape:
            sd_ok[k] = v
        elif k in own:
            skipped_shape.append((k, tuple(v.shape), tuple(own[k].shape)))
    missing, unexpected = model.load_state_dict(sd_ok, strict=False)
    print(f"  ckpt loaded: {checkpoint_path}", file=sys.stderr)
    print(f"    loaded={len(sd_ok)}  missing={len(missing)}  "
          f"unexpected={len(unexpected)}  shape_skipped={len(skipped_shape)}",
          file=sys.stderr)
    if skipped_shape:
        for k, sshape, oshape in skipped_shape[:3]:
            print(f"    skip shape: {k} ckpt={sshape} model={oshape}",
                  file=sys.stderr)
    if missing:
        print(f"    missing[:5]={missing[:5]}", file=sys.stderr)
    if unexpected:
        print(f"    unexpected[:5]={unexpected[:5]}", file=sys.stderr)


def build_model(name, device, checkpoint=None):
    if name == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        if checkpoint:
            _load_checkpoint(m, checkpoint)
        feat_ext = torch.nn.Sequential(
            m.features, m.avgpool, torch.nn.Flatten())
        dim = 576
    elif name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        if checkpoint:
            _load_checkpoint(m, checkpoint)
        feat_ext = torch.nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
            m.avgpool, torch.nn.Flatten())
        dim = 2048
    else:
        raise ValueError(f"Unknown model: {name}")
    feat_ext.to(device).eval()
    return feat_ext, dim


def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hierarchy-csv", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="resnet50",
                    choices=["mobilenet_v3_small", "resnet50"])
    ap.add_argument("--checkpoint", default="",
                    help="Optional fine-tuned state_dict (.pt). "
                         "Loaded with strict=False so head shape mismatch is OK.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--target-nonbird", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=5000)
    ap.add_argument("--downsample-to", type=int, default=0,
                    help="Controlled down-sampling: after the usual "
                         "Resize(256)+CenterCrop(224), reduce the crop to "
                         "S x S and enlarge it back to 224 x 224, so that the "
                         "network sees an image carrying only S x S pixels of "
                         "information. 0 disables it and reproduces the "
                         "original pipeline exactly.")
    args = ap.parse_args()

    device = auto_device() if args.device == "auto" else args.device
    print(f"device: {device}", file=sys.stderr)
    print(f"num_workers: {args.num_workers}  batch_size: {args.batch_size}",
          file=sys.stderr)

    ckpt = args.checkpoint or None
    feat_ext, dim = build_model(args.model, device, checkpoint=ckpt)
    print(f"model: {args.model}  feature_dim: {dim}  "
          f"checkpoint: {ckpt or '(ImageNet pretrain)'}", file=sys.stderr)

    ops = [transforms.Resize(256), transforms.CenterCrop(224)]
    if args.downsample_to > 0:
        # Simulate the operational input: the detector hands the classifier a
        # crop carrying only S x S pixels, which is then enlarged to the network
        # input size. Both directions use bicubic resampling on the PIL image,
        # so the reduction is anti-aliased rather than a naive decimation.
        s = args.downsample_to
        bicubic = transforms.InterpolationMode.BICUBIC
        ops += [transforms.Resize((s, s), interpolation=bicubic),
                transforms.Resize((224, 224), interpolation=bicubic)]
        print(f"downsample: 224 -> {s} -> 224 (bicubic both ways)",
              file=sys.stderr)
    else:
        print("downsample: off (native 224 crop)", file=sys.stderr)
    ops += [transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]
    transform = transforms.Compose(ops)

    with open(args.hierarchy_csv) as f:
        rows = list(csv.DictReader(f))

    bird_rows = [r for r in rows if r["stage1"] == "bird"]
    nonbird_rows = [r for r in rows if r["stage1"] == "nonbird"]
    target = (args.target_nonbird if args.target_nonbird > 0
              else len(bird_rows))
    rng = np.random.default_rng(args.seed)
    if len(nonbird_rows) > target:
        idx = rng.choice(len(nonbird_rows), size=target, replace=False)
        nonbird_rows = [nonbird_rows[i] for i in sorted(idx)]
    sample_rows = bird_rows + nonbird_rows
    print(f"Sampled: {len(sample_rows)}  (bird {len(bird_rows)} + "
          f"nonbird {len(nonbird_rows)})", file=sys.stderr)

    class_s3 = sorted({r["stage3"] for r in sample_rows if r["stage3"]})
    class_s4 = sorted({r["stage4"] for r in sample_rows if r["stage4"]})
    s1_map = {"nonbird": 0, "bird": 1}
    s2_map = {"non_raptor": 0, "raptor": 1}
    s3_map = {c: i for i, c in enumerate(class_s3)}
    s4_map = {c: i for i, c in enumerate(class_s4)}

    ds = HierarchyDataset(sample_rows, transform)
    dl = DataLoader(ds,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=False,
                    pin_memory=(device == "cuda"),
                    persistent_workers=(args.num_workers > 0))

    N = len(sample_rows)
    X_buf = np.zeros((N, dim), dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    t0 = time.time()
    processed = 0
    skipped = 0
    last_log_processed = 0
    for batch_indices, batch_tensors, batch_ok in dl:
        ok_idx_in_batch = batch_ok.nonzero(as_tuple=True)[0]
        if len(ok_idx_in_batch) > 0:
            xb = batch_tensors[ok_idx_in_batch].to(device, non_blocking=True)
            with torch.no_grad():
                feats = feat_ext(xb).cpu().numpy()
            for j, k in enumerate(ok_idx_in_batch.tolist()):
                gi = int(batch_indices[k].item())
                X_buf[gi] = feats[j]
                valid_mask[gi] = True
        skipped += int(len(batch_indices) - len(ok_idx_in_batch))
        processed += int(len(batch_indices))
        if processed - last_log_processed >= args.progress_every:
            el = time.time() - t0
            rate = processed / el if el > 0 else 0
            eta = (N - processed) / rate if rate > 0 else 0
            print(f"  [{processed}/{N}] {el:.0f}s  {rate:.1f}/s  "
                  f"ETA {eta:.0f}s  skipped={skipped}", file=sys.stderr)
            last_log_processed = processed

    # 失敗した行は除外
    keep_idx = np.where(valid_mask)[0]
    X = X_buf[keep_idx]
    kept_rows = [sample_rows[i] for i in keep_idx]

    el = time.time() - t0
    print(f"Done. X={X.shape} skipped={skipped} kept={len(kept_rows)} "
          f"elapsed={el:.0f}s ({len(kept_rows)/el:.1f}/s)", file=sys.stderr)

    y_s1 = np.array([s1_map[r["stage1"]] for r in kept_rows], dtype=np.int8)
    y_s2 = np.array([s2_map.get(r["stage2"], -1) if r["stage2"] else -1
                     for r in kept_rows], dtype=np.int8)
    y_s3 = np.array([s3_map.get(r["stage3"], -1) if r["stage3"] else -1
                     for r in kept_rows], dtype=np.int8)
    y_s4 = np.array([s4_map.get(r["stage4"], -1) if r["stage4"] else -1
                     for r in kept_rows], dtype=np.int16)
    labels = np.array([r["label"] for r in kept_rows])
    iucn_gt = np.array([r["iucn_gt"] for r in kept_rows])
    paths = np.array([r["image_path"] for r in kept_rows])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
        X=X, y_s1=y_s1, y_s2=y_s2, y_s3=y_s3, y_s4=y_s4,
        labels=labels, iucn_gt=iucn_gt, paths=paths,
        class_names_s3=np.array(class_s3),
        class_names_s4=np.array(class_s4))
    print(f"Saved {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
