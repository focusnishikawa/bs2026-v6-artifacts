#!/usr/bin/env python3
"""Hybrid inference for bs2026 v5-B: Stage 1 uses full_r50, Stage 2-4 uses r50_partial.

Rationale (2026-04-25):
  - full_r50 wins on Stage 1 (0.9869) and Stage 5 (0.8679)
  - r50_partial wins on Stage 2-4 (0.9226 / 0.7946 / 0.2308)
  Hybrid = best-of-both. Stage 5 is a lookup from Stage 4 + Stage 1(=nonbird->OTHERS).

Both GT features npz must have identical `paths` order (verified same_order=True).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb


def stage5_lookup(species_label, class_to_taxon):
    if not species_label:
        return "OTHERS"
    entry = class_to_taxon.get(species_label)
    if not entry:
        return "OTHERS"
    iucn = entry.get("iucn_status")
    return iucn if iucn in {"CR", "EN", "VU"} else "OTHERS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-stage1", required=True,
                    help="Models dir providing stage1.txt (e.g. models_v5b_full_r50)")
    ap.add_argument("--models-stage234", required=True,
                    help="Models dir providing stage2/3/4.txt (e.g. models_v5b_r50_partial)")
    ap.add_argument("--features-stage1", required=True,
                    help="Features npz aligned with --models-stage1")
    ap.add_argument("--features-stage234", required=True,
                    help="Features npz aligned with --models-stage234")
    ap.add_argument("--class-to-taxon",
                    default="configs/class_to_taxon_v5.json")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    m1_dir = Path(args.models_stage1)
    m234_dir = Path(args.models_stage234)
    stage1 = lgb.Booster(model_file=str(m1_dir / "stage1.txt"))
    stage2 = lgb.Booster(model_file=str(m234_dir / "stage2.txt"))
    stage3 = lgb.Booster(model_file=str(m234_dir / "stage3.txt"))
    stage4 = lgb.Booster(model_file=str(m234_dir / "stage4.txt"))
    meta234 = json.loads((m234_dir / "meta.json").read_text())
    class_s3 = meta234["class_names_s3"]
    class_s4 = meta234["class_names_s4"]

    f1 = np.load(args.features_stage1, allow_pickle=True)
    f234 = np.load(args.features_stage234, allow_pickle=True)
    X1 = f1["X"]
    X234 = f234["X"]
    paths1 = f1["paths"].tolist()
    paths234 = f234["paths"].tolist()
    if paths1 != paths234:
        print("ERROR: paths mismatch between the two feature npz files.",
              file=sys.stderr)
        sys.exit(2)
    N = len(paths1)
    print(f"Hybrid inferring: N={N}  "
          f"stage1={m1_dir.name}  stage2-4={m234_dir.name}",
          file=sys.stderr)

    class_to_taxon = json.loads(Path(args.class_to_taxon).read_text())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_path", "stage1", "stage2", "stage3", "stage4",
              "stage5_iucn", "conf_s1", "conf_s2", "conf_s3", "conf_s4",
              "top5_s4"]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        t0 = time.time()
        for i, path in enumerate(paths1):
            x1 = X1[i].reshape(1, -1)
            x234 = X234[i].reshape(1, -1)
            # Stage 1 on full_r50 features
            p1 = float(stage1.predict(x1)[0])
            s1 = "bird" if p1 >= 0.5 else "nonbird"
            conf1 = p1 if s1 == "bird" else 1.0 - p1
            row = {
                "image_path": path,
                "stage1": s1, "stage2": "", "stage3": "", "stage4": "",
                "stage5_iucn": "OTHERS",
                "conf_s1": conf1, "conf_s2": "", "conf_s3": "",
                "conf_s4": "", "top5_s4": "",
            }
            if s1 == "nonbird":
                w.writerow(row)
                continue

            # Stage 2-4 on r50_partial features
            p2 = float(stage2.predict(x234)[0])
            s2 = "raptor" if p2 >= 0.5 else "non_raptor"
            conf2 = p2 if s2 == "raptor" else 1.0 - p2
            row["stage2"] = s2
            row["conf_s2"] = conf2
            if s2 == "non_raptor":
                w.writerow(row)
                continue

            probs3 = stage3.predict(x234)[0]
            i3 = int(np.argmax(probs3))
            s3 = class_s3[i3]
            row["stage3"] = s3
            row["conf_s3"] = float(probs3[i3])
            if s3 != "Accipitriformes":
                w.writerow(row)
                continue

            probs4 = stage4.predict(x234)[0]
            i4 = int(np.argmax(probs4))
            s4 = class_s4[i4]
            row["stage4"] = s4
            row["conf_s4"] = float(probs4[i4])
            top5_idx = np.argsort(probs4)[::-1][:5]
            row["top5_s4"] = "|".join(class_s4[k] for k in top5_idx)
            row["stage5_iucn"] = stage5_lookup(s4, class_to_taxon)
            w.writerow(row)

            if (i + 1) % args.progress_every == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{N}] {el:.1f}s  {(i+1)/el:.1f}/s",
                      file=sys.stderr)

    print(f"Saved {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
