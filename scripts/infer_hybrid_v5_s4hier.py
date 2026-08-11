#!/usr/bin/env python3
"""Phase B-4 hybrid inference with hierarchical Stage-4 (4a→4b genus chain).

Pipeline:
  Stage 1   : --models-stage1     (e.g. full_y_cleaned)
  Stage 2-3 : --models-stage23    (e.g. partial_y_cleaned)
  Stage 4   : --models-stage4-hier  (contains stage4a.txt + stage4b_<genus>.txt)

Stage 4 chain (Z-method, cumulative-90%):
  probs_4a = stage4a(x)                      shape (n_genera,)
  Select top-K genera by cumulative prob >= --cumulative-thresh (default 0.9).
  For each selected genus g:
    if multi-species (4b model exists):
        probs_4b = stage4b_g(x)              shape (n_species_in_g,)
        per_species_score = probs_4a[g] * probs_4b[within_idx]
    else (single-species genus):
        per_species_score = probs_4a[g]      (the single species)
  Unselected genera contribute 0 to all their species (hard cutoff).
  Top-1 = argmax, Top-5 = argsort desc[:5] of the per-species score vector.

Notes:
  - When --cumulative-thresh is 1.0 (or negative), all genera are evaluated
    (full Bayesian product, no truncation).
  - When the number of selected species < 5, top-5 is naturally truncated.
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


def load_stage4_hier(model_dir):
    """Load 4a + all 4b boosters and metas from a single directory.

    Returns:
      stage4a            lgb.Booster
      meta_4a            dict
      stage4b_per_genus  dict[genus_name -> lgb.Booster]
      meta_4b_per_genus  dict[genus_name -> dict]
      species_to_genus   list[int]   parallel to class_names_s4
      class_names_s4     list[str]
    """
    model_dir = Path(model_dir)
    stage4a = lgb.Booster(model_file=str(model_dir / "stage4a.txt"))
    meta_4a = json.loads((model_dir / "meta_4a.json").read_text())
    species_to_genus = meta_4a["species_idx_to_genus_idx"]
    class_names_s4 = meta_4a["class_names_s4"]

    stage4b_per_genus = {}
    meta_4b_per_genus = {}
    for genus_name in meta_4a["genus_names"]:
        meta_path = model_dir / f"meta_4b_{genus_name}.json"
        booster_path = model_dir / f"stage4b_{genus_name}.txt"
        if not meta_path.exists():
            # No 4b meta: must be a single-species genus (no training run)
            continue
        meta_4b = json.loads(meta_path.read_text())
        if meta_4b.get("skipped"):
            meta_4b_per_genus[genus_name] = meta_4b
            continue
        if not booster_path.exists():
            print(f"WARN: meta_4b exists but booster missing for {genus_name}",
                  file=sys.stderr)
            continue
        stage4b_per_genus[genus_name] = lgb.Booster(
            model_file=str(booster_path))
        meta_4b_per_genus[genus_name] = meta_4b

    return (stage4a, meta_4a, stage4b_per_genus, meta_4b_per_genus,
            species_to_genus, class_names_s4)


def predict_stage4_hier(x, stage4a, stage4b_per_genus, meta_4a,
                        meta_4b_per_genus, n_species_total,
                        cumulative_thresh):
    """Return (per_species_probs, probs_4a, n_selected_genera)."""
    probs_4a = stage4a.predict(x)[0]  # (n_genera,)
    n_genera = len(probs_4a)
    genus_names = meta_4a["genus_names"]

    # Genus → list of species idx
    species_to_genus = meta_4a["species_idx_to_genus_idx"]
    genus_to_species = {}
    for sp_idx, g_idx in enumerate(species_to_genus):
        genus_to_species.setdefault(g_idx, []).append(sp_idx)

    # Select genera by cumulative threshold
    if cumulative_thresh >= 1.0:
        selected_g = set(range(n_genera))
    else:
        sort_idx = np.argsort(probs_4a)[::-1]
        cum = 0.0
        selected_g = set()
        for gi in sort_idx:
            selected_g.add(int(gi))
            cum += float(probs_4a[gi])
            if cum >= cumulative_thresh:
                break

    probs = np.zeros(n_species_total, dtype=np.float64)
    for g_idx in selected_g:
        sp_indices = genus_to_species.get(g_idx, [])
        if not sp_indices:
            continue
        g_name = genus_names[g_idx]
        p_g = float(probs_4a[g_idx])

        if g_name in stage4b_per_genus:
            booster_4b = stage4b_per_genus[g_name]
            meta_4b = meta_4b_per_genus[g_name]
            probs_4b = booster_4b.predict(x)[0]
            within_to_orig = meta_4b["within_to_orig_s4_idx"]
            for w_idx, orig_idx in enumerate(within_to_orig):
                probs[orig_idx] = p_g * float(probs_4b[w_idx])
        else:
            # No 4b model (single-species genus or skipped):
            # uniform within genus (1 species → all p_g goes to it)
            n_in = len(sp_indices)
            for sp_idx in sp_indices:
                probs[sp_idx] = p_g / n_in

    return probs, probs_4a, len(selected_g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-stage1", required=True)
    ap.add_argument("--models-stage23", required=True)
    ap.add_argument("--models-stage4-hier", required=True,
                    help="Dir containing stage4a.txt + stage4b_<genus>.txt + "
                         "meta_4a.json + meta_4b_<genus>.json")
    ap.add_argument("--features", required=True)
    ap.add_argument("--cumulative-thresh", type=float, default=0.9,
                    help="Top-K genus selection by cumulative prob "
                         "(1.0=all genera, 0.9=Z-method default)")
    ap.add_argument("--class-to-taxon",
                    default="configs/class_to_taxon_v5.json")
    ap.add_argument("--output", default="predictions_s4hier.csv")
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    m1_dir = Path(args.models_stage1)
    m23_dir = Path(args.models_stage23)
    m4_dir = Path(args.models_stage4_hier)

    stage1 = lgb.Booster(model_file=str(m1_dir / "stage1.txt"))
    stage2 = lgb.Booster(model_file=str(m23_dir / "stage2.txt"))
    stage3 = lgb.Booster(model_file=str(m23_dir / "stage3.txt"))
    meta23 = json.loads((m23_dir / "meta.json").read_text())
    class_s3 = meta23["class_names_s3"]

    (stage4a, meta_4a, stage4b_per_genus, meta_4b_per_genus,
     species_to_genus, class_s4) = load_stage4_hier(m4_dir)
    n_species_total = len(class_s4)
    n_4b_models = len(stage4b_per_genus)
    n_genera = len(meta_4a["genus_names"])
    print(f"Stage 4 hier: {n_genera} genera, {n_4b_models} 4b boosters, "
          f"{n_species_total} species",
          file=sys.stderr)

    f = np.load(args.features, allow_pickle=True)
    X = f["X"]
    paths = f["paths"].tolist()
    N = len(paths)
    thresh = args.cumulative_thresh
    print(f"Hybrid+S4-hier: N={N}  S1={m1_dir.name}  S23={m23_dir.name}  "
          f"S4={m4_dir.name}  cum_thresh={thresh}", file=sys.stderr)

    class_to_taxon = json.loads(Path(args.class_to_taxon).read_text())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_path", "stage1", "stage2", "stage3", "stage4",
              "stage5_iucn", "conf_s1", "conf_s2", "conf_s3", "conf_s4",
              "top5_s4", "n_genera_selected", "top1_genus"]

    selected_genera_stats = []
    with out_path.open("w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fields)
        w.writeheader()
        t0 = time.time()
        for i, path in enumerate(paths):
            x = X[i].reshape(1, -1)

            p1 = float(stage1.predict(x)[0])
            s1 = "bird" if p1 >= 0.5 else "nonbird"
            conf1 = p1 if s1 == "bird" else 1.0 - p1
            row = {
                "image_path": path,
                "stage1": s1, "stage2": "", "stage3": "", "stage4": "",
                "stage5_iucn": "OTHERS",
                "conf_s1": conf1, "conf_s2": "", "conf_s3": "",
                "conf_s4": "", "top5_s4": "",
                "n_genera_selected": "", "top1_genus": "",
            }
            if s1 == "nonbird":
                w.writerow(row)
                continue

            p2 = float(stage2.predict(x)[0])
            s2 = "raptor" if p2 >= 0.5 else "non_raptor"
            conf2 = p2 if s2 == "raptor" else 1.0 - p2
            row["stage2"] = s2
            row["conf_s2"] = conf2
            if s2 == "non_raptor":
                w.writerow(row)
                continue

            probs3 = stage3.predict(x)[0]
            i3 = int(np.argmax(probs3))
            s3 = class_s3[i3]
            row["stage3"] = s3
            row["conf_s3"] = float(probs3[i3])
            if s3 != "Accipitriformes":
                w.writerow(row)
                continue

            probs, probs_4a, n_sel = predict_stage4_hier(
                x, stage4a, stage4b_per_genus, meta_4a,
                meta_4b_per_genus, n_species_total, thresh)
            i4 = int(np.argmax(probs))
            s4 = class_s4[i4]
            row["stage4"] = s4
            row["conf_s4"] = float(probs[i4])
            top5_idx = np.argsort(probs)[::-1][:5]
            top5_idx = [int(k) for k in top5_idx if probs[k] > 0]
            row["top5_s4"] = "|".join(class_s4[k] for k in top5_idx)
            row["stage5_iucn"] = stage5_lookup(s4, class_to_taxon)
            row["n_genera_selected"] = n_sel
            row["top1_genus"] = meta_4a["genus_names"][int(np.argmax(probs_4a))]
            selected_genera_stats.append(n_sel)
            w.writerow(row)

            if (i + 1) % args.progress_every == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{N}] {el:.1f}s  {(i+1)/el:.1f}/s",
                      file=sys.stderr)

    if selected_genera_stats:
        arr = np.array(selected_genera_stats)
        print(f"n_genera_selected: mean={arr.mean():.2f} "
              f"min={arr.min()} max={arr.max()} median={np.median(arr):.0f}",
              file=sys.stderr)
    print(f"Saved {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
