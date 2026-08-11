#!/usr/bin/env python3
"""Rebuild the hierarchy-label CSVs for the v6 re-experiment, with HPC paths.

Why this exists
---------------
The original ``hierarchy_labels_v5b_full_y.csv`` (151,539 rows) was lost with the
HPC workspace.  It can be reconstructed exactly from two local artefacts:

    data/hierarchy_labels_v5b_full.csv   144,707 rows
  + the train_v5_inat manifest rows not referenced by it   6,900 rows
  = 151,607 rows, of which 68 images are missing on disk   -> 151,539

which is precisely the count reported in the paper.

``partial_y`` (145,669 rows) cannot be reconstructed: it was defined by a
427-label allow list (``v5b_partial_inat_labels.txt``) that was lost as well.
The re-experiment therefore uses ``full_y`` for every stage, and this deviation
is reported in the paper.  The flat-vs-hierarchical comparison is unaffected in
kind because both Stage 4 variants are trained on the same features.

Outputs (paths rewritten to the HPC rebuild root):
    data/hierarchy_labels_v5b_full_y.csv       training set
    data/hierarchy_labels_gt_84_y.csv          840 bird GT
    data/hierarchy_labels_gt_nonbird_y.csv     held-out non-bird GT (if present)
"""
import argparse
import csv
import json
import os
from pathlib import Path

# Roots of the local image corpus. These were the author's paths; override them
# with BS2026_LOCAL_V5 / BS2026_LOCAL_V4GT to run this elsewhere.
LOCAL_V5 = os.environ.get(
    "BS2026_LOCAL_V5", "/Users/afsc0002/work_claude/bs/inaturalist/bs2026_v5")
LOCAL_V4GT = os.environ.get(
    "BS2026_LOCAL_V4GT", "/Users/afsc0002/work_claude/bs/inaturalist/bs2026_v4_gt")
FIELDS = ["image_path", "label", "stage1", "stage2", "stage3", "stage4", "iucn_gt"]


def remap(path, hpc_root):
    """Rewrite a local absolute path to its location under the HPC root."""
    for src, dst in (
        (f"{LOCAL_V5}/data/ds_beta_bird", f"{hpc_root}/data/ds_beta_bird"),
        (f"{LOCAL_V5}/data/train_v5_inat", f"{hpc_root}/data/train_v5_inat"),
        (f"{LOCAL_V5}/data/gt_v5_critical", f"{hpc_root}/data/gt_v5_critical"),
        (f"{LOCAL_V5}/data/gt_nonbird_v6", f"{hpc_root}/data/gt_nonbird_v6"),
        (f"{LOCAL_V4GT}/gt_v4", f"{hpc_root}/data/gt_v4"),
    ):
        if path.startswith(src):
            return dst + path[len(src):]
    return path


def stages_for(label, c2t):
    """Derive the five hierarchy fields from the taxon table."""
    if label == "nonbird":
        return "nonbird", "", "", "", "OTHERS"
    e = c2t.get(label)
    if e is None:
        return None
    s2 = "raptor" if e.get("is_raptor") else "non_raptor"
    s3 = e.get("raptor_order") or ""
    s4 = label if s3 == "Accipitriformes" else ""
    iu = e.get("iucn_status")
    return "bird", s2, s3, s4, (iu if iu in ("CR", "EN", "VU") else "OTHERS")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpc-root",
                    default="/home1/gfsi/ufsi0002/bs2026_v5_rebuild")
    ap.add_argument("--class-to-taxon", default="configs/class_to_taxon_v5.json")
    ap.add_argument("--check-local", action="store_true",
                    help="drop rows whose image is missing on the local disk")
    args = ap.parse_args()

    c2t = json.loads(Path(args.class_to_taxon).read_text())

    # ---- training set: full_y ------------------------------------------
    base = list(csv.DictReader(open("data/hierarchy_labels_v5b_full.csv")))
    referenced = {r["image_path"] for r in base}
    man = list(csv.DictReader(open("data/train_v5_inat_manifest.csv")))
    inat_root = f"{LOCAL_V5}/data/train_v5_inat/"

    out, dropped_missing, dropped_label = [], 0, 0
    for r in base:
        if args.check_local and not os.path.exists(r["image_path"]):
            dropped_missing += 1
            continue
        out.append({**r, "image_path": remap(r["image_path"], args.hpc_root)})

    added = 0
    for m in man:
        local = inat_root + m["file"]
        if local in referenced:
            continue
        st = stages_for(m["label"], c2t)
        if st is None:
            dropped_label += 1
            continue
        if args.check_local and not os.path.exists(local):
            dropped_missing += 1
            continue
        s1, s2, s3, s4, iu = st
        out.append({"image_path": remap(local, args.hpc_root),
                    "label": m["label"], "stage1": s1, "stage2": s2,
                    "stage3": s3, "stage4": s4, "iucn_gt": iu})
        added += 1

    print(f"training full_y: base {len(base)} + added {added} "
          f"- missing {dropped_missing} - unknown-label {dropped_label} "
          f"= {len(out)}")
    write_csv("data/hierarchy_labels_v5b_full_y.csv", out)

    # ---- bird GT (840) --------------------------------------------------
    gt = list(csv.DictReader(open("data/hierarchy_labels_gt_84.csv")))
    write_csv("data/hierarchy_labels_gt_84_y.csv",
              [{**r, "image_path": remap(r["image_path"], args.hpc_root)}
               for r in gt])

    # ---- non-bird GT ----------------------------------------------------
    nb_dir = Path("data/gt_nonbird_v6")
    if nb_dir.is_dir():
        rows = []
        for p in sorted(nb_dir.rglob("*.jpg")):
            local = str(p.resolve())
            rows.append({"image_path": remap(local, args.hpc_root),
                         "label": "nonbird", "stage1": "nonbird",
                         "stage2": "", "stage3": "", "stage4": "",
                         "iucn_gt": "OTHERS"})
        write_csv("data/hierarchy_labels_gt_nonbird_y.csv", rows)
    else:
        print(f"  (skip) {nb_dir} not present yet")


if __name__ == "__main__":
    main()
