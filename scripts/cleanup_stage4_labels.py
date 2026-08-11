#!/usr/bin/env python3
"""Cleanup Stage 4 class labels: merge duplicate names, drop Looney_Birds.

Merges (old -> canonical, canonical chosen to match GT dir names):
  Akaonosuri_Red-tailed_Hawk  -> Akaonosuri_Red_Tailed_Hawk  (GT: neither; keep larger)
  Red-shouldered_Hawk         -> Red_Shouldered_Hawk         (GT form)
  Rough_Leg_Buzzard           -> Rough-legged_Hawk           (GT form)

Drops:
  Looney_Birds (176 mislabeled images; taxon_id wrongly points to Golden Eagle)

Result: 75 -> 71 Stage 4 classes. X, y_s1/s2/s3 unchanged.
"""
import argparse
import numpy as np
from pathlib import Path


MERGE_MAP = {
    "Akaonosuri_Red-tailed_Hawk": "Akaonosuri_Red_Tailed_Hawk",
    "Red-shouldered_Hawk":        "Red_Shouldered_Hawk",
    "Rough_Leg_Buzzard":          "Rough-legged_Hawk",
}
DROP_SET = {"Looney_Birds"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Input features npz (with X, y_s1/s2/s3/s4, class_names_s3/s4, paths)")
    ap.add_argument("--output", required=True,
                    help="Output features npz (cleaned y_s4, class_names_s4)")
    args = ap.parse_args()

    d = np.load(args.input, allow_pickle=True)
    X = d["X"]
    y_s1 = d["y_s1"]
    y_s2 = d["y_s2"]
    y_s3 = d["y_s3"]
    y_s4 = d["y_s4"]
    class_s3 = d["class_names_s3"]
    class_s4_old = d["class_names_s4"].tolist()
    paths = d["paths"]

    # Build new class list: drop DROP_SET, replace MERGE_MAP keys with canonical
    kept_old = [c for c in class_s4_old if c not in DROP_SET and c not in MERGE_MAP]
    # Ensure canonicals are present
    canonicals = set(MERGE_MAP.values())
    for c in canonicals:
        if c not in kept_old:
            raise RuntimeError(f"canonical class {c!r} not found in old class list")
    class_s4_new = kept_old  # preserves original relative order for canonicals

    # Old index -> new index map (or -1 if dropped)
    new_index = {c: i for i, c in enumerate(class_s4_new)}
    old_to_new = np.full(len(class_s4_old), -1, dtype=np.int32)
    for old_i, c in enumerate(class_s4_old):
        if c in DROP_SET:
            old_to_new[old_i] = -1
        elif c in MERGE_MAP:
            old_to_new[old_i] = new_index[MERGE_MAP[c]]
        else:
            old_to_new[old_i] = new_index[c]

    # Remap y_s4
    y_s4_new = np.full_like(y_s4, -1)
    mask = y_s4 >= 0
    y_s4_new[mask] = old_to_new[y_s4[mask]]
    # Note: samples previously in Looney_Birds now have y_s4_new == -1
    # Their y_s3 was 'Accipitriformes' index — also set y_s3 = -1 for those samples
    # so they don't pollute Stage 3 training either.
    looney_old_idx = class_s4_old.index("Looney_Birds") if "Looney_Birds" in class_s4_old else -1
    drop_mask = (y_s4 == looney_old_idx) if looney_old_idx >= 0 else np.zeros_like(y_s4, dtype=bool)

    y_s3_new = y_s3.copy()
    y_s2_new = y_s2.copy()
    y_s1_new = y_s1.copy()
    y_s3_new[drop_mask] = -1
    # Leave y_s2 (raptor/non_raptor) alone — Looney_Birds was raptor per orig CSV,
    # but since they're likely mislabeled images of unknown content, drop from s2 too.
    y_s2_new[drop_mask] = -1
    # y_s1 (bird/nonbird) — keep as bird=1 (these 176 were all labeled bird; safer
    # to keep them helping Stage 1 unless we're sure they're not bird).
    # Actually since content is unknown and labels are suspect, drop from s1 too.
    y_s1_new[drop_mask] = -1

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output,
             X=X, paths=paths,
             y_s1=y_s1_new, y_s2=y_s2_new, y_s3=y_s3_new, y_s4=y_s4_new,
             class_names_s3=class_s3,
             class_names_s4=np.array(class_s4_new, dtype=object))

    # Report
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Old Stage 4 classes: {len(class_s4_old)}")
    print(f"New Stage 4 classes: {len(class_s4_new)}")
    print(f"Dropped classes: {sorted(DROP_SET)}")
    print(f"Merged: {MERGE_MAP}")
    # Count changes
    old_s4_valid = int((y_s4 >= 0).sum())
    new_s4_valid = int((y_s4_new >= 0).sum())
    dropped_samples = int(drop_mask.sum())
    print(f"Stage 4 valid samples: {old_s4_valid} -> {new_s4_valid} "
          f"(dropped {dropped_samples} Looney_Birds samples)")

    # Per-class sample count change for merged canonicals
    for old_cls, canon in MERGE_MAP.items():
        old_i = class_s4_old.index(old_cls)
        canon_i = class_s4_old.index(canon)
        new_canon_i = new_index[canon]
        old_count = int((y_s4 == old_i).sum())
        canon_old_count = int((y_s4 == canon_i).sum())
        new_count = int((y_s4_new == new_canon_i).sum())
        print(f"  {old_cls} ({old_count}) + {canon} ({canon_old_count}) "
              f"-> {canon} ({new_count})")


if __name__ == "__main__":
    main()
