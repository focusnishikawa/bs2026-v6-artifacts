#!/usr/bin/env python3
"""Train LightGBM models for bs2026 v5-A/B hierarchical classification.

Stage models:
  stage1: bird / nonbird           (binary,     all samples ~221K)
  stage2: raptor / non_raptor      (binary,     bird only ~108K)
  stage3: 4 raptor orders          (multiclass, raptor only ~11K)
  stage4: Accipitriformes species  (multiclass, Accipitri only ~6.4K, 71 classes)

Phase B-4 hierarchical Stage 4 (case 2):
  stage4a: genus prediction        (multiclass, Accipitri only, ~42 genera)
  stage4b: within-genus species    (multiclass, one model per multi-species genus)

Input: features .npz from extract_features.py / extract_cnn_features_v2.py
Output:
  models/stage{1,2,3,4}.txt        (LightGBM booster text)
  models/stage4a.txt               (4a if --stages 4a)
  models/stage4b_<genus>.txt       (4b per-genus, if --stages 4b --genus NAME)
  models/meta.json                 (class names, params, metrics for 1-4)
  models/meta_4a.json              (4a meta: genus list, species→genus map)
  models/meta_4b_<genus>.json      (4b meta per-genus)

CLI:
  --features PATH           default: data/features_ds_beta.npz
  --output-dir PATH         default: models
  --seed INT                default: 42
  --num-rounds INT          default: 1000
  --early-stopping INT      default: 50
  --stages CSV              default: 1,2,3,4   (tokens: 1,2,3,4,4a,4b)
  --genus NAME              required for --stages 4b
  --class-to-taxon PATH     default: configs/class_to_taxon_v5.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.utils.class_weight import compute_class_weight


PARAMS_BINARY = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 0.1,
    "is_unbalance": True,
    "verbose": -1,
}


def derive_genus_mapping(class_names_s4, class_to_taxon):
    """Build species_idx → genus_idx mapping from class_to_taxon_v5.json.

    Returns:
      genus_names           list[str]   ordered (by genus_taxon_id asc)
      species_to_genus_idx  list[int]   parallel to class_names_s4
    """
    species_to_genus_id = []
    species_to_genus_name = {}
    for sp in class_names_s4:
        entry = class_to_taxon.get(sp)
        if entry is None or "genus_taxon_id" not in entry:
            raise ValueError(
                f"No genus_taxon_id for species '{sp}' in class_to_taxon")
        species_to_genus_id.append(entry["genus_taxon_id"])
        species_to_genus_name[entry["genus_taxon_id"]] = entry["genus_name"]

    unique_genera = sorted(set(species_to_genus_id))
    genus_id_to_idx = {gid: i for i, gid in enumerate(unique_genera)}
    genus_names = [species_to_genus_name[g] for g in unique_genera]
    species_to_genus_idx = [genus_id_to_idx[gid] for gid in species_to_genus_id]
    return genus_names, species_to_genus_idx, unique_genera


def filter_to_genus(class_names_s4, class_to_taxon, genus_name):
    """Return (within_class_names, within_to_orig_s4_idx)."""
    in_genus_orig_idx = []
    in_genus_names = []
    for i, sp in enumerate(class_names_s4):
        entry = class_to_taxon.get(sp)
        if entry and entry.get("genus_name") == genus_name:
            in_genus_orig_idx.append(i)
            in_genus_names.append(sp)
    if not in_genus_orig_idx:
        raise ValueError(f"No species found for genus '{genus_name}'")
    return in_genus_names, in_genus_orig_idx


def make_params_multi(num_class, min_leaf=5):
    return {
        "objective": "multiclass",
        "num_class": num_class,
        "metric": ["multi_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": min_leaf,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 0.2,
        "verbose": -1,
    }


def train_stage(name, X, y, params, seed=42, num_rounds=1000, es_rounds=50,
                class_weight=None):
    print(f"\n=== Training {name} ===", file=sys.stderr)
    print(f"  samples: {len(X)}  features: {X.shape[1]}", file=sys.stderr)
    uniq, cnt = np.unique(y, return_counts=True)
    print(f"  class distribution: {dict(zip(uniq.tolist(), cnt.tolist()))}",
          file=sys.stderr)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)

    sample_weight_train = None
    if class_weight == "balanced":
        cw = compute_class_weight("balanced",
                                   classes=uniq.astype(np.int32),
                                   y=y_train)
        class_to_w = {int(c): float(w) for c, w in zip(uniq, cw)}
        sample_weight_train = np.array(
            [class_to_w[int(c)] for c in y_train], dtype=np.float64)
        print(f"  class_weight=balanced applied. "
              f"weight min/max/mean={sample_weight_train.min():.3f}/"
              f"{sample_weight_train.max():.3f}/{sample_weight_train.mean():.3f}",
              file=sys.stderr)

    d_train = lgb.Dataset(X_train, label=y_train, weight=sample_weight_train)
    d_val = lgb.Dataset(X_val, label=y_val, reference=d_train)

    t0 = time.time()
    booster = lgb.train(
        params,
        d_train,
        num_boost_round=num_rounds,
        valid_sets=[d_train, d_val],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(es_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    elapsed = time.time() - t0
    print(f"  train time: {elapsed:.1f}s  best_iter={booster.best_iteration}",
          file=sys.stderr)

    # eval
    if params["objective"] == "binary":
        pred_val = booster.predict(X_val)
        pred_bin = (pred_val >= 0.5).astype(int)
        acc = accuracy_score(y_val, pred_bin)
        print(f"  val acc: {acc:.4f}", file=sys.stderr)
        metrics = {"val_acc": float(acc)}
    else:
        pred_val = booster.predict(X_val)
        pred_cls = pred_val.argmax(axis=1)
        acc = accuracy_score(y_val, pred_cls)
        n_classes = params["num_class"]
        k = min(5, n_classes)
        # top_k_accuracy_score is unreliable for n_classes==2 (treats as binary)
        # and meaningless (top-2 = 1.0 always). Skip and use acc only.
        if n_classes >= 3:
            top5 = top_k_accuracy_score(y_val, pred_val, k=k,
                                        labels=list(range(n_classes)))
            print(f"  val acc: {acc:.4f}  top-{k}: {top5:.4f}",
                  file=sys.stderr)
            metrics = {"val_acc": float(acc), f"val_top{k}": float(top5)}
        else:
            print(f"  val acc: {acc:.4f}  (n_classes={n_classes}, "
                  f"skip top-k)", file=sys.stderr)
            metrics = {"val_acc": float(acc)}

    metrics["train_time_sec"] = float(elapsed)
    metrics["best_iteration"] = int(booster.best_iteration)
    metrics["n_samples"] = int(len(X))
    metrics["n_train"] = int(len(X_train))
    metrics["n_val"] = int(len(X_val))
    metrics["class_weight"] = class_weight if class_weight else "none"
    return booster, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_ds_beta.npz")
    ap.add_argument("--output-dir", default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-rounds", type=int, default=1000)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--stages", default="1,2,3,4",
                    help="Tokens: 1,2,3,4,4a,4b (comma-separated)")
    ap.add_argument("--genus", default="",
                    help="Required when --stages contains 4b")
    ap.add_argument("--class-to-taxon",
                    default="configs/class_to_taxon_v5.json",
                    help="For 4a/4b: source of genus info per species")
    ap.add_argument("--num-threads", type=int, default=0,
                    help="LightGBM num_threads (0=auto)")
    ap.add_argument("--class-weight", default="none",
                    choices=["none", "balanced"],
                    help="Apply class_weight to multiclass stages (3,4). "
                         "Binary stages (1,2) already use is_unbalance=True.")
    args = ap.parse_args()
    cw = None if args.class_weight == "none" else args.class_weight

    # Parse stages: ints (1,2,3,4) and special tokens (4a,4b)
    stages = set()
    do_4a = False
    do_4b = False
    for tok in args.stages.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "4a":
            do_4a = True
        elif tok == "4b":
            do_4b = True
        else:
            stages.add(int(tok))

    if do_4b and not args.genus:
        print("ERROR: --genus required when --stages contains 4b",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.features}", file=sys.stderr)
    data = np.load(args.features, allow_pickle=True)
    X = data["X"]
    y_s1 = data["y_s1"]
    y_s2 = data["y_s2"]
    y_s3 = data["y_s3"]
    y_s4 = data["y_s4"]
    class_s3 = data["class_names_s3"].tolist()
    class_s4 = data["class_names_s4"].tolist()
    print(f"X={X.shape}  y_s1={y_s1.shape}", file=sys.stderr)

    results = {}

    if 1 in stages:
        b, m = train_stage("stage1_bird_nonbird",
                           X, y_s1.astype(np.int32),
                           PARAMS_BINARY, seed=args.seed,
                           num_rounds=args.num_rounds,
                           es_rounds=args.early_stopping)
        b.save_model(str(out_dir / "stage1.txt"),
                     num_iteration=b.best_iteration)
        results["stage1"] = m

    if 2 in stages:
        mask = y_s2 >= 0
        b, m = train_stage("stage2_raptor_nonraptor",
                           X[mask], y_s2[mask].astype(np.int32),
                           PARAMS_BINARY, seed=args.seed,
                           num_rounds=args.num_rounds,
                           es_rounds=args.early_stopping)
        b.save_model(str(out_dir / "stage2.txt"),
                     num_iteration=b.best_iteration)
        results["stage2"] = m

    if 3 in stages:
        mask = y_s3 >= 0
        params_m = make_params_multi(len(class_s3), min_leaf=10)
        b, m = train_stage("stage3_order",
                           X[mask], y_s3[mask].astype(np.int32),
                           params_m, seed=args.seed,
                           num_rounds=args.num_rounds,
                           es_rounds=args.early_stopping,
                           class_weight=cw)
        b.save_model(str(out_dir / "stage3.txt"),
                     num_iteration=b.best_iteration)
        results["stage3"] = m

    if 4 in stages:
        mask = y_s4 >= 0
        n_classes = len(class_s4)
        params_m = make_params_multi(n_classes, min_leaf=3)
        b, m = train_stage("stage4_accipiter_species",
                           X[mask], y_s4[mask].astype(np.int32),
                           params_m, seed=args.seed,
                           num_rounds=args.num_rounds,
                           es_rounds=args.early_stopping,
                           class_weight=cw)
        b.save_model(str(out_dir / "stage4.txt"),
                     num_iteration=b.best_iteration)
        results["stage4"] = m

    # === Phase B-4 hierarchical Stage 4 ===
    if do_4a or do_4b:
        ct_path = Path(args.class_to_taxon)
        if not ct_path.exists():
            print(f"ERROR: --class-to-taxon not found: {ct_path}",
                  file=sys.stderr)
            sys.exit(1)
        class_to_taxon = json.loads(ct_path.read_text())

    if do_4a:
        mask = y_s4 >= 0
        y_s4_arr = y_s4[mask].astype(np.int32)
        genus_names, species_to_genus_idx, unique_genera = derive_genus_mapping(
            class_s4, class_to_taxon)
        n_genera = len(genus_names)
        print(f"\n[4a] {n_genera} genera derived from {len(class_s4)} species",
              file=sys.stderr)
        # Map per-sample y_s4 (species idx) -> genus idx
        y_s4a = np.array([species_to_genus_idx[int(s)] for s in y_s4_arr],
                         dtype=np.int32)
        params_m = make_params_multi(n_genera, min_leaf=5)
        if args.num_threads > 0:
            params_m["num_threads"] = args.num_threads
        b, m = train_stage("stage4a_genus",
                           X[mask], y_s4a, params_m,
                           seed=args.seed,
                           num_rounds=args.num_rounds,
                           es_rounds=args.early_stopping,
                           class_weight=cw)
        b.save_model(str(out_dir / "stage4a.txt"),
                     num_iteration=b.best_iteration)
        meta_4a = {
            "stage": "4a",
            "n_genera": n_genera,
            "genus_names": genus_names,
            "genus_taxon_ids": unique_genera,
            "class_names_s4": class_s4,
            "species_idx_to_genus_idx": species_to_genus_idx,
            "metrics": m,
            "params": params_m,
        }
        with (out_dir / "meta_4a.json").open("w") as f:
            json.dump(meta_4a, f, indent=2, ensure_ascii=False)
        results["stage4a"] = m

    if do_4b:
        within_class_names, within_to_orig = filter_to_genus(
            class_s4, class_to_taxon, args.genus)
        n_within = len(within_class_names)
        print(f"\n[4b] genus='{args.genus}' has {n_within} species: "
              f"{within_class_names}", file=sys.stderr)
        if n_within <= 1:
            print(f"  SKIP training: single-species genus", file=sys.stderr)
            meta_4b = {
                "stage": "4b",
                "genus": args.genus,
                "skipped": True,
                "reason": "single_species_genus",
                "n_within_classes": n_within,
                "within_class_names": within_class_names,
                "within_to_orig_s4_idx": within_to_orig,
            }
            with (out_dir / f"meta_4b_{args.genus}.json").open("w") as f:
                json.dump(meta_4b, f, indent=2, ensure_ascii=False)
        else:
            mask_acc = y_s4 >= 0
            y_s4_arr = y_s4[mask_acc].astype(np.int32)
            in_set = set(within_to_orig)
            sample_mask_within_acc = np.array(
                [int(y) in in_set for y in y_s4_arr], dtype=bool)
            orig_to_within = {orig: w for w, orig in enumerate(within_to_orig)}
            y_within = np.array(
                [orig_to_within[int(y)]
                 for y in y_s4_arr[sample_mask_within_acc]],
                dtype=np.int32)
            X_acc = X[mask_acc]
            X_g = X_acc[sample_mask_within_acc]
            print(f"  samples for 4b: {len(X_g)}", file=sys.stderr)
            # min_leaf adaptive: smaller leaves for small genera
            min_leaf = 2 if n_within <= 3 else 3
            params_m = make_params_multi(n_within, min_leaf=min_leaf)
            if args.num_threads > 0:
                params_m["num_threads"] = args.num_threads
            b, m = train_stage(f"stage4b_{args.genus}",
                               X_g, y_within, params_m,
                               seed=args.seed,
                               num_rounds=args.num_rounds,
                               es_rounds=args.early_stopping,
                               class_weight=cw)
            b.save_model(str(out_dir / f"stage4b_{args.genus}.txt"),
                         num_iteration=b.best_iteration)
            meta_4b = {
                "stage": "4b",
                "genus": args.genus,
                "skipped": False,
                "n_within_classes": n_within,
                "within_class_names": within_class_names,
                "within_to_orig_s4_idx": within_to_orig,
                "metrics": m,
                "params": params_m,
            }
            with (out_dir / f"meta_4b_{args.genus}.json").open("w") as f:
                json.dump(meta_4b, f, indent=2, ensure_ascii=False)
            results[f"stage4b_{args.genus}"] = m

    meta = {
        "features": args.features,
        "feature_dim": int(X.shape[1]),
        "seed": args.seed,
        "num_rounds": args.num_rounds,
        "early_stopping": args.early_stopping,
        "class_weight": args.class_weight,
        "stages": results,
        "class_names_s3": class_s3,
        "class_names_s4": class_s4,
        "params_binary": PARAMS_BINARY,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nSaved meta to {out_dir / 'meta.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
