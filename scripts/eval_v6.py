#!/usr/bin/env python3
"""v6 evaluation: correct denominators, cluster bootstrap CIs, and a
pre-specified non-inferiority test for flat vs hierarchical Stage 4.

What this fixes relative to scripts/compare_phase_b.py
------------------------------------------------------
1. Denominators.  The old script set ``n = len(rows)`` before the loop, then
   ``continue``-skipped GT classes absent from the taxon table, and counted
   Stage 2/3 only on eligible samples - yet divided all of them by 840.  Stage 3
   was the worst hit: it counted only true raptors (370) but divided by 840,
   which turned 0.79 into 0.35.  Every stage here uses its own denominator.

2. Stage 1 is a real bird/non-bird accuracy.  The original GT was 100 % birds,
   so only bird recall was measurable.  Passing --nonbird-predictions adds the
   held-out non-bird GT and Stage 1 becomes an accuracy, with the non-bird
   rejection rate reported separately.

3. Uncertainty.  Every headline number gets a species-level cluster bootstrap
   CI, because the GT is 10 images per species and image-level independence is
   not tenable.

4. Multiplicity.  McNemar p-values are Holm-corrected across the family of
   tests actually reported.

5. Equivalence, not "absence of evidence".  A non-significant McNemar test does
   not establish redundancy.  We pre-specify a margin (default 5 pt absolute on
   Stage 4 top-5) and run a paired non-inferiority test plus TOST equivalence
   on the cluster bootstrap distribution.

6. Class balance context.  The always-OTHERS trivial baseline is reported next
   to Stage 5 accuracy, and the structural recall ceiling of each IUCN category
   (VU cannot exceed 40/60 because 20 VU images are not Accipitriformes) is
   printed so the recall numbers cannot be read as if 60/60 were reachable.
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ------------------------------------------------------------------ helpers

BOOT_N = 10000
BOOT_SEED = 20260810
NI_MARGIN = 0.05  # pre-specified, absolute, on Stage 4 top-5 accuracy


def gt_label_from_path(path):
    return Path(path).parent.name


def is_nonbird_path(path):
    return "gt_nonbird" in path or Path(path).parent.name == "nonbird"


def mcnemar_exact(pairs):
    """pairs: iterable of (a_correct, b_correct). Two-sided exact p.

    Retained for reference only: it assumes image-level independence, which the
    ground truth (10 images per species) violates. The reported p-values come
    from cluster_signflip below.
    """
    b = sum(1 for a, c in pairs if a and not c)
    c = sum(1 for a, c in pairs if not a and c)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, p)


def cluster_signflip(clustered_pairs, n_perm=200000, seed=BOOT_SEED):
    """Cluster-robust paired test by sign-flipping whole clusters.

    clustered_pairs: dict cluster -> list of (a_correct, b_correct).

    The statistic is the net discordance summed over clusters,
    S = sum_i (b_i - c_i).  Under the null of exchangeability within a cluster,
    the sign of each cluster's contribution is symmetric, so we obtain the null
    distribution by flipping cluster signs.  With one observation per cluster
    this reduces to the exact McNemar test; with 10 correlated images per
    species it does not, which is the entire point.  Exact enumeration is used
    when the number of informative clusters is small enough.
    """
    contrib = []
    b_tot = c_tot = 0
    for pairs in clustered_pairs.values():
        bi = sum(1 for a, c in pairs if a and not c)
        ci = sum(1 for a, c in pairs if not a and c)
        b_tot += bi
        c_tot += ci
        if bi != ci:
            contrib.append(bi - ci)
    m = len(contrib)
    if m == 0:
        return b_tot, c_tot, 1.0, 0
    obs = abs(sum(contrib))
    if m <= 22:                                  # exact enumeration
        hits = total = 0
        for mask in range(1 << m):
            s = sum(v if (mask >> i) & 1 else -v for i, v in enumerate(contrib))
            total += 1
            if abs(s) >= obs - 1e-12:
                hits += 1
        return b_tot, c_tot, hits / total, m
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        s = sum(v if rng.getrandbits(1) else -v for v in contrib)
        if abs(s) >= obs - 1e-12:
            hits += 1
    return b_tot, c_tot, (hits + 1) / (n_perm + 1), m


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    out = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(idx):
        adj = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, adj)
        out[i] = prev
    return out


def cluster_bootstrap(units, stat, n=BOOT_N, seed=BOOT_SEED):
    """Resample clusters (species) with replacement.

    units: dict cluster -> list of per-image records
    stat:  callable(list_of_records) -> float or None
    """
    keys = list(units)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        picked = [units[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [r for grp in picked for r in grp]
        v = stat(flat)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return lo, hi


def fmt_ci(point, ci):
    lo, hi = ci
    if lo is None:
        return f"{point:.4f}"
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


# ------------------------------------------------------------------ loading

def nonbird_taxon_map(manifest):
    """obs-id -> taxon name, so that non-bird images cluster by taxon.

    Clustering them by output folder would give only 10 clusters (the iconic
    taxa) for 840 images spanning 607 taxa, which would badly overstate the
    within-cluster correlation of the non-bird half of the ground truth.
    """
    m = {}
    try:
        for r in csv.DictReader(open(manifest)):
            m[Path(r["file"]).stem] = r.get("taxon_name") or r.get("taxon_id") or ""
    except OSError:
        pass
    return m


def load_records(pred_csv, c2t, nonbird=False, nb_taxa=None):
    """Return per-image records with GT and prediction fields resolved."""
    nb_taxa = nb_taxa or {}
    recs = []
    for r in csv.DictReader(open(pred_csv)):
        path = r["image_path"]
        if nonbird or is_nonbird_path(path):
            stem = Path(path).stem
            taxon = nb_taxa.get(stem) or gt_label_from_path(path)
            recs.append({
                "path": path, "cluster": "nonbird:" + str(taxon),
                "gt_s1": "nonbird", "gt_s2": None, "gt_s3": None,
                "gt_s4": None, "gt_iucn": "OTHERS", "in_taxon": True,
                "p_s1": r["stage1"], "p_s2": r.get("stage2"),
                "p_s3": r.get("stage3"), "p_s4": r.get("stage4"),
                "p_iucn": r.get("stage5_iucn"),
                "top5": [t for t in (r.get("top5_s4") or "").split("|") if t],
            })
            continue
        lab = gt_label_from_path(path)
        e = c2t.get(lab)
        if e is None:
            # GT class outside the classifier's label space: it still has a
            # true IUCN status of OTHERS and is still a bird, so it belongs in
            # the Stage 1 and Stage 5 denominators. The old script dropped it
            # from the numerator but kept it in the denominator.
            # Asian Dollarbird (Coraciiformes) and Hoopoe (Bucerotiformes):
            # outside the classifier's label space but unambiguously birds and
            # unambiguously non-raptors, so they belong in the Stage 2
            # denominator rather than being dropped from it.
            recs.append({
                "path": path, "cluster": lab, "gt_s1": "bird",
                "gt_s2": "non_raptor",
                "gt_s3": None, "gt_s4": None, "gt_iucn": "OTHERS",
                "in_taxon": False,
                "p_s1": r["stage1"], "p_s2": r.get("stage2"),
                "p_s3": r.get("stage3"), "p_s4": r.get("stage4"),
                "p_iucn": r.get("stage5_iucn"),
                "top5": [t for t in (r.get("top5_s4") or "").split("|") if t],
            })
            continue
        s3 = e.get("raptor_order") or None
        iu = e.get("iucn_status")
        recs.append({
            "path": path, "cluster": lab, "gt_s1": "bird",
            "gt_s2": "raptor" if e.get("is_raptor") else "non_raptor",
            "gt_s3": s3,
            "gt_s4": lab if s3 == "Accipitriformes" else None,
            "gt_iucn": iu if iu in ("CR", "EN", "VU") else "OTHERS",
            "in_taxon": True,
            "p_s1": r["stage1"], "p_s2": r.get("stage2"),
            "p_s3": r.get("stage3"), "p_s4": r.get("stage4"),
            "p_iucn": r.get("stage5_iucn"),
            "top5": [t for t in (r.get("top5_s4") or "").split("|") if t],
        })
    return recs


# ------------------------------------------------------------------ metrics

def acc_s1(recs):
    el = [r for r in recs if r["gt_s1"]]
    return sum(r["p_s1"] == r["gt_s1"] for r in el) / len(el) if el else None


def acc_s2(recs):
    el = [r for r in recs if r["gt_s2"]]
    return sum(r["p_s2"] == r["gt_s2"] for r in el) / len(el) if el else None


def acc_s3(recs):
    el = [r for r in recs if r["gt_s3"]]
    return sum(r["p_s3"] == r["gt_s3"] for r in el) / len(el) if el else None


def acc_s4_top1(recs):
    el = [r for r in recs if r["gt_s4"]]
    return sum(r["p_s4"] == r["gt_s4"] for r in el) / len(el) if el else None


def acc_s4_top5(recs):
    el = [r for r in recs if r["gt_s4"]]
    return sum(r["gt_s4"] in r["top5"] for r in el) / len(el) if el else None


def acc_s5(recs):
    return (sum(r["p_iucn"] == r["gt_iucn"] for r in recs) / len(recs)
            if recs else None)


def recall_cat(recs, cat):
    el = [r for r in recs if r["gt_iucn"] == cat]
    return sum(r["p_iucn"] == cat for r in el) / len(el) if el else None


def precision_cat(recs, cat):
    el = [r for r in recs if r["p_iucn"] == cat]
    return sum(r["gt_iucn"] == cat for r in el) / len(el) if el else None


def macro_recall_s4(recs):
    el = [r for r in recs if r["gt_s4"]]
    per = defaultdict(lambda: [0, 0])
    for r in el:
        per[r["gt_s4"]][1] += 1
        per[r["gt_s4"]][0] += (r["p_s4"] == r["gt_s4"])
    return (sum(a / b for a, b in per.values()) / len(per)) if per else None


def protected_prf(recs, protected):
    """Precision / recall for the 'protected species' alert decision."""
    tp = sum(1 for r in recs if r["gt_s4"] in protected and r["p_s4"] in protected)
    fp = sum(1 for r in recs if r["gt_s4"] not in protected and r["p_s4"] in protected)
    fn = sum(1 for r in recs if r["gt_s4"] in protected and r["p_s4"] not in protected)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return prec, rec, tp, fp, fn


# ------------------------------------------------------------------- report

def clusters_of(recs):
    d = defaultdict(list)
    for r in recs:
        d[r["cluster"]].append(r)
    return d


def evaluate(name, recs, protected, boot):
    units = clusters_of(recs)
    print(f"\n=== {name} ===")
    n_bird = sum(1 for r in recs if r["gt_s1"] == "bird")
    n_nb = sum(1 for r in recs if r["gt_s1"] == "nonbird")
    print(f"  images: {len(recs)}  (bird {n_bird}, non-bird {n_nb})")

    out = {"name": name, "n": len(recs), "n_bird": n_bird, "n_nonbird": n_nb}
    # Each metric is bootstrapped over the clusters it is actually defined on.
    # Resampling all 691 clusters for a Stage 4 statistic that lives on 26
    # Accipitriformes species would dilute the resample and distort the
    # interval.
    rows = [
        ("Stage 1 bird/non-bird acc", acc_s1, lambda r: True),
        ("Stage 2 raptor/non-raptor", acc_s2, lambda r: bool(r["gt_s2"])),
        ("Stage 3 raptor order", acc_s3, lambda r: bool(r["gt_s3"])),
        ("Stage 4 top-1", acc_s4_top1, lambda r: bool(r["gt_s4"])),
        ("Stage 4 top-5", acc_s4_top5, lambda r: bool(r["gt_s4"])),
        ("Stage 4 macro recall", macro_recall_s4, lambda r: bool(r["gt_s4"])),
        ("Stage 5 IUCN acc", acc_s5, lambda r: True),
    ]
    for label, fn, elig in rows:
        v = fn(recs)
        if v is None:
            continue
        sub = clusters_of([r for r in recs if elig(r)])
        ci = cluster_bootstrap(sub, fn, n=boot) if boot else (None, None)
        el = {
            "Stage 1 bird/non-bird acc": len(recs),
            "Stage 2 raptor/non-raptor": sum(1 for r in recs if r["gt_s2"]),
            "Stage 3 raptor order": sum(1 for r in recs if r["gt_s3"]),
            "Stage 4 top-1": sum(1 for r in recs if r["gt_s4"]),
            "Stage 4 top-5": sum(1 for r in recs if r["gt_s4"]),
            "Stage 4 macro recall": sum(1 for r in recs if r["gt_s4"]),
            "Stage 5 IUCN acc": len(recs),
        }[label]
        print(f"  {label:28s} n={el:5d}  {fmt_ci(v, ci)}")
        out[label] = {"value": v, "ci": ci, "n": el}

    if n_nb:
        nb = [r for r in recs if r["gt_s1"] == "nonbird"]
        rej = sum(r["p_s1"] == "nonbird" for r in nb) / len(nb)
        print(f"  {'non-bird rejection rate':28s} n={len(nb):5d}  {rej:.4f}")
        out["nonbird_rejection"] = {"value": rej, "n": len(nb)}

    # trivial baseline for Stage 5
    triv = sum(1 for r in recs if r["gt_iucn"] == "OTHERS") / len(recs)
    print(f"  {'(always-OTHERS baseline)':28s} n={len(recs):5d}  {triv:.4f}")
    out["stage5_trivial_baseline"] = triv

    for cat in ("CR", "EN", "VU"):
        rc, pc = recall_cat(recs, cat), precision_cat(recs, cat)
        tot = sum(1 for r in recs if r["gt_iucn"] == cat)
        reachable = sum(1 for r in recs if r["gt_iucn"] == cat and r["gt_s4"])
        if not tot:
            continue
        hit = sum(1 for r in recs if r["gt_iucn"] == cat and r["p_iucn"] == cat)
        pstr = f"{pc:.4f}" if pc is not None else "n/a"
        print(f"  {cat} recall {hit}/{tot} = {rc:.4f}  "
              f"(structural ceiling {reachable}/{tot})   precision {pstr}")
        out[f"{cat}_recall"] = {"hit": hit, "n": tot, "ceiling": reachable,
                                "value": rc, "precision": pc}

    prec, rec, tp, fp, fn_ = protected_prf(recs, protected)
    if rec is not None:
        print(f"  protected-species alert: precision "
              f"{prec if prec is None else round(prec, 4)}  recall {rec:.4f}  "
              f"(tp={tp} fp={fp} fn={fn_})")
        out["protected"] = {"precision": prec, "recall": rec,
                            "tp": tp, "fp": fp, "fn": fn_}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", action="append", required=True,
                    help="LABEL=path/to/preds.csv (repeatable)")
    ap.add_argument("--nonbird-predictions", action="append", default=[],
                    help="LABEL=path/to/nonbird_preds.csv (repeatable)")
    ap.add_argument("--class-to-taxon", default="configs/class_to_taxon_v5.json")
    ap.add_argument("--nonbird-manifest",
                    default="data/gt_nonbird_v6_manifest.csv")
    ap.add_argument("--protected", default="Inuwashi_Golden_Eagle,"
                    "Ojirowashi_White_Tailed_Eagle,Oowashi_Stellers_Sea_Eagle,"
                    "Kumataka_Mountain_Hawk_Eagle")
    ap.add_argument("--flat-label", default="FT-Flat")
    ap.add_argument("--hier-label", default="FT-Hier")
    ap.add_argument("--ni-margin", type=float, default=NI_MARGIN)
    ap.add_argument("--bootstrap", type=int, default=BOOT_N)
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    c2t = json.loads(Path(args.class_to_taxon).read_text())
    protected = set(args.protected.split(","))
    nb_taxa = nonbird_taxon_map(args.nonbird_manifest)

    nb_map = {}
    for spec in args.nonbird_predictions:
        lab, path = spec.split("=", 1)
        nb_map[lab] = path

    sets, results = {}, []
    for spec in args.predictions:
        lab, path = spec.split("=", 1)
        recs = load_records(path, c2t, nb_taxa=nb_taxa)
        if lab in nb_map:
            recs += load_records(nb_map[lab], c2t, nonbird=True, nb_taxa=nb_taxa)
        sets[lab] = recs
        results.append(evaluate(lab, recs, protected, args.bootstrap))

    # ---------------- paired tests, Holm-corrected -------------------------
    print("\n=== paired McNemar tests (Holm-corrected) ===")
    tests, labels = [], []

    done_pairs = set()

    def add(a, b, metric):
        if a not in sets or b not in sets:
            return
        if (a, b, metric) in done_pairs:
            return
        done_pairs.add((a, b, metric))
        ra = {r["path"]: r for r in sets[a]}
        rb = {r["path"]: r for r in sets[b]}
        common = [p for p in ra if p in rb and ra[p]["gt_s4"]]
        if metric == "top1":
            pairs = [(ra[p]["p_s4"] == ra[p]["gt_s4"],
                      rb[p]["p_s4"] == rb[p]["gt_s4"]) for p in common]
        elif metric == "top5":
            pairs = [(ra[p]["gt_s4"] in ra[p]["top5"],
                      rb[p]["gt_s4"] in rb[p]["top5"]) for p in common]
        else:
            common = [p for p in ra if p in rb]
            pairs = [(ra[p]["p_iucn"] == ra[p]["gt_iucn"],
                      rb[p]["p_iucn"] == rb[p]["gt_iucn"]) for p in common]
        clustered = defaultdict(list)
        for pth, pr in zip(common, pairs):
            clustered[ra[pth]["cluster"]].append(pr)
        bb, cc, p, ncl = cluster_signflip(clustered)
        _, _, p_img = mcnemar_exact(pairs)
        tests.append(p)
        labels.append((f"{a} -> {b}", metric, bb, cc, len(pairs), ncl, p_img))

    order = list(sets)
    base = order[0]
    for other in order[1:]:
        for metric in ("top1", "top5", "iucn"):
            add(base, other, metric)
    if args.flat_label in sets and args.hier_label in sets:
        for metric in ("top1", "top5", "iucn"):
            add(args.flat_label, args.hier_label, metric)

    adj = holm(tests) if tests else []
    print("  (p = cluster sign-flip test; p_img = image-level McNemar, shown "
          "for reference only)")
    print(f"  {'comparison':28s} {'metric':6s} {'b':>4s} {'c':>4s} "
          f"{'n':>5s} {'clus':>5s} {'p':>10s} {'p_holm':>10s} {'p_img':>10s}")
    for (name, metric, bb, cc, n, ncl, p_img), p, pa in zip(labels, tests, adj):
        print(f"  {name:28s} {metric:6s} {bb:4d} {cc:4d} {n:5d} {ncl:5d} "
              f"{p:10.3e} {pa:10.3e} {p_img:10.3e}")

    # ---------------- non-inferiority / equivalence ------------------------
    if args.flat_label in sets and args.hier_label in sets:
        f, h = sets[args.flat_label], sets[args.hier_label]
        rf = {r["path"]: r for r in f}
        rh = {r["path"]: r for r in h}
        common = [p for p in rf if p in rh and rf[p]["gt_s4"]]
        units = defaultdict(list)
        for p in common:
            units[rf[p]["cluster"]].append(
                (rf[p]["gt_s4"] in rf[p]["top5"],
                 rh[p]["gt_s4"] in rh[p]["top5"]))

        def diff(recs):
            if not recs:
                return None
            return (sum(b for _, b in recs) / len(recs)
                    - sum(a for a, _ in recs) / len(recs))

        point = diff([x for g in units.values() for x in g])
        lo, hi = cluster_bootstrap(units, diff, n=args.bootstrap)
        d = args.ni_margin
        print(f"\n=== flat vs hier, Stage 4 top-5 "
              f"(pre-specified margin delta = {d:.2f}) ===")
        print(f"  difference (hier - flat): {point:+.4f}  "
              f"95% cluster-bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
        ni = lo > -d
        eq = (lo > -d) and (hi < d)
        print(f"  non-inferiority (CI lower > -delta): "
              f"{'PASS' if ni else 'FAIL'}")
        print(f"  equivalence (CI within +/-delta)  : "
              f"{'PASS' if eq else 'FAIL'}")
        print("  interpretation: only a PASS licenses the word 'redundant'; a "
              "non-significant McNemar test alone does not.")
        results.append({"name": "flat_vs_hier_top5", "diff": point,
                        "ci": [lo, hi], "margin": d,
                        "non_inferior": ni, "equivalent": eq})

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str))
        print(f"\nSaved {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
