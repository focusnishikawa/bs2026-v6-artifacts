#!/usr/bin/env python3
"""Collect a held-out non-bird ground-truth set for Stage 1 evaluation.

Rationale
---------
The 840-image GT of the v6 paper contains only birds, so Stage 1 (bird /
non-bird) could not be evaluated as an accuracy at all - only bird recall was
measurable.  This script builds a non-bird GT that

  * mirrors the composition of the 20,000 non-bird training images
    (Plants 49.8 %, Insects 30.1 %, Fungi 3.9 %, ... ),
  * is guaranteed disjoint from those training images by content MD5,
  * excludes Aves explicitly (``without_taxon_id=3``) so that no bird can slip
    in through the ``Animalia`` catch-all iconic taxon,
  * records per-photo licence and attribution, which the paper needs anyway.

Only CC-BY / CC-BY-NC research-grade observations are used.

Usage
-----
    python3 scripts/collect_gt_nonbird.py --total 840
    python3 scripts/collect_gt_nonbird.py --total 40 --dry-run
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_OBS = "https://api.inaturalist.org/v1/observations"
# iNaturalist asks API clients to identify themselves with a contact address.
# Override with BS2026_CONTACT if you run this yourself.
USER_AGENT = "bs2026-nonbird-gt-collector/0.1 ({})".format(
    os.environ.get("BS2026_CONTACT", "nishikawa@j-focus.or.jp"))
AVES_TAXON_ID = 3

# iconic taxon name -> (output subdir, share of the training non-bird mix)
ICONIC = [
    ("Plantae",        "inat_Plants",            9959),
    ("Insecta",        "inat_Insects",           6028),
    ("Fungi",          "inat_Fungi",              782),
    ("Reptilia",       "inat_Reptiles",           759),
    ("Mammalia",       "inat_Mammals",            534),
    ("Actinopterygii", "inat_Ray-finned_Fishes",  445),
    ("Mollusca",       "inat_Mollusks",           409),
    ("Amphibia",       "inat_Amphibians",         393),
    ("Arachnida",      "inat_Arachnids",          360),
    ("Animalia",       "inat_Animalia",           328),
]


def http_get_json(url, timeout=30.0, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    return None


def http_get_bytes(url, timeout=60.0, retries=2):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                return None
            time.sleep(2.0)
    return None


def to_original_url(u):
    if not u:
        return ""
    for size in ("square", "thumb", "small", "medium", "large"):
        u = u.replace(f"/{size}.", "/original.")
    return u


def training_hashes(train_root):
    """MD5 of every non-bird training image, for disjointness enforcement."""
    hashes = set()
    n = 0
    for dirpath, _, files in os.walk(train_root):
        for fn in files:
            if fn.startswith("."):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, "rb") as f:
                    hashes.add(hashlib.md5(f.read()).hexdigest())
                n += 1
            except OSError:
                continue
    return hashes, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=840,
                    help="total non-bird GT images to collect")
    ap.add_argument("--out-dir", default="data/gt_nonbird_v6")
    ap.add_argument("--manifest", default="data/gt_nonbird_v6_manifest.csv")
    ap.add_argument("--train-nonbird-root",
                    default="data/ds_beta_bird/nonbird")
    ap.add_argument("--delay-sec", type=float, default=1.0)
    ap.add_argument("--per-page", type=int, default=100)
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print(f"hashing training non-bird images under {args.train_nonbird_root} ...",
          flush=True)
    seen, n_train = training_hashes(args.train_nonbird_root)
    print(f"  {n_train} training images, {len(seen)} distinct hashes",
          flush=True)

    share_total = sum(w for _, _, w in ICONIC)
    quotas = {name: round(w / share_total * args.total) for name, _, w in ICONIC}
    # fix rounding drift onto the largest bucket
    drift = args.total - sum(quotas.values())
    quotas["Plantae"] += drift
    print("quotas:", quotas, "sum:", sum(quotas.values()), flush=True)

    manifest_path = Path(args.manifest)
    manifest_rows = []
    grand = 0

    for iconic, subdir, _ in ICONIC:
        want = quotas[iconic]
        if want <= 0:
            continue
        got = 0
        page = 1
        print(f"\n=== {iconic} -> {subdir} (target {want}) ===", flush=True)
        while got < want and page <= args.max_pages:
            q = urllib.parse.urlencode({
                "iconic_taxa": iconic,
                "without_taxon_id": AVES_TAXON_ID,
                "photo_license": "cc-by,cc-by-nc",
                "quality_grade": "research",
                "photos": "true",
                "per_page": args.per_page,
                "page": page,
                "order": "desc",
                "order_by": "created_at",
            })
            try:
                data = http_get_json(f"{API_OBS}?{q}")
            except Exception as e:                       # noqa: BLE001
                print(f"  query failed page={page}: {e}", file=sys.stderr)
                break
            results = (data or {}).get("results") or []
            if not results:
                break
            time.sleep(args.delay_sec)

            for obs in results:
                if got >= want:
                    break
                # belt and braces: never accept anything under Aves
                anc = obs.get("taxon", {}).get("ancestor_ids") or []
                if AVES_TAXON_ID in anc:
                    continue
                photos = obs.get("photos") or []
                if not photos:
                    continue
                ph = photos[0]
                url = to_original_url(ph.get("url"))
                if not url:
                    continue
                obs_id = obs.get("id")
                dest = out_dir / subdir / f"{obs_id}.jpg"
                if dest.exists():
                    continue
                if args.dry_run:
                    print(f"  [dry] {iconic} obs={obs_id} {url}")
                    got += 1
                    continue
                blob = http_get_bytes(url)
                time.sleep(args.delay_sec)
                if not blob:
                    continue
                h = hashlib.md5(blob).hexdigest()
                if h in seen:
                    print(f"  skip (duplicate of training): obs={obs_id}")
                    continue
                seen.add(h)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(blob)
                got += 1
                grand += 1
                manifest_rows.append({
                    "category": subdir,
                    "iconic_taxa": iconic,
                    "obs_id": obs_id,
                    "photo_id": ph.get("id"),
                    "taxon_id": (obs.get("taxon") or {}).get("id"),
                    "taxon_name": (obs.get("taxon") or {}).get("name"),
                    "license": ph.get("license_code"),
                    "attribution": ph.get("attribution"),
                    "url": url,
                    "md5": h,
                    "file": str(dest),
                })
                if got % 25 == 0:
                    print(f"  {got}/{want}", flush=True)
            page += 1
        print(f"  collected {got}/{want} for {iconic}", flush=True)

    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys())
                               if manifest_rows else ["category"])
            w.writeheader()
            w.writerows(manifest_rows)
        print(f"\nwrote {grand} images and {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
