# bs2026-v6-artifacts

> **This repository is archived (read-only).** Its contents and full history were merged on 2026-11-04 into
> [focusnishikawa/bs2026-project](https://github.com/focusnishikawa/bs2026-project) under `papers/bs2026-v6-artifacts/`.
> The release assets and the Zenodo DOIs listed below remain valid and unchanged.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21894830.svg)](https://doi.org/10.5281/zenodo.21894830)

Reproducibility artifacts for the paper:

> **IUCN Red List Threatened Bird Classification from Citizen-Science Images
> using Fine-Tuned CNN Features and a Hierarchical Lightweight Classifier**
> Takeshi Nishikawa (Foundation for Computational Science, FOCUS), 2026.

This repository contains the trained model weights, the prediction files behind
**every table in the paper**, and the evaluation script that computes the
denominators, cluster bootstrap confidence intervals, cluster sign-flip tests,
Holm correction and the pre-specified non-inferiority test.

The intent is that every number in the paper can be **recomputed rather than
taken on trust**. The command in [Reproducing the paper's numbers](#reproducing-the-papers-numbers)
regenerates `results/eval_v6_final.json` byte-for-byte.

---

## Contents

| Path | What it is |
|---|---|
| `scripts/eval_v6.py` | Evaluation script. Standard library only. Produces every reported number. |
| `scripts/run_v6_*.sh` | HPC job scripts: fine-tuning, feature extraction, LightGBM training, inference. Paths inside assume the cluster working directory, so treat them as the documented procedure rather than turnkey scripts. |
| `scripts/finetune_resnet50_v5b.py` | ResNet50 fine-tuning (DDP, bf16). Produces the released checkpoint. |
| `scripts/extract_cnn_features_v2.py` | 2048-dimensional feature extraction. |
| `scripts/train_hierarchy_lgb.py`, `scripts/cleanup_stage4_labels.py` | LightGBM stage training for the flat and hierarchical arms. |
| `scripts/infer_hybrid_v5.py`, `scripts/infer_hybrid_v5_s4hier.py` | Cascade inference producing the prediction CSVs in `results/`. |
| `scripts/build_full_y_labels_v6.py` | Rebuilds the 151,539-row training label CSV from local images. |
| `scripts/collect_gt_nonbird.py` | Collects the 840-image non-bird ground truth from iNaturalist. |
| `configs/class_to_taxon_v5.json` | Class name to iNaturalist taxon ID mapping. |
| `configs/resnet50_ft_v6_meta.json` | Fine-tuning arguments, the 895 sorted class labels, best validation accuracy. |
| `results/v6_40k_*.csv` | **Predictions behind the paper's numbers** (converged, 40,000-round models). |
| `results/v6IN_flat_*.csv` | Predictions for the ImageNet-pretrained baseline (IN-Base). |
| `results/v6_flat_*.csv`, `results/v6_hier_*.csv` | Predictions from the earlier 10,000-round run, kept for the convergence check in RESULTS section 4. |
| `results/*_ds32.csv`, `results/*_ds128.csv` | Predictions for the controlled down-sampling study (RESULTS section 7): same models, ground truth reduced to 32x32 or 128x128 and enlarged back. |
| `results/eval_v6_ds_{in,flat,hier}.json` | Evaluation output of that study, one file per arm (224 vs 128 vs 32). |
| `scripts/run_v6_downsample.sh` | Job script for the down-sampling study: re-extracts ground-truth features at the reduced resolutions and re-runs inference. Retrains nothing. |
| `results/eval_v6_final.json` | **The authoritative evaluation output.** |
| `results/finetune_log.csv`, `results/rank_dist_v6.json` | Inputs for Figures 1 and 3. |
| `manifests/gt_bird_840_manifest.csv` | Per-photograph attribution for the 840 bird evaluation images. |
| `manifests/gt_nonbird_840_manifest.csv` | Per-photograph attribution for the 840 non-bird evaluation images. |
| `manifests/train_inat_manifest.csv` | Per-photograph attribution for the iNaturalist training images. |
| `figs/gen_v6_figs.py` | Regenerates Figures 1, 3 and 4 from `results/`. |
| `RESULTS_v6_rebuild.md` | Confirmed result tables and the reproduction procedure (Japanese). |

Model weights are **not** stored in the git tree because they exceed GitHub's
100 MB per-file limit. They are attached to the release instead.

## Archived version of record

Everything here, **including the model weights**, is archived on Zenodo with a
DOI. Cite the concept DOI, which always resolves to the latest version.

| | DOI |
|---|---|
| Concept (always latest) | [10.5281/zenodo.21894830](https://doi.org/10.5281/zenodo.21894830) |
| Version 1.0.1 (weights included) | [10.5281/zenodo.21894870](https://doi.org/10.5281/zenodo.21894870) |

The Zenodo record is the durable copy. This git repository and its release
assets are the working copy.

## Model weights

Download `resnet50_ft_v6_best.pt` from
[Zenodo](https://doi.org/10.5281/zenodo.21894830) or from the
[latest GitHub release](https://github.com/focusnishikawa/bs2026-v6-artifacts/releases/latest).
Both are the same file; verify with the MD5 below.

| Property | Value |
|---|---|
| File | `resnet50_ft_v6_best.pt` |
| Size | 101,667,110 bytes (96.96 MiB) |
| MD5 | `573d20215bfd92a2e6246b6b6e68eab1` |
| Architecture | ResNet50, classification head replaced with 895 outputs |
| Best validation top-1 | 0.8482 (epoch 19 of 20) |

```bash
curl -L -o resnet50_ft_v6_best.pt \
  https://github.com/focusnishikawa/bs2026-v6-artifacts/releases/latest/download/resnet50_ft_v6_best.pt
md5sum resnet50_ft_v6_best.pt   # expect 573d20215bfd92a2e6246b6b6e68eab1
```

Class labels are in `configs/resnet50_ft_v6_meta.json` under `labels_sorted`,
in the index order used by the checkpoint's output layer. See
[MODEL_CARD.md](MODEL_CARD.md) for training conditions and known limitations.

## Reproducing the paper's numbers

Requires Python 3.8 or newer. No third-party packages are needed for the
evaluation; `figs/gen_v6_figs.py` additionally needs `matplotlib` and `numpy`.

```bash
python3 scripts/eval_v6.py \
  --predictions "IN-Base=results/v6IN_flat_gt_bird.csv" \
  --predictions "FT-Flat=results/v6_40k_flat_gt_bird.csv" \
  --predictions "FT-Hier=results/v6_40k_hier_gt_bird.csv" \
  --nonbird-predictions "IN-Base=results/v6IN_flat_gt_nonbird.csv" \
  --nonbird-predictions "FT-Flat=results/v6_40k_flat_gt_nonbird.csv" \
  --nonbird-predictions "FT-Hier=results/v6_40k_hier_gt_nonbird.csv" \
  --class-to-taxon configs/class_to_taxon_v5.json \
  --nonbird-manifest manifests/gt_nonbird_840_manifest.csv \
  --output-json /tmp/eval_repro.json
```

Runtime is roughly 35 seconds on a laptop. The bootstrap and permutation tests
are seeded (`BOOT_SEED = 20260810`, 10,000 bootstrap resamples, 200,000
sign-flip permutations), so `/tmp/eval_repro.json` is identical to
`results/eval_v6_final.json`:

```bash
python3 -c "import json;a=json.load(open('results/eval_v6_final.json'));b=json.load(open('/tmp/eval_repro.json'));print('identical:',a==b)"
```

Retraining from scratch requires the image corpus and a GPU cluster; the job
scripts in `scripts/` and section 6 of `RESULTS_v6_rebuild.md` document that path.

The down-sampling study is reproduced the same way, one arm at a time, for
example for the hierarchical arm:

```bash
python3 scripts/eval_v6.py \
  --predictions "FT-Hier-224=results/v6_40k_hier_gt_bird.csv" \
  --predictions "FT-Hier-128=results/v6_40k_hier_gt_bird_ds128.csv" \
  --predictions "FT-Hier-32=results/v6_40k_hier_gt_bird_ds32.csv" \
  --nonbird-predictions "FT-Hier-224=results/v6_40k_hier_gt_nonbird.csv" \
  --nonbird-predictions "FT-Hier-128=results/v6_40k_hier_gt_nonbird_ds128.csv" \
  --nonbird-predictions "FT-Hier-32=results/v6_40k_hier_gt_nonbird_ds32.csv" \
  --class-to-taxon configs/class_to_taxon_v5.json \
  --nonbird-manifest manifests/gt_nonbird_840_manifest.csv \
  --output-json /tmp/eval_ds_hier.json
```

This regenerates `results/eval_v6_ds_hier.json`; substitute `v6_40k_flat_` or
`v6IN_flat_` and the matching labels for the other two arms.

## Headline results

All numbers are from `results/eval_v6_final.json`. Brackets are 95 percent
species-level cluster bootstrap intervals.

| Metric | IN-Base | FT-Flat | FT-Hier |
|---|---|---|---|
| Stage 1 bird/non-bird accuracy (n=1,680) | **0.9726** [0.9635, 0.9809] | 0.9577 | 0.9577 |
| Stage 2 raptor/non-raptor (n=840) | 0.9286 | **0.9333** | 0.9333 |
| Stage 3 raptor order (n=370) | 0.7865 | **0.8189** | 0.8189 |
| Stage 4 top-1 (n=260) | 0.2731 | **0.4038** | 0.4038 |
| Stage 4 top-5 (n=260) | 0.5462 | 0.5962 | **0.6115** |
| End-to-end IUCN accuracy (n=1,680) | 0.9369 | 0.9452 | **0.9506** [0.9257, 0.9726] |
| Always-OTHERS trivial baseline | 0.9167 | 0.9167 | 0.9167 |
| Protected-species alert recall | 0.250 | **0.325** | 0.250 |

## Input-resolution study

The deployed classifier receives a PTZ crop of only tens of pixels, so the same
ground truth was reduced to 32x32 and 128x128 (bicubic down and back up) and the
features re-extracted. **No model was retrained**: the backbone and every
LightGBM stage are the ones above, so the columns differ only in input
resolution. Full numbers are in section 7 of `RESULTS_v6_rebuild.md`.

| Metric | IN-Base 224/128/32 | FT-Flat 224/128/32 | FT-Hier 224/128/32 |
|---|---|---|---|
| Stage 1 bird/non-bird | .973 / .966 / **.827** | .958 / .951 / **.626** | same as flat |
| Stage 4 top-1 | .273 / .165 / .069 | .404 / .292 / .065 | .404 / .300 / .062 |
| Stage 4 top-5 | .546 / .389 / .189 | .596 / .542 / .219 | .612 / .573 / .215 |
| End-to-end IUCN | .937 / .927 / .912 | .945 / .939 / .919 | .951 / .934 / .919 |
| Alert precision | .625 / .417 / **.125** | .765 / .714 / **.099** | .833 / .588 / **.101** |
| Alert recall | .250 / .125 / (.575) | .325 / .250 / (.500) | .250 / .250 / (.450) |

Three points are worth stating explicitly:

- At 128x128 the cascade degrades but still works. The top-1 loss is significant
  in every arm (Holm-adjusted p between 0.009 and 0.014); the top-5 loss is not
  significant in the fine-tuned arms.
- At 32x32 it fails. End-to-end accuracy of 0.919 is indistinguishable from the
  always-OTHERS baseline of 0.9167.
- **The alert recall that rises at 32x32 is not an improvement.** Precision
  collapses from 0.765 to 0.099 (13 true / 4 false positives becomes 20 / 183):
  the classifier is firing almost indiscriminately. Read the pair, never the
  recall alone.

## What the evidence does and does not support

These qualifications are part of the result, not caveats bolted on afterwards.

1. **Fine-tuning is significant on top-1 only.** Stage 4 top-1 rises from 0.2731
   to 0.4038 (+13.1 pt, cluster sign-flip test, Holm-corrected p = 0.015). The
   top-5 gain (+5.0 pt, p = 0.39) and the end-to-end IUCN accuracy (p = 0.83)
   are **not** significant.
2. **Hierarchical refinement is not shown to be redundant.** Convergence of the
   hierarchical arm decides the sign of this comparison: run it unconverged and
   the hierarchical arm looks worse (-0.0385), which would read as redundancy;
   with both arms converged the sign reverses (**+0.0154**). Either way the
   pre-specified 5 pt non-inferiority and equivalence tests both FAIL (95
   percent CI [-0.054, +0.104]). The supportable statement is that no
   difference was detected and equivalence was not established. Anyone
   reproducing this comparison should confirm both arms have converged before
   reading the sign.
3. **Fine-tuning makes Stage 1 worse**, from 0.9726 to 0.9577, with non-bird
   rejection dropping from 0.9583 to 0.9274.
4. **Protected-species alert recall is 0.25 to 0.33**, which is the limiting
   quantity for the intended application.
5. **VU recall cannot exceed 40/60** because 20 of the 60 VU images are not
   Accipitriformes and therefore cannot be identified at Stage 4.
6. Stage 5 accuracy must be read against the always-OTHERS trivial baseline of
   0.9167, not against zero.

Image-level McNemar tests are reported in the output as `p_img` **for reference
only**. The ground truth is 10 images per species, so image-level independence
is not tenable; the cluster sign-flip test is the correct one. The two disagree
materially: the IN-Base to FT-Hier IUCN comparison is p = 0.0022 image-level but
p = 0.036 (Holm 0.255) clustered.

## Data provenance and licensing

Evaluation and training images come from
[iNaturalist](https://www.inaturalist.org/). iNaturalist does not own the
photographs; each image is redistributed under the licence its contributor
chose. The manifests in `manifests/` record, per photograph, the observation ID,
photo ID, contributor credit, licence and source URL. Non-bird ground truth was
collected with `without_taxon_id=3` to exclude Aves explicitly, and deduplicated
against the training set by MD5.

**The image files themselves are not redistributed here.** The manifests contain
everything needed to retrieve them from iNaturalist under their original terms.

## Citation

For the artifacts:

```bibtex
@software{nishikawa2026artifacts,
  author    = {Nishikawa, Takeshi},
  title     = {bs2026-v6-artifacts: model weights, predictions and evaluation
               code for IUCN Red List threatened bird classification from
               citizen-science images},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21894830},
  url       = {https://doi.org/10.5281/zenodo.21894830}
}
```

For the paper:

```bibtex
@misc{nishikawa2026iucn,
  author = {Nishikawa, Takeshi},
  title  = {IUCN Red List Threatened Bird Classification from Citizen-Science
            Images using Fine-Tuned CNN Features and a Hierarchical Lightweight
            Classifier},
  year   = {2026},
  note   = {Artifacts: \url{https://doi.org/10.5281/zenodo.21894830}}
}
```

## Licence

- Code, scripts and manifests in this repository: **MIT**, see [LICENSE](LICENSE).
- Released model weights: **CC BY-NC 4.0**, see [LICENSE-WEIGHTS](LICENSE-WEIGHTS).
  The non-commercial condition follows the licences of the iNaturalist
  photographs the model was trained on, many of which are CC BY-NC.
- The photographs themselves remain under their contributors' own licences as
  recorded in `manifests/`.

---

## 日本語による要約

本リポジトリは上記論文の再現用アーティファクト一式である。学習済みモデル重み
（リリースアセット）、論文の全表の根拠となる予測ファイル、および分母・ブート
ストラップ信頼区間・クラスタ符号反転検定・Holm 補正・非劣性検定を計算する評価
スクリプトを収録する。

[Reproducing the paper's numbers](#reproducing-the-papers-numbers) のコマンドは
`results/eval_v6_final.json` をバイト単位で再生成する（実行約35秒、標準ライブラリ
のみ）。確定結果表と再現手順の詳細は [RESULTS_v6_rebuild.md](RESULTS_v6_rebuild.md)
を参照のこと。

主要な留保事項は上記
[What the evidence does and does not support](#what-the-evidence-does-and-does-not-support)
に記載した。とくに、fine-tuning の効果が有意なのは Stage 4 top-1 のみであること、
属レベル階層化の冗長性は本データでは立証されていないこと、保護対象4種のアラート
recall が 0.25〜0.33 にとどまることは、結果の一部として明示的に読まれるべきもので
ある。
