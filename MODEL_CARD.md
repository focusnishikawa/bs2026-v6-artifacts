# Model card: resnet50_ft_v6_best.pt

A ResNet50 fine-tuned on citizen-science bird photographs, used as a frozen
2048-dimensional feature extractor for a downstream cascade of LightGBM stages
that ends in an IUCN Red List category. Released as an asset of
[bs2026-v6-artifacts](https://github.com/focusnishikawa/bs2026-v6-artifacts).

## Model details

| Item | Value |
|---|---|
| Architecture | ResNet50 (torchvision), ImageNet-pretrained, classification head replaced with 895 outputs |
| Fine-tuned layers | Head and `layer4` only; earlier blocks frozen |
| File | `resnet50_ft_v6_best.pt`, 101,667,110 bytes (96.96 MiB) |
| MD5 | `573d20215bfd92a2e6246b6b6e68eab1` |
| Class labels | `configs/resnet50_ft_v6_meta.json`, key `labels_sorted`, 895 entries in output-index order |
| Best validation top-1 | 0.8482 at epoch 19 of 20 |
| Trained | 2026-08-10 |
| Licence | CC BY-NC 4.0 (see LICENSE-WEIGHTS) |

## Training configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 20 (early stopping patience 3; stopped naturally, best at 19) |
| Batch size | 128 per rank, 4 ranks, effective 512 |
| Learning rate | head 4e-4, `layer4` 4e-5 |
| Weight decay | 1e-4 |
| Precision | bfloat16 |
| Validation split | 10 percent |
| Seed | 42 |
| Hardware | 4x NVIDIA H200 NVL, DDP via `torchrun` |
| Wall time | 1,156 s |

Per-epoch loss and accuracy are in `results/finetune_log.csv`. The exact
invocation is `scripts/run_v6_ft.sh`.

## Training data

| Item | Value |
|---|---|
| Total | 151,539 images |
| Birds | 131,539 |
| Non-birds | 20,000 |
| Label space | 895 = 894 bird label variants plus a non-bird label |
| Source | iNaturalist, retrieved 2026-04. Per-photograph attribution in `manifests/train_inat_manifest.csv` |

Note on the label count: an earlier version of this study reported 888 classes
under a `partial_y` label allow-list of 427 entries. That allow-list was lost
with the original cluster workspace and **could not be reconstructed**, so the
whole pipeline was rebuilt on `full_y`, giving 895. The image count reproduced
the previously reported figure exactly, and validation accuracy reproduced to
within 0.001 (0.8482 against 0.8492), but the 888-class configuration itself is
not reproducible from the surviving artifacts.

## Intended use

Feature extraction for the raptor-focused IUCN classification cascade described
in the paper, and comparable research on threatened-species screening from
citizen-science imagery. The intended deployment context is a bird-strike
mitigation system at wind farms, where the cascade narrows candidates for a
human expert.

## Out of scope

- **Unattended labelling.** Stage 4 identifies the correct species outright in
  40 percent of cases across 71 Accipitriformes species, and places it in the
  top five in 61 percent. That supports narrowing candidates for expert review;
  it does not support autonomous determination.
- **Conservation status decisions.** Predicted IUCN categories are not a
  substitute for an assessment.
- **Species or regions outside the label space.** The 895 labels are dominated
  by species present in the training corpus; behaviour on others is undefined.
- **Small or distant subjects.** The evaluation uses citizen-science photographs
  in which the bird is reasonably large in frame. Related work in this project
  found that classifiers of this family degrade sharply below roughly 35 pixels
  across, which is the regime a fixed wide-angle field camera actually produces.

## Evaluation

Ground truth is 1,680 images: 840 bird images at 10 images per species across 84
species, plus 840 non-bird images collected with `without_taxon_id=3` to exclude
Aves and deduplicated against the training set by MD5.

Because the ground truth is 10 images per species, image-level independence does
not hold. All confidence intervals are species-level cluster bootstraps (10,000
resamples) and all paired tests are species-level cluster sign-flip tests with
Holm correction. Non-bird images are clustered by their 607 source taxa.

Headline numbers, and the full set of qualifications on what they support, are
in [README.md](README.md). The authoritative output is
`results/eval_v6_final.json`.

## Limitations and known failure modes

1. **Fine-tuning degrades Stage 1.** Bird/non-bird accuracy falls from 0.9726
   (ImageNet baseline) to 0.9577, and non-bird rejection from 0.9583 to 0.9274.
   Specialising the features for fine-grained bird discrimination costs
   generality. For a deployment whose first job is rejecting non-birds, the
   ImageNet baseline is the better feature extractor.
2. **Protected-species alert recall is 0.25 to 0.33.** For the four protected
   species that motivate the application, the system misses two thirds to three
   quarters of true alerts. Precision is comparatively good (0.63 to 0.83). This
   is the limiting quantity for practical use.
3. **VU recall is structurally capped at 40/60**, because 20 of the 60 VU images
   are not Accipitriformes and so cannot be resolved at Stage 4.
4. **End-to-end accuracy is close to a trivial baseline.** Always predicting
   OTHERS scores 0.9167; the best configuration reaches 0.9506. Overall accuracy
   is the wrong headline metric for this task.
5. **The gain from fine-tuning is significant only on Stage 4 top-1.** Top-5 and
   end-to-end IUCN accuracy are not significant once multiplicity and clustering
   are handled.
6. **Class imbalance and geographic bias.** Citizen-science imagery
   over-represents accessible regions, common species and adult plumages.
   Behaviour on juveniles, rare morphs and under-photographed regions is not
   characterised.

## Ethical considerations

Locations of threatened species can enable harm such as poaching or nest
disturbance. The manifests in this repository carry iNaturalist observation IDs;
iNaturalist itself obscures coordinates for sensitive taxa, and that obscuring
should not be circumvented. No coordinates are redistributed here.

Photographs remain the property of their contributors and are redistributed only
by reference, under the licences recorded in `manifests/`.

## Citation

See [README.md](README.md).
