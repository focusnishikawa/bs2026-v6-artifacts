#!/usr/bin/env bash
# ImageNet-pretrained (no fine-tuning) baseline arm: extract -> train -> infer.
set -u
ROOT=/home1/gfsi/ufsi0002/bs2026_v5_rebuild
cd "$ROOT" || exit 1
LOG=logs/v6IN_chain.log
APP=/home1/gfsi/ufsi0002/apptainer/bin/apptainer
SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
T=32
echo "=== IN chain start $(date +%F\ %T) ===" >> "$LOG"

bash scripts/run_v6IN_extract.sh >> "$LOG" 2>&1
echo "extract rc=$?" >> "$LOG"

run() { "$APP" exec --bind /home1 --env OMP_NUM_THREADS=$T --env MKL_NUM_THREADS=$T \
        --env OPENBLAS_NUM_THREADS=$T --env NUMEXPR_NUM_THREADS=$T "$SIF" "$@"; }

FEAT=data/features_v6IN_full_y_r50_ft_cleaned.npz
run python3 scripts/cleanup_stage4_labels.py \
    --input data/features_v6IN_full_y_r50_ft.npz --output "$FEAT" >> "$LOG" 2>&1
echo "cleanup rc=$?" >> "$LOG"

mkdir -p models_v6IN_flat
run python3 scripts/train_hierarchy_lgb.py --features "$FEAT" \
    --output-dir models_v6IN_flat --stages 1,2,3,4 \
    --num-rounds 40000 --early-stopping 100 --num-threads $T \
    --class-to-taxon configs/class_to_taxon_v5.json >> "$LOG" 2>&1
echo "train rc=$?" >> "$LOG"

for SPLIT in gt_bird gt_nonbird; do
    run python3 scripts/infer_hybrid_v5.py \
        --models-stage1 models_v6IN_flat --models-stage234 models_v6IN_flat \
        --features-stage1 data/features_v6IN_${SPLIT}_r50_ft.npz \
        --features-stage234 data/features_v6IN_${SPLIT}_r50_ft.npz \
        --class-to-taxon configs/class_to_taxon_v5.json \
        --output preds/v6IN_flat_${SPLIT}.csv >> "$LOG" 2>&1
    echo "infer $SPLIT rc=$?" >> "$LOG"
done
echo "=== IN chain finished $(date +%F\ %T) ===" >> "$LOG"
echo done > logs/v6IN_chain.done
