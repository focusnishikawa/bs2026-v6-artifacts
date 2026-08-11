#!/usr/bin/env bash
# v6 re-experiment step 2: extract fine-tuned ResNet50 features on hpciaiss1.
# Three feature files: the full_y training set, the 840-image bird GT, and the
# newly collected held-out non-bird GT (needed to make Stage 1 a real
# bird/non-bird accuracy rather than bird recall only).
set -u

ROOT=/home1/gfsi/ufsi0002/bs2026_v5_rebuild
cd "$ROOT" || exit 1
mkdir -p logs data

SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
TORCH_CACHE=/home1/gfsi/ufsi0002/bird_detection/torch_cache
CKPT=NONE
LOG=logs/v6IN_extract_stderr.log
DONE=logs/v6IN_extract.done
ELAPSED=logs/v6IN_extract.elapsed


T0=$(date +%s)
echo "=== START $(date '+%F %T') ===" | tee -a "$LOG"

run_one() {
    local TAG=$1 CSV=$2 OUT=$3
    shift 3
    echo "--- $TAG ---" | tee -a "$LOG"
    apptainer exec --nv --bind /home1 --env PYTHONNOUSERSITE=1 \
        --env TORCH_HOME="$TORCH_CACHE" \
        "$SIF" \
        python3 scripts/extract_cnn_features_v2.py \
            --hierarchy-csv "$CSV" \
            --output "$OUT" \
            --model resnet50 \
            \
            --batch-size 256 \
            --num-workers 16 \
            --device cuda \
            "$@" \
            2>>"$LOG"
    echo "  $TAG rc=$?" | tee -a "$LOG"
}

run_one "full_y"     data/hierarchy_labels_v5b_full_y.csv \
                     data/features_v6IN_full_y_r50_ft.npz
run_one "gt_bird"    data/hierarchy_labels_gt_84_y.csv \
                     data/features_v6IN_gt_bird_r50_ft.npz
# The non-bird GT CSV has zero bird rows, and extract_cnn_features_v2.py
# defaults --target-nonbird to len(bird_rows); leaving it at the default would
# subsample the non-bird set down to 0 images and emit an empty feature file.
run_one "gt_nonbird" data/hierarchy_labels_gt_nonbird_y.csv \
                     data/features_v6IN_gt_nonbird_r50_ft.npz \
                     --target-nonbird 840

T1=$(date +%s)
DT=$((T1 - T0))
echo "=== END $(date '+%F %T') elapsed=${DT}s ===" | tee -a "$LOG"
echo "$DT" > "$ELAPSED"
echo 0 > "$DONE"
