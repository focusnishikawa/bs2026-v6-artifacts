#!/usr/bin/env bash
# v6 re-experiment step 4b: inference with the CONVERGED (40,000-round) models.
#
# This is the script that produced the prediction files behind every table in
# the paper. run_v6_infer.sh is kept only because it produced the discarded
# 10,000-round run, whose flat arm hit its cap.
set -u

ROOT=/home1/gfsi/ufsi0002/bs2026_v5_rebuild
cd "$ROOT" || exit 1
mkdir -p logs preds

SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
APPTAINER=${APPTAINER_BIN:-apptainer}
C2T=configs/class_to_taxon_v5.json
LOG=logs/v6_infer40k_stderr.log
DONE=logs/v6_infer40k.done

run() { "$APPTAINER" exec --bind /home1 "$SIF" "$@"; }

T0=$(date +%s)
echo "=== START $(date '+%F %T') ===" | tee -a "$LOG"

for SPLIT in gt_bird gt_nonbird; do
    FEAT=data/features_v6_${SPLIT}_r50_ft_cleaned.npz
    [ -f "$FEAT" ] || FEAT=data/features_v6_${SPLIT}_r50_ft.npz

    echo "--- flat / $SPLIT ---" | tee -a "$LOG"
    run python3 scripts/infer_hybrid_v5.py \
        --models-stage1 models_v6_flat40k \
        --models-stage234 models_v6_flat40k \
        --features-stage1 "$FEAT" \
        --features-stage234 "$FEAT" \
        --class-to-taxon "$C2T" \
        --output preds/v6_40k_flat_${SPLIT}.csv 2>>"$LOG"
    echo "  flat/$SPLIT rc=$?" | tee -a "$LOG"

    echo "--- hier / $SPLIT ---" | tee -a "$LOG"
    run python3 scripts/infer_hybrid_v5_s4hier.py \
        --models-stage1 models_v6_flat40k \
        --models-stage23 models_v6_flat40k \
        --models-stage4-hier models_v6_hier40k \
        --features "$FEAT" \
        --class-to-taxon "$C2T" \
        --output preds/v6_40k_hier_${SPLIT}.csv 2>>"$LOG"
    echo "  hier/$SPLIT rc=$?" | tee -a "$LOG"
done

T1=$(date +%s)
echo "=== END $(date '+%F %T') elapsed=$((T1 - T0))s ===" | tee -a "$LOG"
echo 0 > "$DONE"
