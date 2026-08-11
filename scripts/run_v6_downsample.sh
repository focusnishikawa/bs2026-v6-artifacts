#!/usr/bin/env bash
# v6 controlled down-sampling study: how far does the cascade degrade when the
# classifier is handed a crop carrying only 32x32 or 128x128 pixels?
#
# Nothing is re-trained.  The LightGBM stages (models_v6_flat40k,
# models_v6_hier40k, models_v6IN_flat) and the fine-tuned backbone are the ones
# behind every table in the paper; only the ground-truth features are recomputed
# at the reduced input resolutions, so the comparison isolates input resolution.
#
# Native 224 is not recomputed here: the existing preds/v6_40k_*.csv and
# preds/v6IN_flat_*.csv already are that condition.
set -u

ROOT=/home1/gfsi/ufsi0002/bs2026_v5_rebuild
cd "$ROOT" || exit 1
mkdir -p logs preds data

SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
APP=${APPTAINER_BIN:-/home1/gfsi/ufsi0002/apptainer/bin/apptainer}
TORCH_CACHE=/home1/gfsi/ufsi0002/bird_detection/torch_cache
CKPT=models/resnet50_ft_v6/best.pt
C2T=configs/class_to_taxon_v5.json
LOG=logs/v6_downsample.log
DONE=logs/v6_downsample.done
SCALES="32 128"

rm -f "$DONE"
T0=$(date +%s)
echo "=== START $(date '+%F %T') ===" | tee -a "$LOG"

# ------------------------------------------------------------------ extract
# One apptainer exec at a time: the SIF is 10.9 GB and parallel execs on the
# shared Lustre mount have thrashed before.
extract() {
    local TAG=$1 CSV=$2 OUT=$3 S=$4 CK=$5
    shift 5
    echo "--- extract $TAG (S=$S, ckpt=${CK:-ImageNet}) ---" | tee -a "$LOG"
    "$APP" exec --nv --bind /home1 --env PYTHONNOUSERSITE=1 \
        --env TORCH_HOME="$TORCH_CACHE" \
        "$SIF" \
        python3 scripts/extract_cnn_features_v2.py \
            --hierarchy-csv "$CSV" \
            --output "$OUT" \
            --model resnet50 \
            ${CK:+--checkpoint "$CK"} \
            --batch-size 256 \
            --num-workers 16 \
            --device cuda \
            --downsample-to "$S" \
            "$@" \
            2>>"$LOG"
    echo "  extract $TAG rc=$?" | tee -a "$LOG"
}

for S in $SCALES; do
    # fine-tuned backbone
    extract "ft/gt_bird/$S"    data/hierarchy_labels_gt_84_y.csv \
        "data/features_v6_gt_bird_r50_ft_ds${S}.npz"    "$S" "$CKPT"
    extract "ft/gt_nonbird/$S" data/hierarchy_labels_gt_nonbird_y.csv \
        "data/features_v6_gt_nonbird_r50_ft_ds${S}.npz" "$S" "$CKPT" \
        --target-nonbird 840
    # ImageNet-pretrained baseline backbone
    extract "in/gt_bird/$S"    data/hierarchy_labels_gt_84_y.csv \
        "data/features_v6IN_gt_bird_r50_ft_ds${S}.npz"    "$S" ""
    extract "in/gt_nonbird/$S" data/hierarchy_labels_gt_nonbird_y.csv \
        "data/features_v6IN_gt_nonbird_r50_ft_ds${S}.npz" "$S" "" \
        --target-nonbird 840
done

# ------------------------------------------------------------------- infer
run() { "$APP" exec --bind /home1 "$SIF" "$@"; }

for S in $SCALES; do
    for SPLIT in gt_bird gt_nonbird; do
        FT_FEAT="data/features_v6_${SPLIT}_r50_ft_ds${S}.npz"
        IN_FEAT="data/features_v6IN_${SPLIT}_r50_ft_ds${S}.npz"

        echo "--- infer flat / $SPLIT / $S ---" | tee -a "$LOG"
        run python3 scripts/infer_hybrid_v5.py \
            --models-stage1 models_v6_flat40k \
            --models-stage234 models_v6_flat40k \
            --features-stage1 "$FT_FEAT" \
            --features-stage234 "$FT_FEAT" \
            --class-to-taxon "$C2T" \
            --output "preds/v6_40k_flat_${SPLIT}_ds${S}.csv" 2>>"$LOG"
        echo "  flat/$SPLIT/$S rc=$?" | tee -a "$LOG"

        echo "--- infer hier / $SPLIT / $S ---" | tee -a "$LOG"
        run python3 scripts/infer_hybrid_v5_s4hier.py \
            --models-stage1 models_v6_flat40k \
            --models-stage23 models_v6_flat40k \
            --models-stage4-hier models_v6_hier40k \
            --features "$FT_FEAT" \
            --class-to-taxon "$C2T" \
            --output "preds/v6_40k_hier_${SPLIT}_ds${S}.csv" 2>>"$LOG"
        echo "  hier/$SPLIT/$S rc=$?" | tee -a "$LOG"

        echo "--- infer IN-base / $SPLIT / $S ---" | tee -a "$LOG"
        run python3 scripts/infer_hybrid_v5.py \
            --models-stage1 models_v6IN_flat \
            --models-stage234 models_v6IN_flat \
            --features-stage1 "$IN_FEAT" \
            --features-stage234 "$IN_FEAT" \
            --class-to-taxon "$C2T" \
            --output "preds/v6IN_flat_${SPLIT}_ds${S}.csv" 2>>"$LOG"
        echo "  IN/$SPLIT/$S rc=$?" | tee -a "$LOG"
    done
done

T1=$(date +%s)
echo "=== END $(date '+%F %T') elapsed=$((T1 - T0))s ===" | tee -a "$LOG"
echo 0 > "$DONE"
