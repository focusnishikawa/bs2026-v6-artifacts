#!/usr/bin/env bash
# v6 re-experiment step 1: ResNet50 fine-tuning on hpciaiss1 (DDP, 4x H200 NVL).
# Reproduces the settings reported in the paper: head lr 4e-4, layer4 lr 4e-5,
# effective batch 512 (128 x 4), max 20 epochs, early stopping patience 3, bf16.
set -u

ROOT=/home1/gfsi/ufsi0002/bs2026_v5_rebuild
cd "$ROOT" || exit 1
mkdir -p logs models

SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
TORCH_CACHE=/home1/gfsi/ufsi0002/bird_detection/torch_cache
LOG=logs/v6_ft_stderr.log
DONE=logs/v6_ft.done
ELAPSED=logs/v6_ft.elapsed

T0=$(date +%s)
echo "=== START $(date '+%F %T') ===" | tee -a "$LOG"

apptainer exec --nv --bind /home1 --env PYTHONNOUSERSITE=1 \
    --env TORCH_HOME="$TORCH_CACHE" \
    "$SIF" \
    torchrun --standalone --nproc_per_node=4 \
        scripts/finetune_resnet50_v5b.py \
        --hierarchy-csv data/hierarchy_labels_v5b_full_y.csv \
        --output-dir models/resnet50_ft_v6 \
        --epochs 20 --batch-size 128 \
        --lr-head 4e-4 --lr-layer4 4e-5 \
        --num-workers 8 \
        --patience 3 \
        2>>"$LOG"
RC=$?

T1=$(date +%s)
DT=$((T1 - T0))
echo "=== END $(date '+%F %T') rc=$RC elapsed=${DT}s ===" | tee -a "$LOG"
echo "$DT" > "$ELAPSED"
echo "$RC" > "$DONE"
