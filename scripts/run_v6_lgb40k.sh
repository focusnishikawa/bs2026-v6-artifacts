#!/usr/bin/env bash
# v6 re-experiment step 3b: raise the boosting-round cap so that BOTH arms are
# genuinely converged.
#
# In the 10,000-round run the flat Stage 4 stopped at best_iter=10000, i.e. it
# hit the cap and was still improving, while the hierarchical Stage 4a converged
# on its own at 2,451 rounds. Comparing them at that point would have repeated
# the original defect (an unconverged arm) with the roles reversed. Here both
# arms get a 40,000-round budget with the same early stopping, so whichever
# stops first does so because it converged, not because it ran out of budget.
set -u

ROOT=/home1/gfsi/ufsi0002/bs2026_v5_rebuild
cd "$ROOT" || exit 1
mkdir -p logs models_v6_flat40k models_v6_hier40k

SIF=/home1/gfsi/ufsi0002/bird_detection/bird_python.sif
APPTAINER=${APPTAINER_BIN:-apptainer}
FEAT=data/features_v6_full_y_r50_ft_cleaned.npz
C2T=configs/class_to_taxon_v5.json
ROUNDS=40000
ES=100
THREADS=${THREADS:-32}
LOG=logs/v6_lgb40k_stderr.log
DONE=logs/v6_lgb40k.done

run() {
    "$APPTAINER" exec --bind /home1 \
        --env OMP_NUM_THREADS="$THREADS" \
        --env MKL_NUM_THREADS="$THREADS" \
        --env OPENBLAS_NUM_THREADS="$THREADS" \
        --env NUMEXPR_NUM_THREADS="$THREADS" \
        "$SIF" "$@"
}

T0=$(date +%s)
echo "=== START $(date '+%F %T') rounds=$ROUNDS ===" | tee -a "$LOG"

echo "--- flat stage 4 ---" | tee -a "$LOG"
run python3 scripts/train_hierarchy_lgb.py \
    --features "$FEAT" --output-dir models_v6_flat40k \
    --stages 4 --num-rounds "$ROUNDS" --early-stopping "$ES" \
    --num-threads "$THREADS" --class-to-taxon "$C2T" 2>>"$LOG"
echo "  flat rc=$?" | tee -a "$LOG"

echo "--- hier stage 4a ---" | tee -a "$LOG"
run python3 scripts/train_hierarchy_lgb.py \
    --features "$FEAT" --output-dir models_v6_hier40k \
    --stages 4a --num-rounds "$ROUNDS" --early-stopping "$ES" \
    --num-threads "$THREADS" --class-to-taxon "$C2T" 2>>"$LOG"
echo "  4a rc=$?" | tee -a "$LOG"

GENERA=$(run python3 -c "
import json
print(' '.join(json.load(open('models_v6_hier40k/meta_4a.json'))['genus_names']))
")
for G in $GENERA; do
    run python3 scripts/train_hierarchy_lgb.py \
        --features "$FEAT" --output-dir models_v6_hier40k \
        --stages 4b --genus "$G" --num-rounds "$ROUNDS" --early-stopping "$ES" \
        --num-threads "$THREADS" --class-to-taxon "$C2T" 2>>"$LOG"
done
echo "  4b done" | tee -a "$LOG"

# stages 1-3 are shared with the 10k run and already converged well inside the
# cap (best_iter 220 / 493 / 114), so reuse them rather than retrain.
for s in stage1 stage2 stage3; do
    cp -f "models_v6_flat/${s}.txt" "models_v6_flat40k/${s}.txt"
done
cp -f models_v6_flat/meta.json models_v6_flat40k/meta_stages123.json 2>/dev/null

T1=$(date +%s)
echo "=== END $(date '+%F %T') elapsed=$((T1 - T0))s ===" | tee -a "$LOG"
echo 0 > "$DONE"
