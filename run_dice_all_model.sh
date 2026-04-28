#!/usr/bin/env bash
# Disentangled Interaction-Concatenated Ensemble (DICE) all learners run script
set -euo pipefail

# =========================
# CONFIGURATION
# =========================
N_RUNS=30
BASE_SEED=42
SEED_STRIDE=1000
TEST_SIZE=0.20
TARGET="Target_throughput_tokens_per_sec"
PER_MODEL_DIR="data/llm_pilot_data/raw_data/per_model"
RESULTS_ROOT="results/runs"

# Generate ONE run id for all learners
RUN_ID=$(date +"%Y%m%d_%H%M%S")

# DICE learners (must match project_dice_main.py)
LEARNERS=("lr" "ridge" "rf" "nn" "xgb")

# Optional ablation flags
# EXTRA_FLAGS="--no-interactions"
# EXTRA_FLAGS="--no-base"
EXTRA_FLAGS=""

# =========================
# SANITY CHECK
# =========================
if [ ! -d "$PER_MODEL_DIR" ]; then
  echo "[ERROR] per_model_dir not found: $PER_MODEL_DIR"
  exit 1
fi

echo "======================================"
echo " Running DICE experiments (grouped)"
echo " RUN_ID      : $RUN_ID"
echo " Runs        : $N_RUNS"
echo " Target      : $TARGET"
echo " Test split  : 80/20"
echo " Base seed   : $BASE_SEED"
echo " Seed stride : $SEED_STRIDE"
echo " Data dir    : $PER_MODEL_DIR"
echo " Results root: $RESULTS_ROOT/$RUN_ID"
echo "======================================"
echo

# =========================
# RUN DICE PER LEARNER
# =========================
for L in "${LEARNERS[@]}"; do
  echo "--------------------------------------"
  echo "[DICE] Learner = $L"
  echo "--------------------------------------"

  python src/project_dice_main.py "$L" "$N_RUNS" \
    --per-model-dir "$PER_MODEL_DIR" \
    --target "$TARGET" \
    --test-size "$TEST_SIZE" \
    --base-seed "$BASE_SEED" \
    --seed-stride "$SEED_STRIDE" \
    --results-root "$RESULTS_ROOT" \
    --run-id "$RUN_ID" \
    $EXTRA_FLAGS

  echo "[DONE] Learner = $L"
  echo
done

echo "======================================"
echo " All DICE learner experiments completed"
echo " Results grouped under:"
echo "   $RESULTS_ROOT/$RUN_ID/"
echo "======================================"
