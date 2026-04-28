#!/usr/bin/env bash
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

GATES=("ridge" "nn" "rf" "lr")

# =========================
# ENV CHECK (OPTIONAL)
# =========================
if [ ! -d "$PER_MODEL_DIR" ]; then
  echo "[ERROR] per_model_dir not found: $PER_MODEL_DIR"
  exit 1
fi

echo "======================================"
echo " Running DCPL experiments for all gates"
echo " Runs        : $N_RUNS"
echo " Target      : $TARGET"
echo " Test split  : 80/20"
echo " Base seed   : $BASE_SEED"
echo " Seed stride : $SEED_STRIDE"
echo "======================================"
echo

# =========================
# RUN DCPL PER GATE
# =========================
for GATE in "${GATES[@]}"; do
  echo "--------------------------------------"
  echo "[DCPL] Running gate = $GATE"
  echo "--------------------------------------"

  python src/project_main.py dcpl "$N_RUNS" \
    --gate-kind "$GATE" \
    --per-model-dir "$PER_MODEL_DIR" \
    --target "$TARGET" \
    --test-size "$TEST_SIZE" \
    --base-seed "$BASE_SEED" \
    --seed-stride "$SEED_STRIDE" \
    --results-root "$RESULTS_ROOT"

  echo "[DONE] Gate = $GATE"
  echo
done

echo "======================================"
echo " All DCPL gate experiments completed!"
echo "======================================"
