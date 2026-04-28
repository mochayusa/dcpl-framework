#!/usr/bin/env bash
set -euo pipefail

# ======================================
# CONFIGURATION
# ======================================
N_RUNS=30
BASE_SEED=42
SEED_STRIDE=1000
TEST_SIZE=0.20
TARGET="Target_throughput_tokens_per_sec"

PER_MODEL_DIR="data/llm_pilot_data/raw_data/per_model"
RESULTS_ROOT="results/runs"

# DCPL gates to evaluate
GATES=("ridge" "nn" "rf" "lr")

# Generate ONE run id for baseline + DCPL
RUN_ID=$(date +"%Y%m%d_%H%M%S")

# ======================================
# SANITY CHECK
# ======================================
if [ ! -d "$PER_MODEL_DIR" ]; then
  echo "[ERROR] per_model_dir not found: $PER_MODEL_DIR"
  exit 1
fi

echo "======================================"
echo " Running BASELINE + DCPL experiments"
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

# ======================================
# 1) RUN BASELINE (ALL LEARNERS)
# ======================================
echo "--------------------------------------"
echo "[BASELINE] Running all baseline learners"
echo "--------------------------------------"

python src/project_main.py baseline all "$N_RUNS" \
  --per-model-dir "$PER_MODEL_DIR" \
  --target "$TARGET" \
  --test-size "$TEST_SIZE" \
  --base-seed "$BASE_SEED" \
  --seed-stride "$SEED_STRIDE" \
  --results-root "$RESULTS_ROOT" \
  --run-id "$RUN_ID"

echo "[DONE] BASELINE completed"
echo

# ======================================
# 2) RUN DCPL (ALL GATES)
# ======================================
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
    --results-root "$RESULTS_ROOT" \
    --run-id "$RUN_ID"

  echo "[DONE] DCPL gate = $GATE"
  echo
done

# ======================================
# DONE
# ======================================
echo "======================================"
echo " All BASELINE + DCPL experiments done"
echo " Results grouped under:"
echo "   $RESULTS_ROOT/$RUN_ID/"
echo "======================================"
