DATA=data/llm_pilot_data/raw_data/merged_all_models.csv
TARGET=Target_throughput_tokens_per_sec

# This is for running all baseline models on the global merged dataset with 5 runs each.
# for mk in lr ridge rf_light nn llm_pilot; do
#   PYTHONPATH=src python src/experiments/run_baseline_split80_merged_5runs_global.py \
#     --data "$DATA" \
#     --target "$TARGET" \
#     --learner "$mk" \
#     --seeds "42,1042,2042,3042,4042" \
#     --results-root results/runs \
#     --run-name baseline_split80_merged
# done

# This is for running DCPL framework on the global merged dataset with 5 runs.
#!/bin/bash
set -e

export PYTHONPATH=src

DATA="data/llm_pilot_data/raw_data/merged_all_models.csv"
TARGET="Target_throughput_tokens_per_sec"
OUT="results/runs/dcpl_split80_global"

SEEDS=(42 1042 2042 3042 4042)

echo "[INFO] Running DCPL global experiment"

for SEED in "${SEEDS[@]}"; do
  echo "[INFO] Seed = ${SEED}"
  python src/experiments/run_dcpl_split80_merged_5runs_global.py \
    --data "$DATA" \
    --target "$TARGET" \
    --gate ridge \
    --inner-splits 5 \
    --test-size 0.20 \
    --seeds "$SEED" \
    --out-dir "$OUT/seed_${SEED}"
done

