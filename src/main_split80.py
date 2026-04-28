from experiments.run_dcpl_split80_permodel_multirun import run_dcpl_split80_permodel_nx
# from experiments.run_baseline_split80_permodel_multirun import run_baseline_split80_permodel_5x
from experiments.run_baseline_split80_permodel_nested_multirun import run_baseline_split80_permodel_nested_multirun_all_models
from experiments.run_dcpl_split80_permodel import run_dcpl_split80_permodel

# Run baseline multirun all models
BASELINES = ["lr", "ridge", "rf_light", "nn", "llm_pilot"]

run_baseline_split80_permodel_nested_multirun_all_models(
    baselines=BASELINES,
    n_runs=30,
    base_seed=42,
    seed_stride=1000,
    per_model_dir="data/llm_pilot_data/raw_data/per_model",
    target="Target_throughput_tokens_per_sec",
    test_size=0.20,
    results_root="results/runs/30x_baselines/",
    run_name="baseline_split80",
)

# Run DCPL 30x multirun
run_dcpl_split80_permodel_nx(
    n_runs=30,
    base_seed=42,
    seed_stride=1000,
    per_model_dir="data/llm_pilot_data/raw_data/per_model",
    target="Target_throughput_tokens_per_sec",
    gate_kind="ridge",
    inner_splits=5,
    test_size=0.20,
    results_root="results/runs/30x_dcpl/",
    run_name="dcpl_split80",
)

run_dcpl_split80_permodel(
    per_model_dir="data/llm_pilot_data/raw_data/per_model",
    target="Target_throughput_tokens_per_sec",
    gate_kind="ridge",
    inner_splits=5,
    test_size=0.20,
    random_state=42,
    results_root="results/runs/dcpl_30_runs/",
    run_name="dcpl_split80",
)
