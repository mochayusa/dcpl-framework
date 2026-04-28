from experiments.run_baseline import run_baseline_experiment
from experiments.run_additive import run_additive_experiment
from experiments.run_interaction import run_interaction_experiment
from experiments.run_gated_interaction import run_gated_interaction_experiment

DATA = "data/llm_pilot_data/final_data/historical_performance_data_enriched_final.csv"

# model = ['llm_pilot'] #  you can choose more than one from this --> 'rf_light', 'ridge', 'nn', 'lr', 'llm_pilot'
# for m in model:
#     print(f"Running baseline experiment with model: {m}")
#     run_baseline_experiment(DATA, cv_mode="logo_model", model_kind=m)

# run_additive_experiment(DATA, cv_mode="logo_model", model_kind="ridge")
# run_interaction_experiment(DATA, cv_mode="logo_model", base_kind="ridge", inter_kind="ridge")

# run_gated_interaction_experiment(
#         DATA,
#         targets=("throughput", "median_nttft", "median_itl"),
#         cv_mode="logo_model",      # strong generalisation by model
#         gate_kind="ridge",
#         inner_splits=5,
#         run_name="M2_gated_full"
#     )

import pandas as pd
from dcpl.blocks_origin import get_blocks
from dcpl.interactions import build_all_interactions
from experiments.run_ablation import run_ablation_experiments

df = pd.read_csv(DATA, index_col=0)

X_ai, X_nonai, X_wl = get_blocks(df)
inter = build_all_interactions(df, X_ai.columns, X_nonai.columns, X_wl.columns)

df_out = run_ablation_experiments(df, X_ai, X_nonai, X_wl, inter)

df_out.to_csv("results/ablation_results.csv", index=False)
print("[DONE] Ablation results saved to results/ablation_results.csv")
