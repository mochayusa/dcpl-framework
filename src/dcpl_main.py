# src/dcpl_main.py
import pandas as pd
from pathlib import Path

from dcpl.blocks_origin import get_blocks, PERF_COLS
from dcpl.interactions import build_all_interactions
from experiments import run_baseline, run_additive, run_interaction
from evaluation.cv import run_kfold
from evaluation.logo import run_logo
from utils.io import make_run_dir, save_predictions, save_summary, save_manifest
# import os

def main():
    # 1) Load data Change if you have your own dataset
    df = pd.read_csv("data/llm_pilot_data/final_data/historical_performance_data_enriched_final.csv", index_col=0)

    # 2) Create run directory
    run_dir = make_run_dir("results/runs")

    # 3) Define which experiments to run
    experiments = ["baseline", "additive", "interaction"]
    targets = ["throughput", "median_nttft", "median_itl"]
    cv_mode = "logo_model"  # or "kfold10"

    # 4) Precompute blocks (and interactions if needed)
    X_ai, X_nonai, X_wl = get_blocks(df)
    inter = build_all_interactions(df, X_ai.columns, X_nonai.columns, X_wl.columns)

    # 5) Run
    all_rows = []
    for target in targets:
        if cv_mode == "kfold10":
            # baseline
            pred, summary = run_kfold(run_baseline, df, target, X_ai, X_nonai, X_wl)
            save_predictions(run_dir, "baseline", "rf", "kfold10", target, pred)
            all_rows.append(summary)

            # additive
            pred, summary = run_kfold(run_additive, df, target, X_ai, X_nonai, X_wl)
            save_predictions(run_dir, "additive", "ridge", "kfold10", target, pred)
            all_rows.append(summary)

            # interaction
            pred, summary = run_kfold(run_interaction, df, target, X_ai, X_nonai, X_wl, inter)
            save_predictions(run_dir, "interaction", "ridge", "kfold10", target, pred)
            all_rows.append(summary)

        elif cv_mode == "logo_model":
            pred, summary = run_logo(run_interaction,
                                     df,
                                     target,
                                     groups=df["model"],
                                     X_ai=X_ai,
                                     X_nonai=X_nonai,
                                     X_wl=X_wl,
                                     interactions=inter
                                     )
            save_predictions(run_dir, "interaction", "ridge", "logoModel", target, pred)
            all_rows.append(summary)

    # 6) Save global summary + manifest
    save_summary(run_dir, all_rows)
    save_manifest(run_dir, {
        "experiments": experiments,
        "targets": targets,
        "cv_mode": cv_mode
    })

if __name__ == "__main__":
    main()
