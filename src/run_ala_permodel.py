from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from experiments.run_ala_split80_throughput import run_ala_split80_5runs


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

PER_MODEL_DIR = Path("data/llm_pilot_data/raw_data/per_model")
RESULTS_DIR   = Path("results/ala")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 1042, 2042, 3042, 4042]
TEST_SIZE = 0.20


# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------

def run_ala_permodel(
    per_model_dir: Path = PER_MODEL_DIR,
    results_dir: Path = RESULTS_DIR,
    seeds: list[int] = SEEDS,
    test_size: float = TEST_SIZE,
):
    per_model_dir = Path(per_model_dir)
    results_dir = Path(results_dir)

    files = sorted(per_model_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {per_model_dir}")

    all_rows = []

    for csv_path in files:
        model_tag = csv_path.stem
        print(f"\n==============================")
        print(f" Running ALA for model: {model_tag}")
        print(f"==============================")

        out_csv = results_dir / f"{model_tag}_ala_split80_5runs.csv"

        df_runs = run_ala_split80_5runs(
            data_path=csv_path,
            seeds=seeds,
            test_size=test_size,
            out_csv=out_csv,
        )

        # aggregate per model (mean/std across seeds)
        summary = {
            "per_model_file": model_tag,
            "learner": "ALA",
            "cv": "split80_20",
            "target": "Target_throughput_tokens_per_sec",
            "n_runs": len(seeds),
        }

        for m in ["R2", "RMSE", "MAE", "MRE"]:
            summary[f"{m}_mean"] = float(df_runs[m].mean())
            summary[f"{m}_std"]  = float(df_runs[m].std(ddof=1))

        all_rows.append(summary)

    # --------------------------------------------------------
    # Global summary
    # --------------------------------------------------------
    global_df = pd.DataFrame(all_rows)
    global_csv = results_dir / "ala_permodel_split80_summary.csv"
    global_df.to_csv(global_csv, index=False)

    print(f"\n[DONE] Global ALA summary saved to: {global_csv}")
    return global_df


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    run_ala_permodel()
