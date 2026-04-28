# src/run_roofline_permodel.py

from __future__ import annotations
from pathlib import Path
import pandas as pd

from experiments.run_roofline_split80_throughput import run_5runs
from roofline.constants import COL_TARGET

PER_MODEL_DIR = Path("data/llm_pilot_data/raw_data/per_model")
RESULTS_DIR = Path("results/roofline/per_model")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 1042, 2042, 3042, 4042]
TEST_SIZE = 0.20

def main():
    files = sorted(PER_MODEL_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {PER_MODEL_DIR}")

    summaries = []

    for p in files:
        model_tag = p.stem
        print(f"\n==============================")
        print(f" Roofline-LR per-model: {model_tag}")
        print(f"==============================")

        out_csv = RESULTS_DIR / f"{model_tag}_roofline_lr_split80_5runs.csv"
        df_runs = run_5runs(p, seeds=SEEDS, out_csv=out_csv, test_size=TEST_SIZE)

        row = {
            "per_model_file": model_tag,
            "learner": "roofline_lr",
            "cv": "split80_20",
            "target": COL_TARGET,
            "n_runs": len(SEEDS),
        }
        for m in ["R2", "MAE", "RMSE", "MRE"]:
            row[f"{m}_mean"] = float(df_runs[m].mean())
            row[f"{m}_std"] = float(df_runs[m].std(ddof=1))
        summaries.append(row)

    g = pd.DataFrame(summaries)
    global_csv = RESULTS_DIR.parent / "roofline_lr_permodel_split80_summary.csv"
    g.to_csv(global_csv, index=False)
    print(f"\n[DONE] Global summary: {global_csv}")

if __name__ == "__main__":
    main()
