#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import subprocess
from pathlib import Path

# ======================================================
# CONFIG
# ======================================================
MERGED_CSV = Path(
    "results/scott_knott_merged_input/merged_baseline_dcpl_gate_dal.csv"
)

RSCRIPT = "/opt/homebrew/bin/Rscript"
R_WRAPPER = "src/analysis/sk_esd_wrapper.R"

OUT_DIR = Path("results/scott_knott_final_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["R2", "MAE", "RMSE", "MRE"]
HIGHER_IS_BETTER = {"R2": True, "MAE": False, "RMSE": False, "MRE": False}

MAX_RUNS = 30  # enforce 30 runs per method


# ======================================================
# Helpers
# ======================================================
def iqr(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return np.percentile(arr, 75) - np.percentile(arr, 25)


def pad_matrix(d):
    max_len = max(len(v) for v in d.values())
    rect = {}
    for k, v in d.items():
        v = list(v)
        if len(v) < max_len:
            v += [np.nan] * (max_len - len(v))
        rect[k] = v
    return pd.DataFrame(rect)


def call_scott_knott(df_matrix, tmp_matrix, tmp_ranks):
    df_matrix.to_csv(tmp_matrix, index=False)
    subprocess.run([RSCRIPT, R_WRAPPER, str(tmp_matrix), str(tmp_ranks)], check=True)
    rk = pd.read_csv(tmp_ranks)
    return pd.Series(rk["Group"].values, index=rk["Method"]).astype(int)


def remap_best_to_1(groups, means, higher_is_better):
    unique = sorted(groups.unique())
    if len(unique) <= 1:
        return groups.apply(lambda _: 1)

    score = {}
    for gid in unique:
        methods = groups[groups == gid].index
        score[gid] = np.mean([means[m] for m in methods])

    ordered = sorted(score.items(), key=lambda x: x[1], reverse=higher_is_better)
    mapping = {gid: i+1 for i, (gid, _) in enumerate(ordered)}

    return groups.map(mapping)


# ======================================================
# MAIN
# ======================================================
print("Loading merged CSV:", MERGED_CSV)
df = pd.read_csv(MERGED_CSV)

print("Total rows:", len(df))
print("Unique LLM:", df["per_model_file"].nunique())
print("Unique methods:", df["method"].nunique())

llms = sorted(df["per_model_file"].unique())

tmp_dir = OUT_DIR / "_tmp"
tmp_dir.mkdir(parents=True, exist_ok=True)

all_rows = []

for llm in llms:

    llm_df = df[df["per_model_file"] == llm].copy()
    print("\n=====================================")
    print("LLM:", llm)
    print("=====================================")

    for metric in METRICS:

        if metric not in llm_df.columns:
            continue

        method_vals = {}

        for method, g in llm_df.groupby("method"):
            s = pd.to_numeric(g[metric], errors="coerce").dropna()
            s = s.iloc[:MAX_RUNS]
            if len(s) >= 2:
                method_vals[method] = s.values

        if len(method_vals) < 2:
            continue

        matrix = pad_matrix(method_vals)

        tmp_matrix = tmp_dir / f"{llm}__{metric}_matrix.csv"
        tmp_ranks = tmp_dir / f"{llm}__{metric}_ranks.csv"

        groups = call_scott_knott(matrix, tmp_matrix, tmp_ranks)

        means = {m: np.mean(v) for m, v in method_vals.items()}
        means_series = pd.Series(means)

        groups = remap_best_to_1(
            groups,
            means_series,
            higher_is_better=HIGHER_IS_BETTER[metric]
        )

        rows = []

        for method, vals in method_vals.items():
            arr = np.asarray(vals)
            rows.append({
                "LLM": llm,
                "Metric": metric,
                "Method": method,
                "Rank": int(groups[method]),
                "Mean": np.mean(arr),
                "Std": np.std(arr),
                "IQR": iqr(arr),
                "n_runs": len(arr)
            })

        result_df = pd.DataFrame(rows)

        if HIGHER_IS_BETTER[metric]:
            result_df = result_df.sort_values(["Rank", "Mean"], ascending=[True, False])
        else:
            result_df = result_df.sort_values(["Rank", "Mean"], ascending=[True, True])

        print("\n--- Metric:", metric, "---")
        print(result_df[["Method", "Rank", "Mean", "Std", "IQR"]].to_string(index=False))

        best = result_df[result_df["Rank"] == 1]["Method"].tolist()
        print("Best group:", ", ".join(best))

        all_rows.extend(rows)

# Save global summary
global_df = pd.DataFrame(all_rows)
global_df.to_csv(OUT_DIR / "scott_knott_all_methods_summary.csv", index=False)

print("\n[OK] Saved final analysis to:", OUT_DIR)
