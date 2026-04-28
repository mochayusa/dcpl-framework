#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import subprocess
from pathlib import Path

# =========================
# CONFIG (NO CLI ARGS)
# =========================
BASELINE_CSV = Path(
    "results/runs/20260130_104106/30x_baselines/"
    "baseline_split80__MULTI_BASELINES__30x_base42/"
    "baseline_split80_permodel_ALL_learners_ALL_runs_stacked.csv"
)

RSCRIPT = "/opt/homebrew/bin/Rscript"
R_WRAPPER = "src/analysis/sk_esd_wrapper.R"

OUT_DIR = Path("results/scott_knott_uploaded_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics present in your baseline stacked file
METRICS = ["R2", "MAE", "RMSE", "MRE"]

# Direction of “better”
HIGHER_IS_BETTER = {"R2": True, "MAE": False, "RMSE": False, "MRE": False}

# Use only first N runs if you want fixed 30, else set to None
RUNS_LIMIT = 30  # set None to use all available rows


# =========================
# Helpers
# =========================
def llm_id_from_per_model_file(x: str) -> str:
    return Path(str(x)).stem


def iqr(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, 75) - np.percentile(vals, 25))


def pad_to_rectangular(cols):
    """cols: dict(method -> list/array) => DataFrame with equal-length columns (padded with NaN)."""
    max_len = max(len(v) for v in cols.values())
    rect = {}
    for k, v in cols.items():
        v = list(v)
        if len(v) < max_len:
            v += [np.nan] * (max_len - len(v))
        rect[k] = v
    df = pd.DataFrame(rect)
    df = df.apply(pd.to_numeric, errors="coerce").astype(float)
    df = df.dropna(axis=1, how="all")
    return df


def call_sk_esd_r(matrix_df: pd.DataFrame, tmp_matrix: Path, tmp_ranks: Path) -> pd.Series:
    """
    Write matrix to CSV -> call Rscript wrapper -> read back Method,Group.
    Returns pandas Series index=Method, value=Group(int)
    """
    matrix_df.to_csv(tmp_matrix, index=False)

    subprocess.run(
        [RSCRIPT, R_WRAPPER, str(tmp_matrix), str(tmp_ranks)],
        check=True
    )

    rk = pd.read_csv(tmp_ranks)
    return pd.Series(rk["Group"].values, index=rk["Method"].astype(str)).astype(int)


def remap_best_is_1(groups: pd.Series, means: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    ScottKnottESD group IDs are not guaranteed to be ordered best->worst.
    This remaps so that rank=1 always means best group based on mean.
    """
    unique = sorted(set(int(x) for x in groups.values if pd.notna(x)))
    if len(unique) <= 1:
        return groups.apply(lambda _: 1).astype(int)

    group_score = {}
    for gid in unique:
        methods = groups.index[groups.values == gid]
        if len(methods) == 0:
            continue
        group_score[gid] = float(np.mean([means[m] for m in methods]))

    ordered = sorted(group_score.items(), key=lambda kv: kv[1], reverse=higher_is_better)
    mapping = {gid: i + 1 for i, (gid, _) in enumerate(ordered)}
    return groups.map(mapping).astype(int)


# =========================
# Main
# =========================
print("Loading:", BASELINE_CSV)
df = pd.read_csv(BASELINE_CSV)

print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

# Derive LLM + Method + Run
df["LLM"] = df["per_model_file"].map(llm_id_from_per_model_file)
df["Method"] = "baseline_" + df["learner"].astype(str)
df["Run"] = pd.to_numeric(df["iteration"], errors="coerce")

llms = sorted(df["LLM"].astype(str).unique())
print(f"\n[INFO] Found {len(llms)} LLM(s):")
for x in llms:
    print(" -", x)

tmp_dir = OUT_DIR / "_tmp"
tmp_dir.mkdir(parents=True, exist_ok=True)

all_rows = []

for llm in llms:
    llm_df = df[df["LLM"].astype(str) == llm].copy()
    llm_rows = []

    print(f"\n==============================")
    print(f"LLM: {llm}")
    print(f"Rows for this LLM: {len(llm_df)}")
    print(f"==============================")

    for metric in METRICS:
        if metric not in llm_df.columns:
            continue

        # Collect values per method (30 runs)
        method_vals = {}
        for method, g in llm_df.groupby("Method"):
            s = pd.to_numeric(g[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
            if RUNS_LIMIT is not None:
                s = s.iloc[:RUNS_LIMIT]
            if len(s) >= 2:
                method_vals[method] = s.tolist()

        if len(method_vals) < 2:
            print(f"\n--- Metric: {metric} ---")
            print("[SKIP] Not enough methods with >=2 runs.")
            continue

        # Make numeric matrix
        matrix_df = pad_to_rectangular(method_vals)

        # Call Scott–Knott in R
        tmp_matrix = tmp_dir / f"{llm}__{metric}__matrix.csv"
        tmp_ranks = tmp_dir / f"{llm}__{metric}__ranks.csv"
        groups = call_sk_esd_r(matrix_df, tmp_matrix, tmp_ranks)

        # Remap so rank=1 always best (recommended for reporting)
        means = pd.Series({m: float(np.mean(v)) for m, v in method_vals.items()})
        groups = remap_best_is_1(groups, means, higher_is_better=HIGHER_IS_BETTER.get(metric, True))

        # Build per-metric summary table
        rows = []
        for method, vals in method_vals.items():
            arr = np.asarray(vals, dtype=float)
            rows.append({
                "LLM": llm,
                "Metric": metric,
                "Method": method,
                "Rank": int(groups.loc[method]) if method in groups.index else 999,
                "n_runs": int(arr.size),
                "Mean": float(np.mean(arr)),
                "Std": float(np.std(arr)),
                "IQR": iqr(arr),
            })

        metric_df = pd.DataFrame(rows)

        # Sort for display: rank ascending; then “better” by mean direction
        if HIGHER_IS_BETTER.get(metric, True):
            metric_df = metric_df.sort_values(["Rank", "Mean"], ascending=[True, False])
        else:
            metric_df = metric_df.sort_values(["Rank", "Mean"], ascending=[True, True])

        print(f"\n--- Metric: {metric} ---")
        print(metric_df[["Method", "Rank", "n_runs", "Mean", "Std", "IQR"]].to_string(index=False))

        # Best methods (rank=1)
        best_methods = metric_df[metric_df["Rank"] == 1]["Method"].tolist()
        if best_methods:
            print(f"Best group (Rank=1): {', '.join(best_methods)}")

        llm_rows.extend(rows)

    # Save per-LLM CSV (all metrics)
    if llm_rows:
        llm_out = pd.DataFrame(llm_rows).sort_values(["Metric", "Rank", "Method"])
        llm_path = OUT_DIR / f"{llm}_summary.csv"
        llm_out.to_csv(llm_path, index=False)
        print(f"\n[OK] Saved per-LLM summary CSV: {llm_path}")

        all_rows.extend(llm_rows)

# Save global CSV
all_df = pd.DataFrame(all_rows).sort_values(["LLM", "Metric", "Rank", "Method"])
global_path = OUT_DIR / "scott_knott_baseline_only_ALL.csv"
all_df.to_csv(global_path, index=False)
print(f"\n[OK] Saved global summary CSV: {global_path}")

# Optional global “how many times each method is best per metric”
print("\n=== GLOBAL BEST-GROUP COUNTS (Rank=1) ===")
for metric in METRICS:
    sub = all_df[all_df["Metric"] == metric]
    if sub.empty:
        continue
    counts = sub[sub["Rank"] == 1]["Method"].value_counts()
    print(f"\nMetric: {metric}")
    print(counts.to_string())
