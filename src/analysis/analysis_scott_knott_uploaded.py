import pandas as pd
import numpy as np
import subprocess
from pathlib import Path

# =========================
# CONFIG
# =========================
BASELINE_CSV = Path("results/runs/20260130_104106/30x_baselines/baseline_split80__MULTI_BASELINES__30x_base42/baseline_split80_permodel_ALL_learners_ALL_runs_stacked.csv")
RSCRIPT = "/opt/homebrew/bin/Rscript"  
R_WRAPPER = "src/analysis/sk_esd_wrapper.R"
OUT_DIR = Path("results/scott_knott_uploaded_output")
OUT_DIR.mkdir(exist_ok=True)

METRICS = ["R2", "MAE", "RMSE", "MRE"]
HIGHER_IS_BETTER = {"R2": True, "MAE": False, "RMSE": False, "MRE": False}

# =========================
# Load baseline file
# =========================
print("Loading:", BASELINE_CSV)
df = pd.read_csv(BASELINE_CSV)

print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

# OPTIONAL: filter specific target if needed
# df = df[df["target"] == "Target_throughput_tokens_per_sec"]

# Extract LLM ID (clean)
df["LLM"] = df["per_model_file"].apply(lambda x: Path(str(x)).stem)

# Method name
df["Method"] = "baseline_" + df["learner"].astype(str)

# Use iteration as run index
df["Run"] = pd.to_numeric(df["iteration"], errors="coerce")


# =========================
# Function to pad matrix
# =========================
def pad_to_rectangular(data_dict):
    max_len = max(len(v) for v in data_dict.values())
    matrix = {}
    for k, v in data_dict.items():
        v = list(v)
        if len(v) < max_len:
            v += [np.nan] * (max_len - len(v))
        matrix[k] = v
    return pd.DataFrame(matrix)


# =========================
# Run Scott–Knott per LLM
# =========================
all_results = []

for llm in sorted(df["LLM"].unique()):
    print(f"\nProcessing LLM: {llm}")

    df_llm = df[df["LLM"] == llm]

    for metric in METRICS:
        if metric not in df_llm.columns:
            continue

        method_values = {}

        for method, g in df_llm.groupby("Method"):
            vals = pd.to_numeric(g[metric], errors="coerce")
            vals = vals.replace([np.inf, -np.inf], np.nan).dropna()

            if len(vals) >= 2:
                method_values[method] = vals.values

        if len(method_values) < 2:
            continue

        # Convert to rectangular matrix
        obs_df = pad_to_rectangular(method_values)
        obs_df = obs_df.apply(pd.to_numeric, errors="coerce")

        # Save temp matrix
        tmp_matrix = OUT_DIR / f"{llm}_{metric}_matrix.csv"
        tmp_ranks  = OUT_DIR / f"{llm}_{metric}_ranks.csv"

        obs_df.to_csv(tmp_matrix, index=False)

        # Call R wrapper
        subprocess.run(
            [RSCRIPT, R_WRAPPER, str(tmp_matrix), str(tmp_ranks)],
            check=True
        )

        ranks_df = pd.read_csv(tmp_ranks)
        rank_map = dict(zip(ranks_df["Method"], ranks_df["Group"]))

        # Store results
        for method, vals in method_values.items():
            vals = np.array(vals)
            all_results.append({
                "LLM": llm,
                "Metric": metric,
                "Method": method,
                "Rank": int(rank_map.get(method, 999)),
                "Mean": float(np.mean(vals)),
                "Std": float(np.std(vals)),
                "IQR": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                "n_runs": len(vals)
            })


# =========================
# Save final result
# =========================
result_df = pd.DataFrame(all_results)
result_df = result_df.sort_values(["LLM", "Metric", "Rank"])

out_path = OUT_DIR / "scott_knott_baseline_only.csv"
result_df.to_csv(out_path, index=False)

print("\nSaved result to:", out_path)


# =========================
# Global summary
# =========================
print("\n=== GLOBAL SUMMARY ===")

for metric in METRICS:
    sub = result_df[result_df["Metric"] == metric]
    if len(sub) == 0:
        continue

    print(f"\nMetric: {metric}")

    rank1_counts = sub[sub["Rank"] == 1]["Method"].value_counts()
    print("Rank-1 count:")
    print(rank1_counts)

    avg_rank = sub.groupby("Method")["Rank"].mean().sort_values()
    print("\nAverage Rank:")
    print(avg_rank)