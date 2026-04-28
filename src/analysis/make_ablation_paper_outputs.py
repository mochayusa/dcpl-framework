from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
RUNS_ROOT = Path("results/runs")
OUT_DIR = Path("results/analysis/ablation_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")

ABLATION_ORDER = [
    "M1_no_interaction",
    "M2a_add_AIxNonAI",
    "M2b_add_AIxN_and_AIxW",
    "M3_full",
]

ABLATION_LABELS = {
    "M1_no_interaction": "M1",
    "M2a_add_AIxNonAI": "M2a",
    "M2b_add_AIxN_and_AIxW": "M2b",
    "M3_full": "M3",
}

MODEL_ORDER = [
    "data_EleutherAI_gpt-neox-20b",
    "data_Salesforce_codegen2-16B",
    "data_bigcode_starcoder",
    "data_bigscience_mt0-xxl",
    "data_google_flan-t5-xl",
    "data_google_flan-t5-xxl",
    "data_google_flan-ul2",
    "data_ibm_mpt-7b-instruct2",
    "data_llama-13b",
    "data_llama-7b",
]

MODEL_LABELS = {
    "data_EleutherAI_gpt-neox-20b": "EleutherAI\\_gpt-neox-20b",
    "data_Salesforce_codegen2-16B": "Salesforce\\_codegen2-16B",
    "data_bigcode_starcoder": "bigcode\\_starcoder",
    "data_bigscience_mt0-xxl": "bigscience\\_mt0-xxl",
    "data_google_flan-t5-xl": "google\\_flan-t5-xl",
    "data_google_flan-t5-xxl": "google\\_flan-t5-xxl",
    "data_google_flan-ul2": "google\\_flan-ul2",
    "data_ibm_mpt-7b-instruct2": "ibm\\_mpt-7b-instruct2",
    "data_llama-13b": "llama-13b",
    "data_llama-7b": "llama-7b",
}

DEFAULT_TARGET = "Target_throughput_tokens_per_sec"


# =========================================================
# HELPERS
# =========================================================
def is_run_dir(path: Path) -> bool:
    return path.is_dir() and RUN_DIR_PATTERN.match(path.name) is not None


def format_mean_std(mean_val: float, std_val: float, decimals: int = 3) -> str:
    if pd.isna(mean_val):
        return "--"
    if pd.isna(std_val):
        return f"{mean_val:.{decimals}f}"
    return f"{mean_val:.{decimals}f}$\\pm${std_val:.{decimals}f}"


def load_and_stack_runs(
    runs_root: Path,
    target_filter: str | None = DEFAULT_TARGET,
) -> pd.DataFrame:
    run_dirs = sorted([p for p in runs_root.iterdir() if is_run_dir(p)])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {runs_root}")

    frames = []
    audit_rows = []

    for run_idx, run_dir in enumerate(run_dirs, start=1):
        csv_path = run_dir / "ablation_metrics_all.csv"

        if not csv_path.exists():
            audit_rows.append(
                {"run_id": run_dir.name, "run_idx": run_idx, "status": "missing_csv", "n_rows": 0}
            )
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            audit_rows.append(
                {
                    "run_id": run_dir.name,
                    "run_idx": run_idx,
                    "status": f"read_error: {type(e).__name__}: {e}",
                    "n_rows": 0,
                }
            )
            continue

        df["run_id"] = run_dir.name
        df["run_idx"] = run_idx

        if target_filter is not None and "target" in df.columns:
            df = df[df["target"] == target_filter].copy()

        audit_rows.append(
            {"run_id": run_dir.name, "run_idx": run_idx, "status": "ok", "n_rows": len(df)}
        )
        frames.append(df)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUT_DIR / "ablation_runs_audit.csv", index=False)

    if not frames:
        raise RuntimeError("No valid run data collected.")

    stacked = pd.concat(frames, ignore_index=True)

    if "experiment" in stacked.columns:
        stacked["experiment"] = pd.Categorical(
            stacked["experiment"], categories=ABLATION_ORDER, ordered=True
        )
    if "model_tag" in stacked.columns:
        stacked["model_tag"] = pd.Categorical(
            stacked["model_tag"], categories=MODEL_ORDER, ordered=True
        )

    return stacked


def recompute_improvement_deltas(stacked: pd.DataFrame) -> pd.DataFrame:
    """
    Create paper-friendly deltas where positive always means better than M1.
    """
    needed = {"run_id", "model_tag", "target", "experiment", "R2", "RMSE"}
    missing = needed - set(stacked.columns)
    if missing:
        raise ValueError(f"Missing required columns for delta recomputation: {sorted(missing)}")

    key_cols = ["run_id", "model_tag", "target"]

    m1 = stacked[stacked["experiment"] == "M1_no_interaction"][
        key_cols + ["R2", "RMSE"]
    ].copy()

    m1 = m1.rename(columns={"R2": "M1_R2", "RMSE": "M1_RMSE"})

    out = stacked.merge(m1, on=key_cols, how="left")

    # positive always means improvement
    out["ΔR2_improve"] = out["R2"] - out["M1_R2"]
    out["ΔRMSE_improve"] = out["M1_RMSE"] - out["RMSE"]

    return out


def make_summary(stacked: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in [
            "R2", "RMSE",
            "ΔR2", "ΔRMSE",
            "ΔR2_improve", "ΔRMSE_improve",
        ]
        if c in stacked.columns
    ]

    group_cols = ["model_tag", "target", "experiment"]
    for optional in ["learner", "cv"]:
        if optional in stacked.columns:
            group_cols.append(optional)

    summary = (
        stacked
        .groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        f"{a}_{b}" if b else a
        for a, b in summary.columns.to_flat_index()
    ]

    rename_map = {
        "model_tag_": "model_tag",
        "target_": "target",
        "experiment_": "experiment",
        "learner_": "learner",
        "cv_": "cv",
    }
    summary = summary.rename(columns=rename_map)
    return summary


def make_global_summary(stacked: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in ["R2", "RMSE", "ΔR2_improve", "ΔRMSE_improve"]
        if c in stacked.columns
    ]

    group_cols = ["target", "experiment"]
    global_summary = (
        stacked
        .groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    global_summary.columns = [
        f"{a}_{b}" if b else a
        for a, b in global_summary.columns.to_flat_index()
    ]
    global_summary = global_summary.rename(
        columns={"target_": "target", "experiment_": "experiment"}
    )
    return global_summary


def build_metric_table_df(
    summary: pd.DataFrame,
    metric: str,
    decimals: int = 3,
    bold_best: bool = True,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    rows = []

    for model in MODEL_ORDER:
        sub = summary[summary["model_tag"] == model].copy()
        if sub.empty:
            continue

        row = {"Model": MODEL_LABELS.get(model, model)}
        means = {}

        for exp in ABLATION_ORDER:
            ss = sub[sub["experiment"] == exp]
            if ss.empty:
                row[ABLATION_LABELS[exp]] = "--"
                continue

            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            mean_val = ss[mean_col].iloc[0] if mean_col in ss.columns else np.nan
            std_val = ss[std_col].iloc[0] if std_col in ss.columns else np.nan

            means[exp] = mean_val
            row[ABLATION_LABELS[exp]] = format_mean_std(mean_val, std_val, decimals)

        valid_means = {k: v for k, v in means.items() if pd.notna(v)}
        if bold_best and valid_means:
            if higher_is_better:
                best_exp = max(valid_means, key=valid_means.get)
            else:
                best_exp = min(valid_means, key=valid_means.get)

            best_label = ABLATION_LABELS[best_exp]
            if row[best_label] != "--":
                row[best_label] = f"\\textbf{{{row[best_label]}}}"

        rows.append(row)

    return pd.DataFrame(rows)


def df_to_latex_booktabs(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_align: str | None = None,
) -> str:
    if column_align is None:
        column_align = "l" + "c" * (len(df.columns) - 1)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{column_align}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(df.columns) + " \\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.tolist()) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_win_count_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many per-model datasets each variant wins on.
    Higher is better for R2, lower is better for RMSE.
    """
    rows = []

    for metric, higher_is_better in [("R2", True), ("RMSE", False)]:
        counts = {exp: 0 for exp in ABLATION_ORDER}

        for model in MODEL_ORDER:
            sub = summary[summary["model_tag"] == model].copy()
            if sub.empty:
                continue

            mean_col = f"{metric}_mean"
            if mean_col not in sub.columns:
                continue

            metric_map = {}
            for exp in ABLATION_ORDER:
                ss = sub[sub["experiment"] == exp]
                if ss.empty:
                    continue
                metric_map[exp] = ss[mean_col].iloc[0]

            metric_map = {k: v for k, v in metric_map.items() if pd.notna(v)}
            if not metric_map:
                continue

            best_exp = max(metric_map, key=metric_map.get) if higher_is_better else min(metric_map, key=metric_map.get)
            counts[best_exp] += 1

        row = {"Metric": metric}
        for exp in ABLATION_ORDER:
            row[ABLATION_LABELS[exp]] = counts[exp]
        rows.append(row)

    return pd.DataFrame(rows)


def make_global_table_df(global_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for exp in ABLATION_ORDER:
        ss = global_summary[global_summary["experiment"] == exp]
        if ss.empty:
            continue

        row = {"Variant": ABLATION_LABELS[exp]}

        for metric in ["R2", "RMSE", "ΔR2_improve", "ΔRMSE_improve"]:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in ss.columns and std_col in ss.columns:
                row[metric] = format_mean_std(
                    ss[mean_col].iloc[0],
                    ss[std_col].iloc[0],
                    decimals=3,
                )

        rows.append(row)

    return pd.DataFrame(rows)


def plot_delta_vs_m1(
    summary: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    out_path: Path,
):
    plot_exps = ["M2a_add_AIxNonAI", "M2b_add_AIxN_and_AIxW", "M3_full"]

    if metric_col not in summary.columns:
        print(f"[WARN] Plot skipped; column not found: {metric_col}")
        return

    plt.figure(figsize=(10, 4.8))

    x = np.arange(len(MODEL_ORDER))
    width = 0.24

    for i, exp in enumerate(plot_exps):
        sub = summary[summary["experiment"] == exp].copy()
        vals = []

        for model in MODEL_ORDER:
            ss = sub[sub["model_tag"] == model]
            vals.append(ss[metric_col].iloc[0] if not ss.empty else np.nan)

        offset = (i - 1) * width
        plt.bar(x + offset, vals, width=width, label=ABLATION_LABELS[exp])

    plt.axhline(0.0, linewidth=1.0)
    plt.xticks(x, [MODEL_LABELS.get(m, m) for m in MODEL_ORDER], rotation=45, ha="right")
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("[1/6] Loading and stacking runs...")
    stacked = load_and_stack_runs(RUNS_ROOT, target_filter=DEFAULT_TARGET)

    print("[2/6] Recomputing improvement deltas...")
    stacked = recompute_improvement_deltas(stacked)

    stacked.to_csv(OUT_DIR / "ablation_all_runs_stacked.csv", index=False)

    print("[3/6] Making summaries...")
    summary = make_summary(stacked)
    summary.to_csv(OUT_DIR / "ablation_summary_per_model.csv", index=False)

    global_summary = make_global_summary(stacked)
    global_summary.to_csv(OUT_DIR / "ablation_summary_global.csv", index=False)

    print("[4/6] Building paper tables...")
    # R2 table
    r2_table = build_metric_table_df(
        summary=summary,
        metric="R2",
        decimals=3,
        bold_best=True,
        higher_is_better=True,
    )
    r2_table.to_csv(OUT_DIR / "table_ablation_r2.csv", index=False)
    (OUT_DIR / "table_ablation_r2.tex").write_text(
        df_to_latex_booktabs(
            r2_table,
            caption=(
                "Ablation study across per-LLM datasets over 30 repeated holdout runs. "
                "M1 uses only main effects, while M2a--M3 progressively add interaction groups. "
                "Best mean R$^2$ per row is highlighted."
            ),
            label="tab:ablation_r2",
        ),
        encoding="utf-8",
    )

    # RMSE table
    rmse_table = build_metric_table_df(
        summary=summary,
        metric="RMSE",
        decimals=3,
        bold_best=True,
        higher_is_better=False,
    )
    rmse_table.to_csv(OUT_DIR / "table_ablation_rmse.csv", index=False)
    (OUT_DIR / "table_ablation_rmse.tex").write_text(
        df_to_latex_booktabs(
            rmse_table,
            caption=(
                "Ablation study across per-LLM datasets over 30 repeated holdout runs. "
                "Lower RMSE is better. Best mean RMSE per row is highlighted."
            ),
            label="tab:ablation_rmse",
        ),
        encoding="utf-8",
    )

    # Win count
    win_df = make_win_count_table(summary)
    win_df.to_csv(OUT_DIR / "table_ablation_win_counts.csv", index=False)
    (OUT_DIR / "table_ablation_win_counts.tex").write_text(
        df_to_latex_booktabs(
            win_df,
            caption=(
                "Number of per-LLM datasets on which each ablation variant achieves "
                "the best mean result over 30 repeated runs."
            ),
            label="tab:ablation_win_counts",
        ),
        encoding="utf-8",
    )

    # Global table
    global_table = make_global_table_df(global_summary)
    global_table.to_csv(OUT_DIR / "table_ablation_global.csv", index=False)
    (OUT_DIR / "table_ablation_global.tex").write_text(
        df_to_latex_booktabs(
            global_table,
            caption=(
                "Global mean$\\pm$std summary of the ablation variants across all per-LLM datasets "
                "and repeated runs."
            ),
            label="tab:ablation_global",
        ),
        encoding="utf-8",
    )

    print("[5/6] Making representative plots...")
    plot_delta_vs_m1(
        summary=summary,
        metric_col="ΔR2_improve_mean",
        ylabel="Improvement in $R^2$ over M1",
        out_path=OUT_DIR / "fig_ablation_delta_r2.png",
    )

    plot_delta_vs_m1(
        summary=summary,
        metric_col="ΔRMSE_improve_mean",
        ylabel="Improvement in RMSE over M1",
        out_path=OUT_DIR / "fig_ablation_delta_rmse.png",
    )

    print("[6/6] Done.")
    print(f"Saved outputs to: {OUT_DIR.resolve()}")
    print("\nMain files for paper:")
    print(" - table_ablation_r2.tex")
    print(" - table_ablation_rmse.tex")
    print(" - table_ablation_win_counts.tex")
    print(" - table_ablation_global.tex")
    print(" - fig_ablation_delta_r2.png")
    print(" - fig_ablation_delta_rmse.png")


if __name__ == "__main__":
    main()