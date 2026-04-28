from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Defaults
# ============================================================

DATA = "data/llm_pilot_data/preliminary/preliminary_sample_100k.csv"
OUTPUT_DIR = "results/preliminary/qualitative_interactions"

TARGET_COL = "Target_throughput_tokens_per_sec"

MODEL_SIZE_COL = "AI_model_n_parameters"
GPU_COL_CANDIDATES = ["NonAI_gpu_type", "NonAI_gpu"]
SEQ_COL = "Workload_n_input_tokens"
CONC_COL = "Workload_reqnum"


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def detect_gpu_col(df: pd.DataFrame) -> str:
    for col in GPU_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find GPU label column. Tried: {GPU_COL_CANDIDATES}"
    )


def safe_log10(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    return np.log10(series.clip(lower=eps))


def make_quantile_bins(
    series: pd.Series,
    q: int,
    precision: int = 3,
) -> pd.Series:
    return pd.qcut(series, q=q, duplicates="drop", precision=precision)


def format_interval_labels(index_like) -> list[str]:
    labels = []
    for item in index_like:
        labels.append(str(item))
    return labels


def median_pivot(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
) -> pd.DataFrame:
    out = (
        df.groupby([row_col, col_col], observed=False)[value_col]
        .median()
        .unstack(col_col)
        .sort_index()
    )
    return out


def dataframe_to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_format: str = "%.2f",
) -> str:
    latex = df.to_latex(
        index=True,
        escape=False,
        na_rep="--",
        float_format=lambda x: float_format % x if pd.notnull(x) else "--",
    )
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex}\n"
        "\\end{table}\n"
    )


def save_table_outputs(
    df: pd.DataFrame,
    out_csv: Path,
    out_tex: Path,
    caption: str,
    label: str,
    float_format: str = "%.2f",
) -> None:
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv)
    tex = dataframe_to_latex_table(df, caption=caption, label=label, float_format=float_format)
    out_tex.write_text(tex, encoding="utf-8")


def plot_heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    output_png: Path,
    annotate: bool = True,
) -> None:
    ensure_dir(output_png.parent)

    fig, ax = plt.subplots(figsize=(10, 6))
    arr = pivot_df.values.astype(float)

    im = ax.imshow(arr, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_xticklabels([str(c) for c in pivot_df.columns], rotation=30, ha="right")

    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_yticklabels([str(i) for i in pivot_df.index])

    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Median throughput (tokens/sec)")

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_violin(
    df: pd.DataFrame,
    x_col: str,
    hue_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_png: Path,
    max_points: int = 30000,
    random_state: int = 42,
) -> None:
    """
    Lightweight violin-like alternative using boxplots when seaborn is unavailable.
    Uses grouped boxplots with slight offsets by hue.
    """
    ensure_dir(output_png.parent)

    plot_df = df[[x_col, hue_col, y_col]].dropna().copy()
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(n=max_points, random_state=random_state)

    x_levels = list(pd.Index(plot_df[x_col].dropna().unique()))
    hue_levels = list(pd.Index(plot_df[hue_col].dropna().unique()))

    x_levels = sorted(x_levels, key=str)
    hue_levels = sorted(hue_levels, key=str)

    fig, ax = plt.subplots(figsize=(12, 6))

    base_positions = np.arange(len(x_levels))
    width = 0.8 / max(1, len(hue_levels))

    for h_idx, hue in enumerate(hue_levels):
        data = []
        positions = []

        for x_idx, x_val in enumerate(x_levels):
            vals = plot_df.loc[
                (plot_df[x_col] == x_val) & (plot_df[hue_col] == hue),
                y_col,
            ].dropna().values

            if len(vals) == 0:
                continue

            pos = base_positions[x_idx] - 0.4 + width / 2 + h_idx * width
            data.append(vals)
            positions.append(pos)

        if data:
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp["boxes"]:
                patch.set_alpha(0.5)
            for median in bp["medians"]:
                median.set_linewidth(1.5)

    ax.set_xticks(base_positions)
    ax.set_xticklabels([str(x) for x in x_levels], rotation=20, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend([str(h) for h in hue_levels], title=hue_col, loc="best")

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_line_trends(
    pivot_df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    output_png: Path,
) -> None:
    ensure_dir(output_png.parent)

    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = [str(c) for c in pivot_df.columns]
    x = np.arange(len(x_labels))

    for idx in pivot_df.index:
        y = pivot_df.loc[idx].values.astype(float)
        ax.plot(x, y, marker="o", label=str(idx))

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Group", loc="best")

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Study 1: AI × Non-AI
# ============================================================

def run_ai_nonai(
    df: pd.DataFrame,
    output_dir: Path,
    target_col: str,
    model_size_bins: int = 4,
) -> dict:
    gpu_col = detect_gpu_col(df)

    needed = [MODEL_SIZE_COL, gpu_col, target_col]
    work = df[needed].dropna().copy()

    work["model_size_bin"] = make_quantile_bins(work[MODEL_SIZE_COL], q=model_size_bins)
    pivot = median_pivot(work, "model_size_bin", gpu_col, target_col)

    save_table_outputs(
        pivot,
        out_csv=output_dir / "tables" / "ai_nonai_median_throughput.csv",
        out_tex=output_dir / "tables" / "ai_nonai_median_throughput.tex",
        caption="Median throughput (tokens/sec) across model-size bins and GPU types.",
        label="tab:ai_nonai_median_throughput",
    )

    plot_heatmap(
        pivot_df=pivot,
        title="AI × Non-AI: median throughput by model size and GPU type",
        xlabel="GPU type",
        ylabel="Model-size bin",
        output_png=output_dir / "figures" / "ai_nonai_heatmap.png",
    )

    plot_grouped_violin(
        df=work,
        x_col=gpu_col,
        hue_col="model_size_bin",
        y_col=target_col,
        title="AI × Non-AI: throughput distributions by GPU type and model-size bin",
        xlabel="GPU type",
        ylabel="Throughput (tokens/sec)",
        output_png=output_dir / "figures" / "ai_nonai_boxplot.png",
    )

    return {
        "rows_used": int(len(work)),
        "gpu_col": gpu_col,
        "model_size_bins": model_size_bins,
        "outputs": {
            "heatmap": str(output_dir / "figures" / "ai_nonai_heatmap.png"),
            "boxplot": str(output_dir / "figures" / "ai_nonai_boxplot.png"),
            "table_csv": str(output_dir / "tables" / "ai_nonai_median_throughput.csv"),
            "table_tex": str(output_dir / "tables" / "ai_nonai_median_throughput.tex"),
        },
    }


# ============================================================
# Study 2: AI × Workload
# ============================================================

def run_ai_workload(
    df: pd.DataFrame,
    output_dir: Path,
    target_col: str,
    model_size_bins: int = 4,
    seq_bins: int = 5,
) -> dict:
    needed = [MODEL_SIZE_COL, SEQ_COL, target_col]
    work = df[needed].dropna().copy()

    work["model_size_bin"] = make_quantile_bins(work[MODEL_SIZE_COL], q=model_size_bins)
    work["sequence_length_bin"] = make_quantile_bins(work[SEQ_COL], q=seq_bins)

    pivot = median_pivot(work, "model_size_bin", "sequence_length_bin", target_col)

    save_table_outputs(
        pivot,
        out_csv=output_dir / "tables" / "ai_workload_median_throughput.csv",
        out_tex=output_dir / "tables" / "ai_workload_median_throughput.tex",
        caption="Median throughput (tokens/sec) across model-size bins and sequence-length bins.",
        label="tab:ai_workload_median_throughput",
    )

    plot_heatmap(
        pivot_df=pivot,
        title="AI × Workload: median throughput by model size and sequence length",
        xlabel="Sequence-length bin",
        ylabel="Model-size bin",
        output_png=output_dir / "figures" / "ai_workload_heatmap.png",
    )

    plot_line_trends(
        pivot_df=pivot,
        title="AI × Workload: sequence-length sensitivity by model-size bin",
        xlabel="Sequence-length bin",
        ylabel="Median throughput (tokens/sec)",
        output_png=output_dir / "figures" / "ai_workload_trends.png",
    )

    return {
        "rows_used": int(len(work)),
        "model_size_bins": model_size_bins,
        "sequence_length_bins": seq_bins,
        "outputs": {
            "heatmap": str(output_dir / "figures" / "ai_workload_heatmap.png"),
            "trend": str(output_dir / "figures" / "ai_workload_trends.png"),
            "table_csv": str(output_dir / "tables" / "ai_workload_median_throughput.csv"),
            "table_tex": str(output_dir / "tables" / "ai_workload_median_throughput.tex"),
        },
    }


# ============================================================
# Study 3: Non-AI × Workload
# ============================================================

def run_nonai_workload(
    df: pd.DataFrame,
    output_dir: Path,
    target_col: str,
    seq_bins: int = 5,
    conc_bins: int = 4,
) -> dict:
    gpu_col = detect_gpu_col(df)

    needed = [gpu_col, SEQ_COL, CONC_COL, target_col]
    work = df[needed].dropna().copy()

    work["sequence_length_bin"] = make_quantile_bins(work[SEQ_COL], q=seq_bins)
    work["concurrency_bin"] = make_quantile_bins(work[CONC_COL], q=conc_bins)

    pivot_seq = median_pivot(work, "sequence_length_bin", gpu_col, target_col)
    pivot_conc = median_pivot(work, "concurrency_bin", gpu_col, target_col)

    save_table_outputs(
        pivot_seq,
        out_csv=output_dir / "tables" / "nonai_workload_seq_median_throughput.csv",
        out_tex=output_dir / "tables" / "nonai_workload_seq_median_throughput.tex",
        caption="Median throughput (tokens/sec) by GPU type and sequence-length bin.",
        label="tab:nonai_workload_seq_median_throughput",
    )

    save_table_outputs(
        pivot_conc,
        out_csv=output_dir / "tables" / "nonai_workload_conc_median_throughput.csv",
        out_tex=output_dir / "tables" / "nonai_workload_conc_median_throughput.tex",
        caption="Median throughput (tokens/sec) by GPU type and concurrency bin.",
        label="tab:nonai_workload_conc_median_throughput",
    )

    plot_heatmap(
        pivot_df=pivot_seq,
        title="Non-AI × Workload: median throughput by sequence length and GPU type",
        xlabel="GPU type",
        ylabel="Sequence-length bin",
        output_png=output_dir / "figures" / "nonai_workload_seq_heatmap.png",
    )

    plot_heatmap(
        pivot_df=pivot_conc,
        title="Non-AI × Workload: median throughput by concurrency and GPU type",
        xlabel="GPU type",
        ylabel="Concurrency bin",
        output_png=output_dir / "figures" / "nonai_workload_conc_heatmap.png",
    )

    return {
        "rows_used": int(len(work)),
        "gpu_col": gpu_col,
        "sequence_length_bins": seq_bins,
        "concurrency_bins": conc_bins,
        "outputs": {
            "seq_heatmap": str(output_dir / "figures" / "nonai_workload_seq_heatmap.png"),
            "conc_heatmap": str(output_dir / "figures" / "nonai_workload_conc_heatmap.png"),
            "seq_table_csv": str(output_dir / "tables" / "nonai_workload_seq_median_throughput.csv"),
            "seq_table_tex": str(output_dir / "tables" / "nonai_workload_seq_median_throughput.tex"),
            "conc_table_csv": str(output_dir / "tables" / "nonai_workload_conc_median_throughput.csv"),
            "conc_table_tex": str(output_dir / "tables" / "nonai_workload_conc_median_throughput.tex"),
        },
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DATA)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--target", type=str, default=TARGET_COL)
    parser.add_argument("--model-size-bins", type=int, default=4)
    parser.add_argument("--seq-bins", type=int, default=5)
    parser.add_argument("--conc-bins", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("Loading data...")
    df = load_data(args.input)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found.")

    required_any = [MODEL_SIZE_COL, SEQ_COL, CONC_COL]
    missing = [c for c in required_any if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Running AI × Non-AI analysis...")
    ai_nonai_meta = run_ai_nonai(
        df=df,
        output_dir=output_dir / "ai_nonai",
        target_col=args.target,
        model_size_bins=args.model_size_bins,
    )

    print("Running AI × Workload analysis...")
    ai_workload_meta = run_ai_workload(
        df=df,
        output_dir=output_dir / "ai_workload",
        target_col=args.target,
        model_size_bins=args.model_size_bins,
        seq_bins=args.seq_bins,
    )

    print("Running Non-AI × Workload analysis...")
    nonai_workload_meta = run_nonai_workload(
        df=df,
        output_dir=output_dir / "nonai_workload",
        target_col=args.target,
        seq_bins=args.seq_bins,
        conc_bins=args.conc_bins,
    )

    meta = {
        "input": args.input,
        "target_col": args.target,
        "model_size_bins": args.model_size_bins,
        "sequence_bins": args.seq_bins,
        "concurrency_bins": args.conc_bins,
        "ai_nonai": ai_nonai_meta,
        "ai_workload": ai_workload_meta,
        "nonai_workload": nonai_workload_meta,
    }

    save_json(meta, output_dir / "qualitative_interactions_metadata.json")
    print(f"Done. Outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()