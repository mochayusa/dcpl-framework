#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
IN_CSV = Path("results/scott_knott_final_analysis/scott_knott_all_methods_summary.csv")
OUT_DIR = Path("results/scott_knott_final_analysis/figs_rank_heatmap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# If None: plot all metrics found in the file
ONLY_METRIC = None  # e.g., "MAE" or "R2" or "RMSE" or "MRE"

# If you want to force a method order (optional), fill this list.
# Otherwise methods are ordered by average rank (best to worst).
FORCED_METHOD_ORDER = None  # e.g., ["dcpl_gate_ridge","baseline_rf_light","baseline_nn",...]

# =========================
# HELPERS
# =========================
def _norm_metric(m: str) -> str:
    m = str(m).strip()
    # unify common variants
    m = m.replace("MRE(%)", "MRE").replace("MRE%", "MRE").replace("MRE (%)", "MRE")
    return m

def plot_rank_heatmap(pivot: pd.DataFrame, title: str, outpath: Path):
    """
    pivot: index=LLM, columns=Method, values=Rank (lower better)
    """
    data = pivot.to_numpy(dtype=float)

    # Figure size scales with matrix size
    n_rows, n_cols = data.shape
    fig_w = max(8.0, 0.55 * n_cols)
    fig_h = max(4.5, 0.45 * n_rows)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    # Use a light-to-dark map; lower ranks should look "better".
    # We'll use a normal colormap and invert by setting vmin/vmax normally.
    im = ax.imshow(data, aspect="auto", interpolation="nearest")

    # Ticks
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(pivot.index, fontsize=9)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Method / Algorithm", fontsize=11)
    ax.set_ylabel("LLM", fontsize=11)

    # Annotate each cell with rank (optional but often helpful in papers)
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Scott–Knott rank (1 = best group)", fontsize=10)

    # Tight layout + save
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def build_pivot_for_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[df["Metric"] == metric].copy()

    # pivot: rows=LLM, cols=Method, values=Rank
    pivot = sub.pivot_table(index="LLM", columns="Method", values="Rank", aggfunc="min")

    # Order methods
    if FORCED_METHOD_ORDER:
        cols = [c for c in FORCED_METHOD_ORDER if c in pivot.columns]
        # append any missing methods at the end
        cols += [c for c in pivot.columns if c not in cols]
        pivot = pivot[cols]
    else:
        avg_rank = pivot.mean(axis=0, skipna=True).sort_values()
        pivot = pivot[avg_rank.index.tolist()]

    # Order LLMs (optional: by best method rank or alphabetically)
    pivot = pivot.sort_index()

    return pivot

# =========================
# MAIN
# =========================
def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    # Standardise column names expected: LLM, Metric, Method, Rank
    # (If yours are lowercase etc, map them here.)
    required = {"LLM", "Metric", "Method", "Rank"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    df["Metric"] = df["Metric"].map(_norm_metric)
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")

    metrics = sorted(df["Metric"].dropna().unique().tolist())
    if ONLY_METRIC is not None:
        ONLY = _norm_metric(ONLY_METRIC)
        metrics = [m for m in metrics if m == ONLY]

    if not metrics:
        raise ValueError("No metrics found to plot (check ONLY_METRIC and your CSV Metric values).")

    for metric in metrics:
        pivot = build_pivot_for_metric(df, metric)

        # If you want missing cells to appear clearly, keep NaN.
        # If you prefer filling missing ranks with worst+1:
        # worst = int(np.nanmax(pivot.to_numpy())) if np.isfinite(np.nanmax(pivot.to_numpy())) else 1
        # pivot = pivot.fillna(worst + 1)

        outpath = OUT_DIR / f"rank_heatmap_{metric}.png"
        title = f"Scott–Knott ranks — {metric} (1 = best)"
        plot_rank_heatmap(pivot, title, outpath)
        print(f"[OK] Saved: {outpath}")

if __name__ == "__main__":
    main()
