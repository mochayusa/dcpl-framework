#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
IN_CSV = Path(
    "results/scott_knott_final_analysis/scott_knott_all_methods_summary.csv"
)

OUT_DIR = Path("results/scott_knott_final_analysis/figs/scott_knott")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["R2", "MAE", "RMSE", "MRE"]
HIGHER_IS_BETTER = {"R2": True, "MAE": False, "RMSE": False, "MRE": False}

DPI = 300


# =========================================================
# Publication style
# =========================================================
def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# =========================================================
# Load
# =========================================================
def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # Normalise column names
    if "Rank" in df.columns and "r" not in df.columns:
        df = df.rename(columns={"Rank": "r"})

    df["Metric"] = df["Metric"].str.replace("MRE(%)", "MRE", regex=False)
    df["Metric"] = df["Metric"].str.replace("MRE%", "MRE", regex=False)

    df["r"] = pd.to_numeric(df["r"])
    df["Mean"] = pd.to_numeric(df["Mean"])
    df["Std"] = pd.to_numeric(df["Std"])

    return df


# =========================================================
# Heatmap (ALL LLMs)
# =========================================================
def plot_heatmap(df, metric):

    sub = df[df["Metric"] == metric].copy()
    if sub.empty:
        return

    mat = sub.pivot_table(
        index="Method",
        columns="LLM",
        values="r",
        aggfunc="min"
    )

    method_order = mat.mean(axis=1).sort_values().index
    llm_order = sorted(mat.columns)

    mat = mat.loc[method_order, llm_order]

    fig_w = max(8, 0.5 * len(llm_order))
    fig_h = max(4, 0.35 * len(method_order))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat.values, aspect="auto")

    ax.set_xticks(np.arange(len(llm_order)))
    ax.set_xticklabels(llm_order, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(method_order)

    ax.set_title(f"Scott–Knott Rank Heatmap ({metric})")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rank (1 = best)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"heatmap_{metric}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"heatmap_{metric}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Heatmap saved: {metric}")


# =========================================================
# Ranked barplots (ALL LLMs)
# =========================================================
def plot_bar_for_llm(df, llm, metric):

    sub = df[(df["LLM"] == llm) & (df["Metric"] == metric)].copy()
    if sub.empty:
        return

    hib = HIGHER_IS_BETTER.get(metric, True)

    sub = sub.sort_values(
        by=["r", "Mean"],
        ascending=[True, not hib]
    )

    methods = sub["Method"].values
    means = sub["Mean"].values
    stds = sub["Std"].values
    ranks = sub["r"].values

    fig_h = max(3.5, 0.35 * len(methods))
    fig, ax = plt.subplots(figsize=(8, fig_h))

    y = np.arange(len(methods))

    ax.barh(y, means, xerr=stds, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()

    direction = "higher is better" if hib else "lower is better"
    ax.set_title(f"{llm} — {metric} ({direction})")
    ax.set_xlabel(metric)

    for i, (m, r) in enumerate(zip(means, ranks)):
        ax.text(m, i, f"  r={r}", va="center", fontsize=9)

    fig.tight_layout()

    fname = f"{llm}_{metric}".replace("/", "_")
    fig.savefig(OUT_DIR / f"bar_{fname}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"bar_{fname}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Barplot saved: {llm} — {metric}")


# =========================================================
# MAIN
# =========================================================
def main():

    set_pub_style()

    df = load_data(IN_CSV)

    llms = sorted(df["LLM"].unique())
    metrics = [m for m in METRICS if m in df["Metric"].unique()]

    print("LLMs:", len(llms))
    print("Metrics:", metrics)

    # 1️⃣ Heatmaps (global)
    for metric in metrics:
        plot_heatmap(df, metric)

    # 2️⃣ Barplots for ALL LLMs
    for llm in llms:
        for metric in metrics:
            plot_bar_for_llm(df, llm, metric)

    print("\n[DONE] All figures generated.")
    print("Saved in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
