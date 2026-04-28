#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
IN_CSV = Path("results/scott_knott_final_analysis/scott_knott_all_methods_summary.csv")
OUT_DIR = Path("results/scott_knott_final_analysis/latex_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_ORDER = ["R2", "MAE", "RMSE", "MRE"]  # adjust if your CSV uses MRE%
HIGHLIGHT_COLOR = "red!15"  # LaTeX xcolor
SHOW_RANK_IN_CELL = True    # include small (r=) in cell

# Formatting digits per metric
DECIMALS = {"R2": 3, "MAE": 2, "RMSE": 2, "MRE": 2}

# =========================
# Helpers
# =========================
def latex_escape(s: str) -> str:
    """Escape LaTeX special chars, especially underscores in model/method names."""
    s = str(s)
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("_", "\\_")
    s = s.replace("%", "\\%")
    s = s.replace("&", "\\&")
    s = s.replace("#", "\\#")
    s = s.replace("{", "\\{").replace("}", "\\}")
    s = s.replace("^", "\\textasciicircum{}")
    s = s.replace("~", "\\textasciitilde{}")
    return s

def norm_metric(m: str) -> str:
    m = str(m).strip()
    m = m.replace("MRE(%)", "MRE").replace("MRE%", "MRE").replace("MRE (%)", "MRE")
    return m

def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing one of {candidates}. Found columns: {list(df.columns)}")
    return None

def fmt_mean_std(metric: str, mean: float, std: float) -> str:
    d = DECIMALS.get(metric, 3)
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        return f"{mean:.{d}f}"
    # Use \pm format
    return f"{mean:.{d}f}$\\pm${std:.{d}f}"

def cell_text(metric: str, mean: float, std: float, rank: int | float | None) -> str:
    txt = fmt_mean_std(metric, mean, std)
    if SHOW_RANK_IN_CELL and (rank is not None) and (not pd.isna(rank)):
        txt = f"{txt} {{\\scriptsize (r={int(rank)})}}"
    return txt

def write_tex(path: Path, tex: str):
    path.write_text(tex, encoding="utf-8")
    print(f"[OK] Wrote: {path}")

# =========================
# Main LaTeX builders
# =========================
def build_metric_table(df: pd.DataFrame, metric: str) -> str:
    sub = df[df["Metric"] == metric].copy()
    if sub.empty:
        return ""

    llms = sorted(sub["LLM"].astype(str).unique().tolist())
    methods = sorted(sub["Method"].astype(str).unique().tolist())

    # Order methods by avg rank (lower better), then by name
    avg_rank = sub.groupby("Method")["Rank"].mean().sort_values()
    methods = avg_rank.index.astype(str).tolist()

    # Precompute best methods per LLM (rank==1)
    best_map = {}
    for llm in llms:
        t = sub[sub["LLM"].astype(str) == llm]
        best_methods = set(t.loc[t["Rank"] == 1, "Method"].astype(str).tolist())
        best_map[llm] = best_methods

    # Header
    col_spec = "l" + "c" * len(methods)
    header_cols = "LLM & " + " & ".join(latex_escape(m) for m in methods) + " \\\\"

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{Scott--Knott summary for metric {latex_escape(metric)} (Mean$\\pm$Std over runs). "
                 f"Cells with rank 1 are highlighted.}}")
    lines.append(f"\\label{{tab:sk_{metric.lower()}}}")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(header_cols)
    lines.append("\\midrule")

    # Body
    for llm in llms:
        row = [latex_escape(llm)]
        for method in methods:
            t = sub[(sub["LLM"].astype(str) == llm) & (sub["Method"].astype(str) == method)]
            if t.empty:
                row.append("-")
                continue
            mean = float(t["Mean"].iloc[0]) if "Mean" in t.columns else np.nan
            std  = float(t["Std"].iloc[0]) if "Std" in t.columns else np.nan
            rank = int(t["Rank"].iloc[0]) if "Rank" in t.columns else None

            txt = cell_text(metric, mean, std, rank)

            # Highlight rank-1 group
            if method in best_map.get(llm, set()):
                txt = f"\\cellcolor{{{HIGHLIGHT_COLOR}}}\\textbf{{{txt}}}"

            row.append(txt)
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)

def build_summary_table(df: pd.DataFrame, metrics: list[str]) -> str:
    # Count how many LLMs each method is rank-1 for each metric
    sub = df[df["Metric"].isin(metrics)].copy()
    methods = sorted(sub["Method"].astype(str).unique().tolist())

    counts = []
    for metric in metrics:
        t = sub[sub["Metric"] == metric]
        c = t[t["Rank"] == 1].groupby("Method")["LLM"].nunique()
        # ensure all methods present
        c = c.reindex(methods).fillna(0).astype(int)
        counts.append(c.rename(metric))
    mat = pd.concat(counts, axis=1)

    # highlight max per metric
    max_per_metric = mat.max(axis=0)

    col_spec = "l" + "c" * len(metrics)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{How many LLMs each algorithm ranked best (Rank=1) per metric. "
                 "Maximum per metric is highlighted.}")
    lines.append("\\label{tab:sk_rank1_counts}")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append("Algorithm & " + " & ".join(metrics) + " \\\\")
    lines.append("\\midrule")

    # optional: sort methods by total bests
    mat["__total__"] = mat.sum(axis=1)
    mat = mat.sort_values("__total__", ascending=False).drop(columns="__total__")

    for method, row in mat.iterrows():
        r = [latex_escape(method)]
        for metric in metrics:
            v = int(row[metric])
            cell = str(v)
            if v == int(max_per_metric[metric]) and v > 0:
                cell = f"\\cellcolor{{{HIGHLIGHT_COLOR}}}\\textbf{{{cell}}}"
            r.append(cell)
        lines.append(" & ".join(r) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)

# =========================
# Run
# =========================
def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Input not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    # Normalize / harmonize expected columns
    # expected: LLM, Metric, Method, Rank, Mean, Std, IQR, n_runs (some may vary)
    llm_col    = pick_col(df, ["LLM", "llm", "per_model_file", "model"])
    metric_col = pick_col(df, ["Metric", "metric"])
    method_col = pick_col(df, ["Method", "method", "learner", "algorithm"])
    rank_col   = pick_col(df, ["Rank", "r", "rank"])
    mean_col   = pick_col(df, ["Mean", "mean"])
    std_col    = pick_col(df, ["Std", "std"], required=False)

    # Rename into canonical names
    df = df.rename(columns={
        llm_col: "LLM",
        metric_col: "Metric",
        method_col: "Method",
        rank_col: "Rank",
        mean_col: "Mean",
    })
    if std_col is not None:
        df = df.rename(columns={std_col: "Std"})
    else:
        df["Std"] = np.nan

    df["Metric"] = df["Metric"].map(norm_metric)

    # Numeric coercion
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce").astype("Int64")
    df["Mean"] = pd.to_numeric(df["Mean"], errors="coerce")
    df["Std"]  = pd.to_numeric(df["Std"], errors="coerce")

    # Decide which metrics exist
    existing_metrics = sorted(df["Metric"].dropna().unique().tolist())
    metrics = [m for m in METRICS_ORDER if m in existing_metrics]
    if not metrics:
        metrics = existing_metrics  # fallback: whatever exists

    print("[INFO] Metrics:", metrics)
    print("[INFO] LLMs   :", df["LLM"].nunique())
    print("[INFO] Methods:", df["Method"].nunique())

    # Write one .tex per metric
    for metric in metrics:
        tex = build_metric_table(df, metric)
        if not tex:
            print(f"[WARN] Empty metric table for {metric}, skipped.")
            continue
        write_tex(OUT_DIR / f"table_sk_{metric}.tex", tex)

    # Summary counts table
    tex_sum = build_summary_table(df, metrics)
    write_tex(OUT_DIR / "table_sk_rank1_counts.tex", tex_sum)

    # Also write a master file that inputs all tables
    master_lines = []
    master_lines.append("% Auto-generated master include file")
    master_lines.append("\\usepackage[table]{xcolor}")
    master_lines.append("\\usepackage{booktabs}")
    master_lines.append("\\usepackage{colortbl}")
    master_lines.append("")
    for metric in metrics:
        master_lines.append(f"\\input{{{(OUT_DIR / f'table_sk_{metric}.tex').as_posix()}}}")
        master_lines.append("")
    master_lines.append(f"\\input{{{(OUT_DIR / 'table_sk_rank1_counts.tex').as_posix()}}}")
    write_tex(OUT_DIR / "all_scott_knott_tables.tex", "\n".join(master_lines))

    print("\n[DONE] LaTeX tables generated in:", OUT_DIR)

if __name__ == "__main__":
    main()
