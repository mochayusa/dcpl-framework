#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import pandas as pd
import numpy as np

IN_CSV = Path("results/scott_knott_final_analysis/scott_knott_all_methods_summary.csv")

OUT_DIR = Path("results/scott_knott_final_analysis/latex_tables_per_llm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics in your summary file (adjust if needed)
METRICS = ["R2", "MAE", "RMSE", "MRE"]

# For nicer ordering of methods inside each metric table
HIGHER_IS_BETTER = {"R2": True, "MAE": False, "RMSE": False, "MRE": False}

# Expected columns in your summary CSV
COL_LLM   = "LLM"
COL_MET   = "Metric"
COL_METH  = "Method"
COL_RANK  = "Rank"   # or "r" (we handle both)
COL_MEAN  = "Mean"
COL_STD   = "Std"
COL_IQR   = "IQR"
COL_NRUNS = "n_runs"  # or "n_runs_used" (we handle both)

def latex_escape(s: str) -> str:
    s = str(s)
    return (s.replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("#", "\\#")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("$", "\\$")
            .replace("^", "\\^{}")
            .replace("~", "\\~{}"))

def safe_filename(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:200]

def fmt_mean_std(mean, std, digits=3) -> str:
    if pd.isna(mean):
        return "N/A"
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"

def load_and_normalise() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)

    # Rank column
    if COL_RANK not in df.columns and "r" in df.columns:
        df = df.rename(columns={"r": COL_RANK})

    # n_runs column
    if COL_NRUNS not in df.columns:
        if "n_runs_used" in df.columns:
            df = df.rename(columns={"n_runs_used": COL_NRUNS})

    required = {COL_LLM, COL_MET, COL_METH, COL_RANK, COL_MEAN, COL_STD}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound columns: {list(df.columns)}")

    # normalise metric naming variants
    df[COL_MET] = df[COL_MET].astype(str).str.replace("MRE(%)", "MRE").str.replace("MRE%", "MRE")

    # keep only requested metrics (if present)
    df = df[df[COL_MET].isin(METRICS)].copy()

    # enforce numeric
    for c in [COL_RANK, COL_MEAN, COL_STD]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if COL_IQR in df.columns:
        df[COL_IQR] = pd.to_numeric(df[COL_IQR], errors="coerce")
    else:
        df[COL_IQR] = np.nan

    if COL_NRUNS in df.columns:
        df[COL_NRUNS] = pd.to_numeric(df[COL_NRUNS], errors="coerce")
    else:
        df[COL_NRUNS] = np.nan

    return df

def metric_table_tex(llm: str, metric_df: pd.DataFrame, metric: str) -> str:
    # Sort: primarily by Rank, then by performance direction for readability
    hib = HIGHER_IS_BETTER.get(metric, True)

    # If rank ties exist, order tied methods by mean (better first)
    metric_df = metric_df.copy()
    metric_df["_mean_sort"] = metric_df[COL_MEAN]
    metric_df = metric_df.sort_values(
        by=[COL_RANK, "_mean_sort"],
        ascending=[True, not hib]  # if higher is better -> sort mean desc (ascending False)
    ).drop(columns=["_mean_sort"])

    lines = []
    lines.append(f"\\subsubsection*{{Metric: {latex_escape(metric)}}}")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.1}")
    lines.append(f"\\caption{{Scott--Knott per-method summary for {latex_escape(llm)} on {latex_escape(metric)} (Mean$\\pm$Std over runs).}}")
    lines.append(f"\\label{{tab:sk_{safe_filename(llm)}_{metric}}}")
    lines.append("\\begin{tabular}{l|c|c|c|c}")
    lines.append("\\hline")
    lines.append("\\textbf{Method} & \\textbf{Rank} & \\textbf{Mean$\\pm$Std} & \\textbf{IQR} & \\textbf{$n$} \\\\")
    lines.append("\\hline")

    # highlight rank==1 cells (method name + rank + value)
    for _, r in metric_df.iterrows():
        m = latex_escape(r[COL_METH])
        rk = int(r[COL_RANK]) if pd.notna(r[COL_RANK]) else 999
        ms = fmt_mean_std(r[COL_MEAN], r[COL_STD], digits=3)

        iqr = r[COL_IQR]
        iqr_s = "N/A" if pd.isna(iqr) else f"{iqr:.3f}"

        n = r[COL_NRUNS]
        n_s = "N/A" if pd.isna(n) else str(int(n))

        if rk == 1:
            # requires \usepackage[table]{xcolor} in preamble
            lines.append(f"\\rowcolor{{red!15}} {m} & \\textbf{{{rk}}} & \\textbf{{{ms}}} & {iqr_s} & {n_s} \\\\")
        else:
            lines.append(f"{m} & {rk} & {ms} & {iqr_s} & {n_s} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)

def per_llm_report_tex(df: pd.DataFrame, llm: str) -> str:
    sub = df[df[COL_LLM].astype(str) == str(llm)].copy()

    lines = []
    lines.append("% ================================")
    lines.append(f"% LLM: {llm}")
    lines.append("% ================================")
    lines.append(f"\\section*{{LLM: {latex_escape(llm)}}}")
    lines.append("")

    for metric in METRICS:
        mdf = sub[sub[COL_MET] == metric].copy()
        if mdf.empty:
            lines.append(f"\\subsubsection*{{Metric: {latex_escape(metric)}}}")
            lines.append("\\textit{No rows found for this metric.}")
            lines.append("")
            continue
        lines.append(metric_table_tex(llm, mdf, metric))

    return "\n".join(lines)

def main():
    df = load_and_normalise()

    llms = sorted(df[COL_LLM].astype(str).unique())
    print(f"[INFO] Found {len(llms)} LLM(s).")

    all_tex_parts = []
    for llm in llms:
        tex = per_llm_report_tex(df, llm)
        out_path = OUT_DIR / f"{safe_filename(llm)}__scott_knott_per_llm.tex"
        out_path.write_text(tex, encoding="utf-8")
        print(f"[OK] Wrote: {out_path}")
        all_tex_parts.append(tex)

    # optional: one combined file
    combined = OUT_DIR / "ALL_LLMs__scott_knott_per_llm_combined.tex"
    combined.write_text("\n\n\\newpage\n\n".join(all_tex_parts), encoding="utf-8")
    print(f"[OK] Wrote combined: {combined}")

    print("\nNOTE:")
    print("Add these in your LaTeX preamble:")
    print(r"  \usepackage{float}")
    print(r"  \usepackage[table]{xcolor}")

if __name__ == "__main__":
    main()
