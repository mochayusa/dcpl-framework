#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

IN_CSV = Path("results/scott_knott_final_analysis/scott_knott_all_methods_summary.csv")
OUT_DIR = Path("results/scott_knott_final_analysis/latex_tables_paired")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["R2", "MAE", "RMSE", "MRE"]

# highlight colours (match your style)
BEST_BG = "CBFECE"   # green-ish
# you can also define 2nd/3rd if you want later:
# SECOND_BG = "FFEFCA"
# THIRD_BG  = "C0E3F8"

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

def fmt_mean_std(mean, std, digits=3):
    if pd.isna(mean):
        return "N/A"
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({std:.{digits}f})"

def cell_r(rank):
    return "N/A" if pd.isna(rank) else str(int(rank))

def highlight_if_best(text: str, is_best: bool) -> str:
    if not is_best:
        return text
    return f"\\cellcolor[HTML]{{{BEST_BG}}}\\textbf{{{text}}}"

def build_paired_table(df: pd.DataFrame, metric: str) -> str:
    sub = df[df["Metric"].astype(str) == metric].copy()
    if sub.empty:
        return ""

    # canonicalise column names
    # accept either "Rank" or "r"
    if "Rank" not in sub.columns and "r" in sub.columns:
        sub = sub.rename(columns={"r": "Rank"})
    if "Std" not in sub.columns and "std" in sub.columns:
        sub = sub.rename(columns={"std": "Std"})
    if "Mean" not in sub.columns and "mean" in sub.columns:
        sub = sub.rename(columns={"mean": "Mean"})

    llms = sorted(sub["LLM"].astype(str).unique().tolist())

    # methods order: by average rank (lower is better)
    methods = (sub.groupby("Method")["Rank"]
                 .mean()
                 .sort_values()
                 .index.astype(str)
                 .tolist())

    # map best-per-LLM (could be multiple methods with Rank=1)
    best = {}
    for llm in llms:
        t = sub[sub["LLM"].astype(str) == llm]
        best[llm] = set(t.loc[t["Rank"] == 1, "Method"].astype(str).tolist())

    # --- build LaTeX ---
    # Column spec: Algorithm column + per-LLM (r + meanstd) pairs
    # Using p widths similar to your example (adjust as needed)
    alg_w = "p{2.4cm}"
    r_w   = "p{0.35cm}"
    ms_w  = "p{2.15cm}"

    colspec = alg_w + "|" + "|".join([f"{r_w}{ms_w}" for _ in llms])

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{2pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.05}")
    lines.append(f"\\caption{{Scott--Knott summary for metric {latex_escape(metric)}. "
                 f"Each dataset (LLM) has two subcolumns: rank $r$ and Mean (Std). "
                 f"Cells with $r=1$ are highlighted.}}")
    lines.append(f"\\label{{tab:sk_paired_{metric.lower()}}}")

    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\hline")

    # Header row 1: Algorithm + multicolumn per LLM
    hdr1 = ["\\textbf{Algorithm}"]
    for llm in llms:
        hdr1.append(f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{latex_escape(llm)}}}}}")
    # last column in the row should not end with trailing '|' visually, but LaTeX is fine.
    # We'll remove the final '|' in the *formatting* by manually editing later if desired.
    lines.append(" & ".join(hdr1) + " \\\\ \\cline{2-" + str(1 + 2*len(llms)) + "}")

    # Header row 2: r / Mean(Std) repeated
    hdr2 = [" "]
    for _ in llms:
        hdr2.append("\\textbf{r}")
        hdr2.append("\\textbf{Mean (Std)}")
    lines.append(" & ".join(hdr2) + " \\\\")
    lines.append("\\hline")

    # Body: one row per method
    for method in methods:
        row = [latex_escape(method)]
        for llm in llms:
            t = sub[(sub["Method"].astype(str) == method) & (sub["LLM"].astype(str) == llm)]
            if t.empty:
                row += ["N/A", "N/A"]
                continue

            rank = t["Rank"].iloc[0]
            mean = t["Mean"].iloc[0]
            std  = t["Std"].iloc[0] if "Std" in t.columns else np.nan

            is_best = (method in best.get(llm, set())) and (not pd.isna(rank)) and int(rank) == 1

            r_txt = highlight_if_best(cell_r(rank), is_best)
            ms_txt = highlight_if_best(fmt_mean_std(float(mean), float(std) if pd.notna(std) else np.nan, digits=3), is_best)

            row += [r_txt, ms_txt]

        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}% end resizebox")
    lines.append("\\end{table}")

    return "\n".join(lines)

def main():
    df = pd.read_csv(IN_CSV)
    # sanity
    required = {"LLM", "Metric", "Method"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}. Found: {list(df.columns)}")

    for metric in METRICS:
        tex = build_paired_table(df, metric)
        if not tex.strip():
            print(f"[WARN] No rows for metric={metric}")
            continue
        out_path = OUT_DIR / f"sk_paired_{metric}.tex"
        out_path.write_text(tex, encoding="utf-8")
        print(f"[OK] wrote {out_path}")

if __name__ == "__main__":
    main()
