from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================
MEAN_CSV = Path(
    "results/runs/20260130_104106/30x_dcpl/"
    "dcpl_split80__multirun_30x_gates_summary/"
    "dcpl_gate_summaries_mean_across_runs.csv"
)

OUT_DIR = Path("results/latex_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TEX = OUT_DIR / "dcpl_gate_comparison_boldbest_R2_MAE_RMSE_MRE.tex"

# which gates to include (if some missing, script will use those available)
GATE_ORDER = ["ridge", "lr", "nn", "rf"]

# metrics
METRICS = ["R2", "MAE", "RMSE", "MRE"]

# decimals
DECIMALS = {"R2": 4, "MAE": 2, "RMSE": 2, "MRE": 2}


# ============================================================
# HELPERS
# ============================================================
def latex_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def fmt(x: float, dec: int) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{dec}f}"


def bold(s: str) -> str:
    if s == "":
        return ""
    return r"\textbf{" + s + "}"


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}\nAvailable: {list(df.columns)}")


def metric_best_is_max(metric: str) -> bool:
    # R2 higher is better; others lower is better
    return metric.upper() == "R2"


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    if not MEAN_CSV.exists():
        raise FileNotFoundError(f"Mean CSV not found:\n  {MEAN_CSV}\nCWD: {Path.cwd()}")

    df = pd.read_csv(MEAN_CSV)

    require_cols(df, ["per_model_file", "gate_kind"], "mean_csv")
    for m in METRICS:
        require_cols(df, [m], "mean_csv")

    # normalize gate_kind
    df["gate_kind"] = df["gate_kind"].astype(str).str.lower().str.strip()

    # If you have multiple targets/cv and want to filter, uncomment & edit:
    # df = df[df["target"] == "Target_throughput_tokens_per_sec"].copy()
    # df = df[df["cv"] == "split80_20"].copy()

    # determine available gates
    available_gates = sorted(df["gate_kind"].unique().tolist())
    gates = [g for g in GATE_ORDER if g in available_gates] + [g for g in available_gates if g not in GATE_ORDER]
    if not gates:
        raise RuntimeError("No gate_kind found in the input file.")

    # pivot per metric: rows=dataset, cols=gate, values=metric
    piv = {}
    for metric in METRICS:
        piv[metric] = df.pivot_table(index="per_model_file", columns="gate_kind", values=metric, aggfunc="mean")

    # union datasets across all pivots
    datasets = set()
    for metric in METRICS:
        datasets |= set(piv[metric].index.tolist())
    datasets = sorted(datasets)

    # ------------------------------------------------------------
    # Prepare row data:
    # rows[ds][metric] = list of formatted values per gate + bolding
    # ------------------------------------------------------------
    row_blocks = []  # list of dict: {"dataset": ds, "metric_rows": {metric: [cells...]}}
    for ds in datasets:
        metric_rows = {}

        for metric in METRICS:
            row_series = piv[metric].loc[ds] if ds in piv[metric].index else pd.Series(dtype=float)

            # gather values per gate
            vals = {g: (row_series[g] if g in row_series.index else np.nan) for g in gates}

            # determine best gate for this dataset & metric
            best_gate = None
            present_gates = [g for g in gates if not pd.isna(vals[g])]
            if present_gates:
                if metric_best_is_max(metric):
                    best_gate = max(present_gates, key=lambda g: vals[g])
                else:
                    best_gate = min(present_gates, key=lambda g: vals[g])

            # format + bold
            dec = DECIMALS.get(metric, 3)
            cells = []
            for g in gates:
                s = fmt(vals[g], dec)
                if best_gate == g and s != "":
                    s = bold(s)
                cells.append(s)

            metric_rows[metric] = cells

        row_blocks.append({"dataset": ds, "metric_rows": metric_rows})

    # ------------------------------------------------------------
    # LaTeX output:
    # 4 rows per dataset: R2, MAE, RMSE, MRE
    # Columns: Dataset | Metric | gate1 | gate2 | ...
    # ------------------------------------------------------------
    col_spec = "ll" + ("c" * len(gates))
    header_gates = " & ".join([latex_escape(g) for g in gates])

    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    latex_lines.append(
        r"\caption{DCPL gate comparison using mean over runs. "
        r"Best per dataset is bold (R2: higher is better; MAE/RMSE/MRE: lower is better).}"
    )
    latex_lines.append(r"\label{tab:dcpl-gate-compare-boldbest}")
    latex_lines.append(r"\begin{tabular}{" + col_spec + r"}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Dataset & Metric & " + header_gates + r" \\")
    latex_lines.append(r"\midrule")

    for block in row_blocks:
        ds = block["dataset"]
        ds_tex = latex_escape(ds)

        for metric in METRICS:
            cells = block["metric_rows"][metric]
            latex_lines.append(ds_tex + f" & {latex_escape(metric)} & " + " & ".join(cells) + r" \\")
        latex_lines.append(r"\addlinespace")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    OUT_TEX.write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"[OK] LaTeX saved: {OUT_TEX}")
    print(f"[INFO] gates used: {gates}")
    print(f"[INFO] datasets: {len(datasets)}")


if __name__ == "__main__":
    main()
