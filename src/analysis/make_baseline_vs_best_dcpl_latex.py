from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================
BASELINE_MEAN_CSV = Path(
    "results/runs/20260130_104106/30x_baselines/"
    "baseline_split80__MULTI_BASELINES__30x_base42/"
    "baseline_split80_permodel_mean_across_runs.csv"
)

DCPL_MEAN_CSV = Path(
    "results/runs/20260130_104106/30x_dcpl/"
    "dcpl_split80__multirun_30x_gates_summary/"
    "dcpl_gate_summaries_mean_across_runs.csv"
)

OUT_DIR = Path("results/latex_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TEX = OUT_DIR / "baseline_vs_best_dcpl_boldbest_R2_MAE_RMSE_MRE.tex"

BASELINE_LEARNER_ORDER = ["lr", "ridge", "nn", "rf_light", "llm_pilot"]
GATE_ORDER = ["ridge", "lr", "nn", "rf"]

METRICS = ["R2", "MAE", "RMSE", "MRE"]
DECIMALS = {"R2": 4, "MAE": 2, "RMSE": 2, "MRE": 2}
METRIC_LABELS = {"R2": r"R$^2$", "MAE": "MAE", "RMSE": "RMSE", "MRE": "MRE"}

# Optional filters
TARGET_FILTER = None   # e.g. "Target_throughput_tokens_per_sec"
CV_FILTER = None       # e.g. "split80_20"

# Floating-point tolerance for tie detection
TOL = 1e-12


# ============================================================
# HELPERS
# ============================================================
def latex_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return (
        s.replace("&", r"\&")
         .replace("%", r"\%")
         .replace("#", r"\#")
         .replace("_", r"\_")
    )


def fmt(x: float, dec: int) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{dec}f}"


def bold(s: str) -> str:
    return s if s == "" else r"\textbf{" + s + "}"


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}\nAvailable: {list(df.columns)}")


def maybe_filter(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()

    if TARGET_FILTER is not None and "target" in out.columns:
        out = out[out["target"] == TARGET_FILTER].copy()

    if CV_FILTER is not None and "cv" in out.columns:
        out = out[out["cv"] == CV_FILTER].copy()

    if out.empty:
        raise ValueError(f"[{name}] Data became empty after filtering.")
    return out


def best_dcpl_per_metric(dcpl: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Return one row per per_model_file with the best DCPL gate for a given metric.
    R2: max is best
    MAE/RMSE/MRE: min is best
    """
    if metric == "R2":
        idx = dcpl.groupby("per_model_file")[metric].idxmax()
    else:
        idx = dcpl.groupby("per_model_file")[metric].idxmin()

    idx = idx.dropna().astype(int)

    out = dcpl.loc[idx, ["per_model_file", "gate_kind", metric]].copy()
    out = out.rename(
        columns={
            "gate_kind": f"best_gate_{metric}",
            metric: f"best_dcpl_{metric}",
        }
    )
    return out


def is_higher_better(metric: str) -> bool:
    return metric == "R2"


def is_tied_best(value: float, best_value: float) -> bool:
    if pd.isna(value) or pd.isna(best_value):
        return False
    return abs(float(value) - float(best_value)) <= TOL


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    if not BASELINE_MEAN_CSV.exists():
        raise FileNotFoundError(f"Baseline mean CSV not found:\n  {BASELINE_MEAN_CSV}\nCWD: {Path.cwd()}")
    if not DCPL_MEAN_CSV.exists():
        raise FileNotFoundError(f"DCPL mean CSV not found:\n  {DCPL_MEAN_CSV}\nCWD: {Path.cwd()}")

    # --------------------
    # Load baseline means
    # --------------------
    base = pd.read_csv(BASELINE_MEAN_CSV)
    require_cols(base, ["per_model_file", "learner"] + METRICS, "baseline_mean")
    base = maybe_filter(base, "baseline_mean")
    base["learner"] = base["learner"].astype(str).str.lower().str.strip()

    available_base_learners = sorted(base["learner"].unique().tolist())
    baseline_learners = [l for l in BASELINE_LEARNER_ORDER if l in available_base_learners]
    baseline_learners += [l for l in available_base_learners if l not in baseline_learners]

    # --------------------
    # Load DCPL means
    # --------------------
    dcpl = pd.read_csv(DCPL_MEAN_CSV)
    require_cols(dcpl, ["per_model_file", "gate_kind"] + METRICS, "dcpl_mean")
    dcpl = maybe_filter(dcpl, "dcpl_mean")
    dcpl["gate_kind"] = dcpl["gate_kind"].astype(str).str.lower().str.strip()

    # --------------------
    # Best DCPL per metric
    # --------------------
    best = None
    for metric in METRICS:
        bm = best_dcpl_per_metric(dcpl, metric)
        best = bm if best is None else best.merge(bm, on="per_model_file", how="outer")

    best_map = best.set_index("per_model_file")

    # --------------------
    # Pivot baseline into wide format per metric
    # --------------------
    piv_base = {}
    for metric in METRICS:
        piv = base.pivot_table(
            index="per_model_file",
            columns="learner",
            values=metric,
            aggfunc="mean",
        )
        piv_base[metric] = piv

    datasets = set(best["per_model_file"].tolist())
    for metric in METRICS:
        datasets |= set(piv_base[metric].index.tolist())
    datasets = sorted(datasets)

    # --------------------
    # Build LaTeX table
    # --------------------
    col_spec = "ll" + ("c" * (len(baseline_learners) + 1))
    header_learners = " & ".join(latex_escape(l) for l in baseline_learners)
    header = r"Dataset & Metric & " + header_learners + r" & Best DCPL \\"

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Baseline vs best DCPL comparison using mean over runs. "
        r"Best values per dataset and metric are shown in bold "
        r"($R^2$: higher is better; MAE/RMSE/MRE: lower is better).}"
    )
    lines.append(r"\label{tab:baseline-vs-best-dcpl}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    def get_base_value(ds: str, learner: str, metric: str) -> float:
        piv = piv_base.get(metric)
        if piv is None:
            return np.nan
        if ds in piv.index and learner in piv.columns:
            return piv.loc[ds, learner]
        return np.nan

    for ds in datasets:
        ds_tex = latex_escape(ds)

        for i, metric in enumerate(METRICS):
            dec = DECIMALS.get(metric, 3)

            base_vals = {l: get_base_value(ds, l, metric) for l in baseline_learners}

            best_val = np.nan
            best_gate = ""
            if ds in best_map.index:
                vcol = f"best_dcpl_{metric}"
                gcol = f"best_gate_{metric}"
                if vcol in best_map.columns:
                    best_val = best_map.loc[ds, vcol]
                if gcol in best_map.columns:
                    best_gate = best_map.loc[ds, gcol]

            candidates = [
                float(v) for v in base_vals.values() if not pd.isna(v)
            ]
            if not pd.isna(best_val):
                candidates.append(float(best_val))

            overall_best = np.nan
            if candidates:
                overall_best = max(candidates) if is_higher_better(metric) else min(candidates)

            row_cells = []
            for learner in baseline_learners:
                val = base_vals[learner]
                s = fmt(val, dec)
                if s != "" and is_tied_best(val, overall_best):
                    s = bold(s)
                row_cells.append(s)

            dcpl_str = fmt(best_val, dec)
            if dcpl_str != "" and best_gate:
                dcpl_str = f"{dcpl_str} ({latex_escape(best_gate)})"
            if dcpl_str != "" and is_tied_best(best_val, overall_best):
                dcpl_str = bold(dcpl_str)

            ds_cell = ds_tex if i == 0 else ""
            metric_label = METRIC_LABELS.get(metric, metric)

            lines.append(
                ds_cell + f" & {metric_label} & " + " & ".join(row_cells) + " & " + dcpl_str + r" \\"
            )

        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] LaTeX saved: {OUT_TEX}")
    print(f"[INFO] baseline learners used: {baseline_learners}")
    print(f"[INFO] datasets: {len(datasets)}")


if __name__ == "__main__":
    main()