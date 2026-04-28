#!/usr/bin/env python3
"""
Export a LaTeX TABLE ONLY (no full document), safe from LaTeX underscore errors.

Why this version won't error:
- Keeps escape=False so \\textbf{} and $\\pm$ remain.
- Wraps identifier columns (per_model_file/target/cv) with \\detokenize{...}
  so underscores and special chars won't break LaTeX.
- Escapes underscores in column headers (rf_light -> rf\\_light, etc.).
- Uses booktabs rules in the output (requires \\usepackage{booktabs} in your main tex).

Default inputs are YOUR run folders, but you can override via CLI.

Run:
  python src/analysis_baseline_vs_dcpl_table_only.py --metric MRE
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# DEFAULT INPUT PATHS (YOUR RUNS)
# =========================
BASELINE_DIR = Path(
    "results/runs/20260128_082642/30x_baselines/"
    "baseline_split80__MULTI_BASELINES__30x_base42"
)
DCPL_DIR = Path(
    "results/runs/20260128_024807/30x_dcpl/"
    "dcpl_split80__multirun_30x_base42"
)

DEFAULT_BASELINE_MEAN = BASELINE_DIR / "baseline_split80_permodel_mean_across_runs.csv"
DEFAULT_BASELINE_STD  = BASELINE_DIR / "baseline_split80_permodel_std_across_runs.csv"
DEFAULT_DCPL_MEAN     = DCPL_DIR / "dcpl_split80_permodel_mean_across_runs.csv"
DEFAULT_DCPL_STD      = DCPL_DIR / "dcpl_split80_permodel_std_across_runs.csv"

DEFAULT_OUT_DIR = Path("results/latex_tables")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _infer_join_keys(df: pd.DataFrame) -> list[str]:
    keys = [k for k in ["per_model_file", "target", "cv"] if k in df.columns]
    if "per_model_file" not in keys:
        raise ValueError("Expected 'per_model_file' column in summary.")
    return keys


def _format_mean_std(mean: float, std: float, digits: int) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


def _parse_mean(cell: str) -> float:
    if not isinstance(cell, str) or not cell.strip():
        return np.nan
    left = cell.split("$\\pm$")[0] if "$\\pm$" in cell else cell
    try:
        return float(left)
    except Exception:
        return np.nan


def _is_lower_better(metric: str) -> bool:
    m = metric.strip().lower()
    return any(x in m for x in ["mae", "rmse", "mre", "mape", "error"])


def _wrap_detokenize(x) -> str:
    if pd.isna(x):
        return ""
    return f"\\detokenize{{{str(x)}}}"


def _escape_header(h: str) -> str:
    return str(h).replace("_", r"\_")


def build_wide_table(
    bm: pd.DataFrame,
    bs: pd.DataFrame,
    dm: pd.DataFrame,
    ds: pd.DataFrame,
    metric: str,
    digits: int,
) -> pd.DataFrame:
    join_keys = _infer_join_keys(bm)
    for k in join_keys:
        if k not in dm.columns:
            raise ValueError(f"DCPL mean is missing join key: {k}")

    # ---- baseline wide ----
    if "learner" not in bm.columns or "learner" not in bs.columns:
        raise ValueError("Baseline mean/std must contain 'learner'.")
    if metric not in bm.columns or metric not in bs.columns:
        raise ValueError(f"Metric '{metric}' not found in baseline mean/std.")

    b = bm[join_keys + ["learner", metric]].merge(
        bs[join_keys + ["learner", metric]],
        on=join_keys + ["learner"],
        how="left",
        suffixes=("_mean", "_std"),
    )
    b["val"] = [_format_mean_std(m, s, digits) for m, s in zip(b[f"{metric}_mean"], b[f"{metric}_std"])]
    b_wide = b.pivot_table(index=join_keys, columns="learner", values="val", aggfunc="first").reset_index()

    # ---- dcpl wide ----
    if "learner" not in dm.columns:
        dm = dm.copy()
        dm["learner"] = "DCPL"
    if "learner" not in ds.columns:
        ds = ds.copy()
        ds["learner"] = dm["learner"].iloc[0] if len(dm) else "DCPL"

    if metric not in dm.columns or metric not in ds.columns:
        raise ValueError(f"Metric '{metric}' not found in DCPL mean/std.")

    d = dm[join_keys + ["learner", metric]].merge(
        ds[join_keys + ["learner", metric]],
        on=join_keys + ["learner"],
        how="left",
        suffixes=("_mean", "_std"),
    )
    d["val"] = [_format_mean_std(m, s, digits) for m, s in zip(d[f"{metric}_mean"], d[f"{metric}_std"])]
    d_wide = d.pivot_table(index=join_keys, columns="learner", values="val", aggfunc="first").reset_index()

    merged = b_wide.merge(d_wide, on=join_keys, how="outer")
    merged = merged.sort_values("per_model_file").reset_index(drop=True)
    return merged


def bold_best(df: pd.DataFrame, lower_is_better: bool) -> pd.DataFrame:
    out = df.copy()
    join_keys = [c for c in ["per_model_file", "target", "cv"] if c in out.columns]
    val_cols = [c for c in out.columns if c not in join_keys]

    for i in range(len(out)):
        means = {c: _parse_mean(out.at[i, c]) for c in val_cols}
        means = {c: v for c, v in means.items() if pd.notna(v)}
        if not means:
            continue
        best_col = min(means, key=means.get) if lower_is_better else max(means, key=means.get)
        out.at[i, best_col] = f"\\textbf{{{out.at[i, best_col]}}}"
    return out


def latex_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["per_model_file", "target", "cv"]:
        if col in out.columns:
            out[col] = out[col].map(_wrap_detokenize)
    out.columns = [_escape_header(c) for c in out.columns]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export LaTeX table only (safe from underscore errors).")
    ap.add_argument("--metric", type=str, default="R2", help="Metric: R2, MAE, RMSE, MRE, MRE% ...")
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--no-bold", action="store_true")

    # optional overrides
    ap.add_argument("--baseline-mean", type=str, default=str(DEFAULT_BASELINE_MEAN))
    ap.add_argument("--baseline-std", type=str, default=str(DEFAULT_BASELINE_STD))
    ap.add_argument("--dcpl-mean", type=str, default=str(DEFAULT_DCPL_MEAN))
    ap.add_argument("--dcpl-std", type=str, default=str(DEFAULT_DCPL_STD))

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bm = _read_csv(Path(args.baseline_mean))
    bs = _read_csv(Path(args.baseline_std))
    dm = _read_csv(Path(args.dcpl_mean))
    ds = _read_csv(Path(args.dcpl_std))

    merged = build_wide_table(bm, bs, dm, ds, metric=args.metric, digits=args.digits)

    lower_is_better = _is_lower_better(args.metric)
    merged2 = merged if args.no_bold else bold_best(merged, lower_is_better=lower_is_better)

    latex_df = latex_safe(merged2)

    caption = f"Baseline vs DCPL comparison (mean$\\pm$std over 30 runs, split 80/20) for {args.metric}."
    label = f"tab:baseline_vs_dcpl_{args.metric.lower().replace('%','pct')}"

    # Column format: left for first 3 columns, then centered for the rest
    join_keys = [c for c in ["per_model_file", "target", "cv"] if c in merged.columns]
    col_format = "l" * len(join_keys) + "c" * (latex_df.shape[1] - len(join_keys))

    table_tex = latex_df.to_latex(
        index=False,
        escape=False,       # keep \textbf and $\pm$
        caption=caption,
        label=label,
        column_format=col_format,
    )

    tex_path = out_dir / f"table_baseline_vs_dcpl_{args.metric}.tex"
    tex_path.write_text(table_tex, encoding="utf-8")

    print("[DONE] Wrote LaTeX table:")
    print(f"  {tex_path}")
    print()
    print("Note: Your main LaTeX doc must include:")
    print("  \\usepackage{booktabs}")


if __name__ == "__main__":
    main()
