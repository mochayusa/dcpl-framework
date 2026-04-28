# src/analysis/make_summary_result_gate_dcpl.py

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict

import pandas as pd


# ============================================================
# CONFIG (edit if needed)
# ============================================================
INPUT_ROOT = Path("results/runs/20260130_104106/30x_dcpl")

# Output dir (you can change to other location)
OUT_DIR = Path("results/runs/20260130_104106/30x_dcpl/dcpl_split80__multirun_30x_gates_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_STACKED = OUT_DIR / "dcpl_gate_summaries_ALL_runs_stacked.csv"
OUT_MEAN = OUT_DIR / "dcpl_gate_summaries_mean_across_runs.csv"
OUT_STD = OUT_DIR / "dcpl_gate_summaries_std_across_runs.csv"

# Which metrics to aggregate (only those present will be used)
METRIC_CANDIDATES = ["R2", "MAE", "RMSE", "MRE"]

# Pattern: permodel_split80_dcpl_summary_gate-ridge.csv, gate-nn.csv, gate-rf.csv, gate-lr.csv
GATE_RE = re.compile(r"permodel_split80_dcpl_summary_gate-(?P<gate>[a-zA-Z0-9_]+)\.csv$")


# ============================================================
# Helpers
# ============================================================
def _find_summary_csvs(root: Path) -> List[Path]:
    """Find all parent-level summary CSV files across timestamp dirs."""
    return sorted(root.glob("**/permodel_split80_dcpl_summary_gate-*.csv"))


def _infer_gate_from_filename(p: Path) -> str:
    m = GATE_RE.search(p.name)
    if not m:
        return "unknown"
    return m.group("gate").lower().strip()


def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}\nAvailable: {list(df.columns)}")


def _safe_numeric_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Return numeric columns excluding known identifiers."""
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


# ============================================================
# Main
# ============================================================
def main() -> None:
    if not INPUT_ROOT.exists():
        raise SystemExit(f"[ERROR] INPUT_ROOT does not exist: {INPUT_ROOT}")

    summary_paths = _find_summary_csvs(INPUT_ROOT)
    if not summary_paths:
        raise SystemExit(f"[ERROR] No summary CSV found under: {INPUT_ROOT}")

    print(f"[FOUND] {len(summary_paths)} summary CSV files")

    frames: List[pd.DataFrame] = []

    for i, csv_path in enumerate(summary_paths, start=1):
        gate = _infer_gate_from_filename(csv_path)

        df = pd.read_csv(csv_path)

        # Minimal expected cols in your summary row
        must = ["per_model_file"]
        _require_cols(df, must, name=csv_path.name)

        # Ensure these exist (make if missing)
        if "gate_kind" not in df.columns:
            df["gate_kind"] = gate
        else:
            df["gate_kind"] = df["gate_kind"].astype(str).str.lower().str.strip()

        # Many of your summaries already have learner="dcpl_gate=..."
        if "learner" not in df.columns:
            df["learner"] = f"dcpl_gate={gate}"

        if "target" not in df.columns:
            df["target"] = ""
        if "cv" not in df.columns:
            df["cv"] = "split80_20"
        if "experiment" not in df.columns:
            df["experiment"] = "dcpl_split80"

        # Track provenance
        # parent run folder name is the timestamp folder above the csv
        df["run_folder"] = csv_path.parent.name
        df["summary_path"] = str(csv_path)

        frames.append(df)
        if i % 25 == 0 or i == len(summary_paths):
            print(f"  loaded {i}/{len(summary_paths)}")

    stacked = pd.concat(frames, ignore_index=True)

    # Normalize gate_kind (for safety)
    stacked["gate_kind"] = stacked["gate_kind"].astype(str).str.lower().str.strip()

    # Determine which metrics exist
    metrics_present = [m for m in METRIC_CANDIDATES if m in stacked.columns]
    if not metrics_present:
        # fallback: any numeric columns besides IDs
        id_like = ["per_model_file", "target", "cv", "experiment", "learner", "gate_kind", "run_folder", "summary_path"]
        metrics_present = _safe_numeric_cols(stacked, exclude=id_like)

    print(f"[METRICS] Using metrics: {metrics_present}")

    # Save stacked
    stacked.to_csv(OUT_STACKED, index=False)
    print(f"[OK] stacked saved: {OUT_STACKED}")

    # Group keys
    group_cols = [c for c in ["per_model_file", "target", "cv", "gate_kind"] if c in stacked.columns]

    # Mean/std across runs
    mean_df = stacked.groupby(group_cols, dropna=False)[metrics_present].mean().reset_index()
    std_df = stacked.groupby(group_cols, dropna=False)[metrics_present].std(ddof=1).reset_index()

    mean_df.to_csv(OUT_MEAN, index=False)
    std_df.to_csv(OUT_STD, index=False)

    print(f"[OK] mean saved: {OUT_MEAN}")
    print(f"[OK] std  saved: {OUT_STD}")

    # Optional: quick sanity counts
    counts = (
        stacked.groupby(["gate_kind", "per_model_file"], dropna=False)
        .size()
        .reset_index(name="n_runs_found")
        .sort_values(["gate_kind", "per_model_file"])
    )
    out_counts = OUT_DIR / "dcpl_gate_summaries_run_counts.csv"
    counts.to_csv(out_counts, index=False)
    print(f"[OK] run counts saved: {out_counts}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
