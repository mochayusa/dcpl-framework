from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


RUN_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and RUN_DIR_PATTERN.match(path.name) is not None


def collect_ablation_runs(
    runs_root: str | Path = "results/runs",
    out_dir: str | Path = "results/analysis/ablation",
    expected_rows_per_run: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect all ablation_metrics_all.csv files from repeated run folders,
    stack them, and compute summary statistics.

    Returns:
        stacked_df, summary_df
    """
    runs_root = Path(runs_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in runs_root.iterdir() if is_run_dir(p)])

    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {runs_root}")

    all_frames = []
    audit_rows = []

    for run_idx, run_dir in enumerate(run_dirs, start=1):
        csv_path = run_dir / "ablation_metrics_all.csv"

        if not csv_path.exists():
            audit_rows.append({
                "run_id": run_dir.name,
                "status": "missing_csv",
                "n_rows": 0,
                "path": str(csv_path),
            })
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            audit_rows.append({
                "run_id": run_dir.name,
                "status": f"read_error: {type(e).__name__}: {e}",
                "n_rows": 0,
                "path": str(csv_path),
            })
            continue

        df["run_id"] = run_dir.name
        df["run_idx"] = run_idx

        audit_rows.append({
            "run_id": run_dir.name,
            "status": "ok",
            "n_rows": len(df),
            "path": str(csv_path),
        })

        if expected_rows_per_run is not None and len(df) != expected_rows_per_run:
            print(
                f"[WARN] {run_dir.name}: expected {expected_rows_per_run} rows, "
                f"got {len(df)}"
            )

        all_frames.append(df)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(out_dir / "ablation_runs_audit.csv", index=False)

    if not all_frames:
        raise RuntimeError("No valid ablation_metrics_all.csv files were collected.")

    stacked = pd.concat(all_frames, ignore_index=True)

    # Save stacked/raw file
    stacked.to_csv(out_dir / "ablation_metrics_all_runs_stacked.csv", index=False)

    # Metrics to summarise
    metric_cols = ["R2", "MAE", "RMSE", "MRE", "ΔR2", "ΔMAE", "ΔRMSE", "ΔMRE"]

    # Keep only available metrics
    metric_cols = [c for c in metric_cols if c in stacked.columns]

    group_cols = ["model_tag", "target", "experiment", "learner", "cv"]

    summary = (
        stacked
        .groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )

    # Flatten multi-index columns
    summary.columns = [
        f"{a}_{b}" if b else a
        for a, b in summary.columns.to_flat_index()
    ]

    # rename grouped columns back nicely
    rename_map = {
        "model_tag_": "model_tag",
        "target_": "target",
        "experiment_": "experiment",
        "learner_": "learner",
        "cv_": "cv",
    }
    summary = summary.rename(columns=rename_map)

    summary.to_csv(out_dir / "ablation_metrics_summary_mean_std.csv", index=False)

    # Also create a compact mean±std table-like file
    compact = summary.copy()
    for m in ["R2", "MAE", "RMSE", "MRE", "ΔR2", "ΔMAE", "ΔRMSE", "ΔMRE"]:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        if mean_col in compact.columns and std_col in compact.columns:
            compact[f"{m}_mean_std"] = (
                compact[mean_col].round(6).astype(str)
                + " ± "
                + compact[std_col].round(6).astype(str)
            )

    compact.to_csv(out_dir / "ablation_metrics_summary_compact.csv", index=False)

    return stacked, summary


if __name__ == "__main__":
    stacked, summary = collect_ablation_runs(
        runs_root="results/runs",
        out_dir="results/analysis/ablation",
        expected_rows_per_run=40,  # 10 models × 1 target × 4 ablations
    )

    print("\n[Done]")
    print(f"Stacked rows: {len(stacked)}")
    print(f"Summary rows: {len(summary)}")
    print("Saved to: results/analysis/ablation")