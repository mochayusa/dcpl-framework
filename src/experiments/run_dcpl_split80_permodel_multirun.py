#  src/experiments/run_dcpl_split80_permodel_multirun.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.run_dcpl_split80_permodel import run_dcpl_split80_permodel


@dataclass(frozen=True)
class MultiRunResult:
    out_dir: Path
    parent_dirs: list[Path]
    summary_paths: list[Path]
    stacked_csv: Path
    mean_csv: Path
    std_csv: Path


def _find_global_summary_csv(parent_dir: Path, gate_kind: str) -> Path:
    """
    Prefer the exact parent-level global summary written by run_dcpl_split80_permodel():

      parent_dir / f"permodel_split80_dcpl_summary_gate-{gate_kind}.csv"

    Fallback: newest dcpl_summary-like CSV under parent_dir.
    """
    exact = parent_dir / f"permodel_split80_dcpl_summary_gate-{gate_kind}.csv"
    if exact.exists():
        return exact

    candidates = sorted(parent_dir.glob("**/*dcpl_summary*.csv"))
    if not candidates:
        # last fallback: any csv (newest)
        candidates = sorted(parent_dir.glob("**/*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found under: {parent_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_dcpl_split80_permodel_nx(
    *,
    n_runs: int = 30,
    base_seed: int = 42,
    seed_stride: int = 100,
    per_model_dir: str | Path,
    target: str,
    gate_kind: str = "ridge",
    inner_splits: int = 5,
    test_size: float = 0.20,
    results_root: str | Path = "results/runs",
    run_name: str = "dcpl_split80",
    schema: str | Path | None = None,
) -> MultiRunResult:
    """
    Runs DCPL n_runs times with different seeds, then produces mean/std summaries.

    Seeds used: base_seed + i * seed_stride  (i=0..n_runs-1)
    """
    results_root = Path(results_root)

    out_dir = results_root / f"{run_name}__multirun_{n_runs}x_base{base_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    parent_dirs: list[Path] = []
    summary_paths: list[Path] = []
    frames: list[pd.DataFrame] = []

    for i in range(n_runs):
        seed_i = int(base_seed + i * seed_stride)
        run_name_i = f"{run_name}_iter{i+1:02d}_seed{seed_i}"

        parent_dir = Path(
            run_dcpl_split80_permodel(
                per_model_dir=per_model_dir,
                target=target,
                gate_kind=gate_kind,
                inner_splits=inner_splits,
                test_size=test_size,
                random_state=seed_i,
                results_root=results_root,
                run_name=run_name_i,
                schema=schema,
            )
        )
        parent_dirs.append(parent_dir)

        summary_csv = _find_global_summary_csv(parent_dir, gate_kind=gate_kind)
        summary_paths.append(summary_csv)

        df = pd.read_csv(summary_csv)

        # traceability
        df["iteration"] = i + 1
        df["seed"] = seed_i
        if "gate_kind" not in df.columns:
            df["gate_kind"] = gate_kind
        if "target" not in df.columns:
            df["target"] = target
        if "cv" not in df.columns:
            df["cv"] = "split80_20"

        df["run_name_iter"] = run_name_i
        df["parent_dir"] = str(parent_dir)
        df["summary_csv"] = str(summary_csv)

        frames.append(df)

        print(f"[ITER {i+1}/{n_runs}] seed={seed_i} -> {summary_csv}")

    if not frames:
        raise RuntimeError("No summaries collected.")

    stacked = pd.concat(frames, ignore_index=True)

    # --- group keys ---
    if "per_model_file" not in stacked.columns:
        raise ValueError("Expected 'per_model_file' in global summary to compute per-model mean/std.")

    group_cols = [c for c in ["per_model_file", "target", "learner", "cv", "gate_kind"] if c in stacked.columns]

    # --- numeric columns ---
    drop_cols = set(group_cols + ["iteration", "seed", "run_name_iter", "parent_dir", "summary_csv"])
    numeric_cols = [
        c for c in stacked.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(stacked[c])
    ]

    mean_df = stacked.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    std_df  = stacked.groupby(group_cols, dropna=False)[numeric_cols].std(ddof=1).reset_index()

    # --- save ---
    stacked_csv = out_dir / "dcpl_split80_permodel_ALL_runs_stacked.csv"
    mean_csv    = out_dir / "dcpl_split80_permodel_mean_across_runs.csv"
    std_csv     = out_dir / "dcpl_split80_permodel_std_across_runs.csv"

    stacked.to_csv(stacked_csv, index=False)
    mean_df.to_csv(mean_csv, index=False)
    std_df.to_csv(std_csv, index=False)

    print(f"[DONE] stacked saved: {stacked_csv}")
    print(f"[DONE] mean saved:    {mean_csv}")
    print(f"[DONE] std  saved:    {std_csv}")

    return MultiRunResult(
        out_dir=out_dir,
        parent_dirs=parent_dirs,
        summary_paths=summary_paths,
        stacked_csv=stacked_csv,
        mean_csv=mean_csv,
        std_csv=std_csv,
    )
