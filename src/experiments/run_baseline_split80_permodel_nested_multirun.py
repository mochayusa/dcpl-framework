from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from experiments.run_baseline_split80_permodel_nested import run_baseline_split80_permodel_nested


@dataclass(frozen=True)
class MultiLearnerMultiRunResult:
    out_dir: Path
    stacked_csv: Path
    mean_csv: Path
    std_csv: Path


def _find_global_summary_csv(parent_dir: Path, model_kind: str) -> Path:
    """
    Prefer the parent-level global summary saved by run_baseline_split80_permodel_nested():
      parent_dir / f"permodel_split80_baseline_summary_{model_kind}.csv"
    Fallback: newest baseline_summary csv.
    """
    exact = parent_dir / f"permodel_split80_baseline_summary_{model_kind}.csv"
    if exact.exists():
        return exact

    candidates = sorted(parent_dir.glob("**/*baseline_summary*.csv"))
    if not candidates:
        candidates = sorted(parent_dir.glob("**/*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found under: {parent_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_baseline_split80_permodel_nested_multirun_all_models(
    *,
    baselines: Iterable[str] = ("lr", "ridge", "rf_light", "nn", "llm_pilot"),
    n_runs: int = 30,
    base_seed: int = 42,
    seed_stride: int = 100,
    per_model_dir: str | Path = "data/llm_pilot_data/raw_data/per_model",
    target: str = "Target_throughput_tokens_per_sec",
    test_size: float = 0.20,
    results_root: str | Path = "results/runs",
    run_name: str = "baseline_split80",
) -> MultiLearnerMultiRunResult:
    results_root = Path(results_root)
    out_dir = results_root / f"{run_name}__MULTI_BASELINES__{n_runs}x_base{base_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stacked_frames = []

    for mk in baselines:
        print(f"\n=== Learner: {mk} | {n_runs} runs ===")

        for i in range(n_runs):
            seed_i = int(base_seed + i * seed_stride)
            run_name_i = f"{run_name}_{mk}_iter{i+1:02d}_seed{seed_i}"

            parent_dir = Path(
                run_baseline_split80_permodel_nested(
                    per_model_dir=per_model_dir,
                    target=target,
                    test_size=test_size,
                    random_state=seed_i,
                    model_kind=mk,
                    results_root=results_root,
                    run_name=run_name_i,
                )
            )

            summary_csv = _find_global_summary_csv(parent_dir, model_kind=mk)
            df = pd.read_csv(summary_csv)

            df["iteration"] = i + 1
            df["seed"] = seed_i
            df["learner"] = mk
            df["run_name_iter"] = run_name_i
            df["parent_dir"] = str(parent_dir)
            df["summary_csv"] = str(summary_csv)

            stacked_frames.append(df)

            print(f"[OK] {mk} iter {i+1}/{n_runs} seed={seed_i} -> {summary_csv}")

    if not stacked_frames:
        raise RuntimeError("No summaries collected. Check paths / whether runs are being skipped.")

    stacked = pd.concat(stacked_frames, ignore_index=True)

    # aggregation keys
    group_cols = [c for c in ["per_model_file", "target", "learner", "cv", "experiment"] if c in stacked.columns]
    if "per_model_file" not in group_cols:
        raise ValueError("Expected 'per_model_file' in global summary for per-model mean/std.")

    # numeric columns only
    numeric_cols = [
        c for c in stacked.columns
        if c not in set(group_cols + ["iteration", "seed", "run_name_iter", "parent_dir", "summary_csv"])
        and pd.api.types.is_numeric_dtype(stacked[c])
    ]

    mean_df = stacked.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    std_df  = stacked.groupby(group_cols, dropna=False)[numeric_cols].std(ddof=1).reset_index()

    stacked_csv = out_dir / "baseline_split80_permodel_ALL_learners_ALL_runs_stacked.csv"
    mean_csv    = out_dir / "baseline_split80_permodel_mean_across_runs.csv"
    std_csv     = out_dir / "baseline_split80_permodel_std_across_runs.csv"

    stacked.to_csv(stacked_csv, index=False)
    mean_df.to_csv(mean_csv, index=False)
    std_df.to_csv(std_csv, index=False)

    print(f"\n[DONE] stacked: {stacked_csv}")
    print(f"[DONE] mean:    {mean_csv}")
    print(f"[DONE] std:     {std_csv}")

    return MultiLearnerMultiRunResult(
        out_dir=out_dir,
        stacked_csv=stacked_csv,
        mean_csv=mean_csv,
        std_csv=std_csv,
    )
