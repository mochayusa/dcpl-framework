# src/experiments/run_dice_split80_permodel_multirun.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import pandas as pd

from experiments.run_dice_split80_permodel import run_dice_split80_permodel


@dataclass(frozen=True)
class MultiRunResult:
    out_dir: Path
    parent_dirs: list[Path]
    summary_paths: list[Path]
    stacked_csv: Path
    mean_csv: Path
    std_csv: Path
    timing_csv: Path
    timing_json: Path


def _find_global_summary_csv(parent_dir: Path, learner_kind: str) -> Path:
    """
    Prefer exact global summary written by run_dice_split80_permodel():
      parent_dir / f"permodel_split80_dice_summary_{learner_kind}.csv"
    Fallback: newest *dice_summary*.csv under parent_dir.
    """
    exact = parent_dir / f"permodel_split80_dice_summary_{learner_kind}.csv"
    if exact.exists():
        return exact

    candidates = sorted(parent_dir.glob("**/*dice_summary*.csv"))
    if not candidates:
        candidates = sorted(parent_dir.glob("**/*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found under: {parent_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_dice_split80_permodel_nx(
    *,
    n_runs: int = 30,
    base_seed: int = 42,
    seed_stride: int = 1000,
    per_model_dir: str | Path,
    target: str,
    learner_kind: str = "rf",  # lr | ridge | rf | nn | xgb
    include_base: bool = True,
    include_interactions: bool = True,
    test_size: float = 0.20,
    results_root: str | Path = "results/runs",
    run_name: str = "dice_split80",
    schema: str | Path | None = None,
) -> MultiRunResult:
    results_root = Path(results_root)

    out_dir = results_root / f"{run_name}__{learner_kind}__multirun_{n_runs}x_base{base_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timing").mkdir(parents=True, exist_ok=True)

    parent_dirs: list[Path] = []
    summary_paths: list[Path] = []
    frames: list[pd.DataFrame] = []
    timing_rows: list[dict] = []

    t0_all = time.perf_counter()

    for i in range(n_runs):
        seed_i = int(base_seed + i * seed_stride)
        run_name_i = f"{run_name}_{learner_kind}_iter{i+1:02d}_seed{seed_i}"

        t0 = time.perf_counter()
        parent_dir = Path(
            run_dice_split80_permodel(
                per_model_dir=per_model_dir,
                target=target,
                test_size=test_size,
                random_state=seed_i,
                learner_kind=learner_kind,
                include_base=include_base,
                include_interactions=include_interactions,
                results_root=results_root,
                run_name=run_name_i,
                schema=schema,
            )
        )
        dt = time.perf_counter() - t0

        parent_dirs.append(parent_dir)
        summary_csv = _find_global_summary_csv(parent_dir, learner_kind=learner_kind)
        summary_paths.append(summary_csv)

        df = pd.read_csv(summary_csv)

        # traceability
        df["iteration"] = i + 1
        df["seed"] = seed_i
        df["run_name_iter"] = run_name_i
        df["parent_dir"] = str(parent_dir)
        df["summary_csv"] = str(summary_csv)
        df["dice_include_base"] = int(include_base)
        df["dice_include_interactions"] = int(include_interactions)

        # include target/cv if not present
        if "target" not in df.columns:
            df["target"] = target
        if "cv" not in df.columns:
            df["cv"] = "split80_20"
        if "learner" not in df.columns:
            df["learner"] = f"dice_{learner_kind}"

        frames.append(df)

        timing_rows.append({
            "iteration": i + 1,
            "seed": seed_i,
            "learner_kind": learner_kind,
            "run_name_iter": run_name_i,
            "parent_dir": str(parent_dir),
            "summary_csv": str(summary_csv),
            "seconds": float(dt),
        })

        print(f"[DICE ITER {i+1}/{n_runs}] seed={seed_i} seconds={dt:.2f} -> {summary_csv}")

    if not frames:
        raise RuntimeError("No summaries collected for DICE.")

    stacked = pd.concat(frames, ignore_index=True)

    # group keys
    if "per_model_file" not in stacked.columns:
        raise ValueError("Expected 'per_model_file' in global summary to compute per-model mean/std.")

    group_cols = [c for c in ["per_model_file", "target", "learner", "cv"] if c in stacked.columns]

    drop_cols = set(group_cols + [
        "iteration", "seed", "run_name_iter", "parent_dir", "summary_csv",
        "dice_include_base", "dice_include_interactions"
    ])
    numeric_cols = [
        c for c in stacked.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(stacked[c])
    ]

    mean_df = stacked.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    std_df  = stacked.groupby(group_cols, dropna=False)[numeric_cols].std(ddof=1).reset_index()

    # save
    stacked_csv = out_dir / "dice_split80_permodel_ALL_runs_stacked.csv"
    mean_csv    = out_dir / "dice_split80_permodel_mean_across_runs.csv"
    std_csv     = out_dir / "dice_split80_permodel_std_across_runs.csv"

    stacked.to_csv(stacked_csv, index=False)
    mean_df.to_csv(mean_csv, index=False)
    std_df.to_csv(std_csv, index=False)

    # timing
    timing_df = pd.DataFrame(timing_rows)
    timing_csv = out_dir / "timing" / "dice_multirun_timing.csv"
    timing_df.to_csv(timing_csv, index=False)

    total_seconds = float(time.perf_counter() - t0_all)
    timing_json = out_dir / "timing" / "dice_multirun_timing_summary.json"
    timing_json.write_text(json.dumps({
        "n_runs": n_runs,
        "learner_kind": learner_kind,
        "target": target,
        "test_size": test_size,
        "base_seed": base_seed,
        "seed_stride": seed_stride,
        "include_base": bool(include_base),
        "include_interactions": bool(include_interactions),
        "total_seconds": total_seconds,
        "avg_seconds_per_run": float(timing_df["seconds"].mean()),
        "min_seconds_per_run": float(timing_df["seconds"].min()),
        "max_seconds_per_run": float(timing_df["seconds"].max()),
        "out_dir": str(out_dir),
    }, indent=2))

    print(f"[DONE] stacked: {stacked_csv}")
    print(f"[DONE] mean:    {mean_csv}")
    print(f"[DONE] std:     {std_csv}")
    print(f"[DONE] timing:  {timing_csv}")
    print(f"[DONE] timing summary: {timing_json}")

    return MultiRunResult(
        out_dir=out_dir,
        parent_dirs=parent_dirs,
        summary_paths=summary_paths,
        stacked_csv=stacked_csv,
        mean_csv=mean_csv,
        std_csv=std_csv,
        timing_csv=timing_csv,
        timing_json=timing_json,
    )
