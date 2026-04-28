from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from experiments.run_baseline_split80_permodel_nested_multirun import (
    run_baseline_split80_permodel_nested_multirun_all_models,
)
from experiments.run_dcpl_split80_permodel_multirun import run_dcpl_split80_permodel_nx
from datetime import datetime

DEFAULT_PER_MODEL_DIR = "data/llm_pilot_data/raw_data/per_model"
DEFAULT_TARGET = "Target_throughput_tokens_per_sec"
DEFAULT_TEST_SIZE = 0.20
DEFAULT_BASE_SEED = 42
DEFAULT_SEED_STRIDE = 1000


BASELINE_ALL = ["lr", "ridge", "rf_light", "nn", "llm_pilot"]

def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="project_main",
        description="Run baseline or DCPL experiments with 80/20 split and multi-run seeds.",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # ---- baseline ----
    p_base = sub.add_parser("baseline", help="Run baseline experiments (multi-run).")
    p_base.add_argument(
        "which",
        type=str,
        help="Which baseline to run: 'all' or one of: lr, ridge, rf_light, nn, llm_pilot",
    )
    p_base.add_argument("n_runs", type=int, help="Number of runs (e.g., 30).")

    # ---- dcpl ----
    p_dcpl = sub.add_parser("dcpl", help="Run DCPL experiments (multi-run).")
    p_dcpl.add_argument("n_runs", type=int, help="Number of runs (e.g., 30).")

    # ---- shared optional args ----
    for sp in [p_base, p_dcpl]:
        sp.add_argument("--per-model-dir", type=str, default=DEFAULT_PER_MODEL_DIR)
        sp.add_argument("--target", type=str, default=DEFAULT_TARGET)
        sp.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
        sp.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
        sp.add_argument("--seed-stride", type=int, default=DEFAULT_SEED_STRIDE)
        sp.add_argument("--results-root", type=str, default="results/runs")
        sp.add_argument("--run-id", type=str, default=None, help="Optional run id folder name. If not set, uses timestamp.")


    # ---- dcpl specific ----
    p_dcpl.add_argument("--gate-kind", type=str, default="ridge", choices=["lr", "ridge", "nn", "rf"])
    p_dcpl.add_argument("--inner-splits", type=int, default=5)

    return p.parse_args()


def run_baseline_cli(args: argparse.Namespace) -> None:
    which = args.which.lower().strip()
    n_runs = int(args.n_runs)

    if which == "all":
        baselines: List[str] = BASELINE_ALL
    else:
        if which not in BASELINE_ALL:
            raise SystemExit(
                f"Unknown baseline '{which}'. Use 'all' or one of: {', '.join(BASELINE_ALL)}"
            )
        baselines = [which]

    out = run_baseline_split80_permodel_nested_multirun_all_models(
        baselines=baselines,
        n_runs=n_runs,
        base_seed=int(args.base_seed),
        seed_stride=int(args.seed_stride),
        per_model_dir=args.per_model_dir,
        target=args.target,
        test_size=float(args.test_size),
        results_root=Path(args.results_root) / f"{n_runs}x_baselines",
        run_name="baseline_split80",
    )

    print("\n=== BASELINE DONE ===")
    print(f"out_dir   : {out.out_dir}")
    print(f"stacked   : {out.stacked_csv}")
    print(f"mean_csv  : {out.mean_csv}")
    print(f"std_csv   : {out.std_csv}")


def run_dcpl_cli(args: argparse.Namespace) -> None:
    n_runs = int(args.n_runs)

    out = run_dcpl_split80_permodel_nx(
        n_runs=n_runs,
        base_seed=int(args.base_seed),
        seed_stride=int(args.seed_stride),
        per_model_dir=args.per_model_dir,
        target=args.target,
        gate_kind=str(args.gate_kind),
        inner_splits=int(args.inner_splits),
        test_size=float(args.test_size),
        results_root=Path(args.results_root) / f"{n_runs}x_dcpl",
        run_name="dcpl_split80",
    )

    print("\n=== DCPL DONE ===")
    print(f"out_dir   : {out.out_dir}")
    print(f"stacked   : {out.stacked_csv}")
    print(f"mean_csv  : {out.mean_csv}")
    print(f"std_csv   : {out.std_csv}")


def main() -> None:
    args = _parse_args()

    per_model_dir = Path(args.per_model_dir)
    if not per_model_dir.exists():
        raise SystemExit(f"--per-model-dir does not exist: {per_model_dir}")

    run_id = args.run_id or make_run_id()
    args.results_root = str(Path(args.results_root) / run_id)

    if args.command == "baseline":
        run_baseline_cli(args)
    elif args.command == "dcpl":
        run_dcpl_cli(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
