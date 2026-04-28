# src/project_dice_main.py
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import time

from experiments.run_dice_split80_permodel_multirun import run_dice_split80_permodel_nx


DEFAULT_PER_MODEL_DIR = "data/llm_pilot_data/raw_data/per_model"
DEFAULT_TARGET = "Target_throughput_tokens_per_sec"
DEFAULT_TEST_SIZE = 0.20
DEFAULT_BASE_SEED = 42
DEFAULT_SEED_STRIDE = 1000


VALID_LEARNERS = ["lr", "ridge", "rf", "nn", "xgb"]


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="project_dice_main",
        description="Run DICE (Disentangled Interaction-Concatenated Ensemble) with 80/20 split and multi-run seeds.",
    )

    p.add_argument("learner", type=str, choices=VALID_LEARNERS,
                  help="Learner for DICE: lr | ridge | rf | nn | xgb")
    p.add_argument("n_runs", type=int, help="Number of runs, e.g., 30")

    p.add_argument("--per-model-dir", type=str, default=DEFAULT_PER_MODEL_DIR)
    p.add_argument("--target", type=str, default=DEFAULT_TARGET)
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--seed-stride", type=int, default=DEFAULT_SEED_STRIDE)
    p.add_argument("--results-root", type=str, default="results/runs")
    p.add_argument("--run-id", type=str, default=None,
                  help="Optional run id folder name. If not set, uses timestamp.")

    # ablation flags
    p.add_argument("--no-base", action="store_true", help="Disable base features (AI+NonAI+Workload).")
    p.add_argument("--no-interactions", action="store_true", help="Disable interaction features.")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    per_model_dir = Path(args.per_model_dir)
    if not per_model_dir.exists():
        raise SystemExit(f"--per-model-dir does not exist: {per_model_dir}")

    include_base = not bool(args.no_base)
    include_interactions = not bool(args.no_interactions)

    if not include_base and not include_interactions:
        raise SystemExit("Invalid configuration: --no-base and --no-interactions both set (no features left).")

    run_id = args.run_id or make_run_id()
    results_root = Path(args.results_root) / run_id / "dice_runs"

    t0 = time.perf_counter()

    out = run_dice_split80_permodel_nx(
        n_runs=int(args.n_runs),
        base_seed=int(args.base_seed),
        seed_stride=int(args.seed_stride),
        per_model_dir=str(per_model_dir),
        target=str(args.target),
        learner_kind=str(args.learner),
        include_base=include_base,
        include_interactions=include_interactions,
        test_size=float(args.test_size),
        results_root=str(results_root),
        run_name="dice_split80",
    )

    total = time.perf_counter() - t0

    print("\n=== DICE DONE ===")
    print(f"learner         : {args.learner}")
    print(f"n_runs          : {args.n_runs}")
    print(f"include_base    : {include_base}")
    print(f"include_inter   : {include_interactions}")
    print(f"out_dir         : {out.out_dir}")
    print(f"stacked_csv     : {out.stacked_csv}")
    print(f"mean_csv        : {out.mean_csv}")
    print(f"std_csv         : {out.std_csv}")
    print(f"timing_csv      : {out.timing_csv}")
    print(f"timing_json     : {out.timing_json}")
    print(f"total_seconds   : {total:.2f}")


if __name__ == "__main__":
    main()
