from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dcpl.framework import gated_blocks_and_interactions_fold_predict
from dcpl.metrics import compute_metrics
from experiments.common import prepare_run


# ---------------------------
# helpers
# ---------------------------

def _ensure_inter_keys(inter: dict) -> None:
    need = {"AIxNonAI", "AIxWorkload", "NonAIxWorkload"}
    missing = need - set(inter.keys())
    if missing:
        raise KeyError(
            f"Interaction dict missing keys: {sorted(missing)}. "
            f"Available keys: {list(inter.keys())}"
        )


# ---------------------------
# core: one run (one seed)
# ---------------------------

def run_one_split80_dcpl_global(
    df: pd.DataFrame,
    *,
    target: str,
    seed: int,
    test_size: float,
    gate_kind: str,
    inner_splits: int,
) -> dict:
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found. Available: {list(df.columns)}")

    # Canonical feature construction (blocks + interactions)
    # We don't need to save anything here; just reuse the pipeline.
    _, _, X_ai, X_nonai, X_wl, inter = prepare_run(
        df=df,
        results_root="results/tmp",
        run_name="tmp_dcpl_global",
        include_interactions=True,
    )

    if not isinstance(inter, dict):
        raise TypeError(
            "prepare_run(..., include_interactions=True) must return a dict for interactions "
            "as the 6th return value."
        )
    _ensure_inter_keys(inter)

    # 80/20 split indices
    idx_train, idx_test = train_test_split(
        df.index,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    # Blocks
    X_ai_tr, X_ai_te = X_ai.loc[idx_train], X_ai.loc[idx_test]
    X_nonai_tr, X_nonai_te = X_nonai.loc[idx_train], X_nonai.loc[idx_test]
    X_wl_tr, X_wl_te = X_wl.loc[idx_train], X_wl.loc[idx_test]

    # Interactions (must align with idx)
    inter_train = {k: v.loc[idx_train] for k, v in inter.items()}
    inter_test  = {k: v.loc[idx_test]  for k, v in inter.items()}

    y_tr = df.loc[idx_train, target]
    y_te = df.loc[idx_test, target].values.astype(float)

    # DCPL prediction (your actual framework)
    y_pred = gated_blocks_and_interactions_fold_predict(
        X_ai_tr, X_ai_te,
        X_nonai_tr, X_nonai_te,
        X_wl_tr, X_wl_te,
        inter_train=inter_train,
        inter_test=inter_test,
        y_train=y_tr,
        inner_splits=inner_splits,
        gate_kind=gate_kind,
    )

    met = compute_metrics(y_te, y_pred)

    return {
        "dataset_scope": "merged_all_models",
        "target": target,
        "learner": f"dcpl_gate={gate_kind}",
        "cv": "split80_20",
        "seed": int(seed),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        **met,
    }


# ---------------------------
# 5x runner + mean/std
# ---------------------------

def run_dcpl_split80_global_5runs(
    data_path: str | Path,
    *,
    target: str,
    gate_kind: str = "ridge",
    inner_splits: int = 5,
    seeds: list[int] = (42, 1042, 2042, 3042, 4042),
    test_size: float = 0.20,
    out_dir: str | Path = "results/dcpl_global",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    rows = []
    for i, s in enumerate(seeds, start=1):
        print(f"\n=== DCPL global split80 | run {i}/{len(seeds)} | seed={s} | gate={gate_kind} ===")
        row = run_one_split80_dcpl_global(
            df,
            target=target,
            seed=s,
            test_size=test_size,
            gate_kind=gate_kind,
            inner_splits=inner_splits,
        )
        rows.append(row)

        print(
            f"R2={row['R2']:.4f}, RMSE={row['RMSE']:.4f}, "
            f"MAE={row['MAE']:.4f}, MRE={row['MRE']:.2f}%"
        )

    df_runs = pd.DataFrame(rows)

    # aggregate mean/std (sample std)
    agg = {
        "dataset_scope": "merged_all_models",
        "target": target,
        "learner": f"dcpl_gate={gate_kind}",
        "cv": "split80_20",
        "n_runs": len(seeds),
    }
    for m in ["R2", "MAE", "RMSE", "MRE"]:
        agg[f"{m}_mean"] = float(df_runs[m].mean())
        agg[f"{m}_std"]  = float(df_runs[m].std(ddof=1))

    df_summary = pd.DataFrame([agg])

    out_runs = out_dir / f"dcpl_global_split80_5runs_gate={gate_kind}.csv"
    out_sum  = out_dir / f"dcpl_global_split80_5runs_gate={gate_kind}_mean_std.csv"

    df_runs.to_csv(out_runs, index=False)
    df_summary.to_csv(out_sum, index=False)

    print(f"\n[OK] Saved per-run:  {out_runs}")
    print(f"[OK] Saved mean/std: {out_sum}")

    return df_runs, df_summary


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Merged CSV path")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--gate", type=str, default="ridge")
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--seeds", type=str, default="42,1042,2042,3042,4042")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--out-dir", type=str, default="results/dcpl_global")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    run_dcpl_split80_global_5runs(
        data_path=args.data,
        target=args.target,
        gate_kind=args.gate,
        inner_splits=args.inner_splits,
        seeds=seeds,
        test_size=args.test_size,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
