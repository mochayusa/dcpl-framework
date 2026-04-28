from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dcpl.metrics import compute_metrics
from dcpl.framework import gated_blocks_and_interactions_fold_predict
from dcpl.interactions import build_all_interactions

from dcpl.blocks import get_blocks_relaxed


ABLATIONS: Dict[str, Dict[str, bool]] = {
    "M1_no_interaction":    {"AIxNonAI": False, "AIxWorkload": False, "NonAIxWorkload": False},
    "M2a_add_AIxNonAI":     {"AIxNonAI": True,  "AIxWorkload": False, "NonAIxWorkload": False},
    "M2b_add_AIxN_and_AIxW":{"AIxNonAI": True,  "AIxWorkload": True,  "NonAIxWorkload": False},
    "M3_full":              {"AIxNonAI": True,  "AIxWorkload": True,  "NonAIxWorkload": True},
}


def _make_unique_dir(path: Path) -> Path:
    """Create a unique directory; if exists append _1, _2, ..."""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
        return path
    k = 1
    while True:
        cand = Path(f"{path}_{k}")
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        k += 1


def _filter_interactions(inter: Dict[str, pd.DataFrame], cfg: Dict[str, bool]) -> Dict[str, pd.DataFrame]:
    """Return inter dict where disabled blocks become empty (0 columns) but keep row alignment."""
    out = {}
    for name, block in inter.items():
        if cfg.get(name, False):
            out[name] = block
        else:
            # empty block with same index
            out[name] = block.iloc[:, :0]
    return out


def _run_one_ablation_split80(
    df: pd.DataFrame,
    target: str,
    model_tag: str,
    test_size: float,
    random_state: int,
    inner_splits: int,
    gate_kind: str,
    base_kind: str,
    inter_kind: str,
) -> pd.DataFrame:
    """
    Runs M1/M2a/M2b/M3 on ONE df (single model CSV) using a single 80/20 split.
    Returns a dataframe with metrics and deltas relative to M1.
    """

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in df columns.")

    # Build blocks (canonical numeric blocks)
    X_ai, X_nonai, X_wl = get_blocks_relaxed(df)

    # Build interactions FROM BLOCKS (not raw df) to avoid alias KeyErrors
    df_blocks = pd.concat([X_ai, X_nonai, X_wl], axis=1)
    inter_all = build_all_interactions(df_blocks, X_ai.columns.tolist(), X_nonai.columns.tolist(), X_wl.columns.tolist())

    # Train/test split
    idx_train, idx_test = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # Slice blocks
    Xa_tr, Xa_te = X_ai.loc[idx_train], X_ai.loc[idx_test]
    Xn_tr, Xn_te = X_nonai.loc[idx_train], X_nonai.loc[idx_test]
    Xw_tr, Xw_te = X_wl.loc[idx_train], X_wl.loc[idx_test]

    y_tr = df.loc[idx_train, target]
    y_te = df.loc[idx_test, target]

    # Slice interactions
    inter_tr_all = {k: v.loc[idx_train] for k, v in inter_all.items()}
    inter_te_all = {k: v.loc[idx_test] for k, v in inter_all.items()}

    rows = []
    m1_metrics = None

    for exp_name, cfg in ABLATIONS.items():
        inter_tr = _filter_interactions(inter_tr_all, cfg)
        inter_te = _filter_interactions(inter_te_all, cfg)

        # Predict (gated + blocks + interactions)
        y_pred = gated_blocks_and_interactions_fold_predict(
            Xa_tr, Xa_te,
            Xn_tr, Xn_te,
            Xw_tr, Xw_te,
            inter_tr, inter_te,
            y_tr,
            base_kind=base_kind,
            inter_kind=inter_kind,
            inner_splits=inner_splits,
            gate_kind=gate_kind
        )

        met = compute_metrics(np.asarray(y_te, dtype=float), np.asarray(y_pred, dtype=float))

        row = {
            "experiment": exp_name,
            "learner": f"dcpl_gate={gate_kind}",
            "cv": "split80_20",
            "target": target,
            "model_tag": model_tag,
            "n_train": int(len(idx_train)),
            "n_test": int(len(idx_test)),
            **met,
        }
        rows.append(row)

        if exp_name == "M1_no_interaction":
            m1_metrics = row

    if m1_metrics is None:
        raise RuntimeError("M1_no_interaction did not run; cannot compute deltas.")

    # Add deltas relative to M1
    out = []
        # Add deltas relative to M1
    out = []
    for r in rows:
        out.append({
            **r,
            "ΔR2":   r["R2"]   - m1_metrics["R2"],
            "ΔMAE":  r["MAE"]  - m1_metrics["MAE"],
            "ΔRMSE": r["RMSE"] - m1_metrics["RMSE"],
            "ΔMRE":  r["MRE"]  - m1_metrics["MRE"],
        })

    return pd.DataFrame(out)


def run_ablation_split80_permodel(
    per_model_dir: str | Path,
    targets: Iterable[str],
    out_root: str | Path = "results/runs",
    run_name: str = "ablation_split80_permodel",
    test_size: float = 0.20,
    random_state: int = 42,
    inner_splits: int = 5,
    gate_kind: str = "ridge",
    base_kind: str = "ridge",
    inter_kind: str = "ridge",
) -> Path:
    """
    Run ablations for EACH per-model CSV in per_model_dir, for EACH target in targets.
    Creates one parent run folder and one subfolder per model_tag.
    Saves:
      - per-model CSV: ablation_metrics.csv
      - parent global summary: ablation_metrics_all.csv
    """
    per_model_dir = Path(per_model_dir)
    files = sorted(per_model_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {per_model_dir}")

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    parent = _make_unique_dir(Path(out_root) / run_id)
    (parent / "per_model").mkdir(parents=True, exist_ok=True)

    all_frames: List[pd.DataFrame] = []

    for csv_path in files:
        model_tag = csv_path.stem
        print(f"\n[RUN] {model_tag}  ({csv_path})")

        df = pd.read_csv(csv_path)

        # For safety: drop rows where target is missing will happen inside per-target run if needed
        model_dir = _make_unique_dir(parent / "per_model" / model_tag)

        model_frames = []
        for tgt in targets:
            if tgt not in df.columns:
                print(f"  [SKIP target] {tgt} not in columns")
                continue

            try:
                df_out = _run_one_ablation_split80(
                    df=df,
                    target=tgt,
                    model_tag=model_tag,
                    test_size=test_size,
                    random_state=random_state,
                    inner_splits=inner_splits,
                    gate_kind=gate_kind,
                    base_kind=base_kind,
                    inter_kind=inter_kind,
                )
                model_frames.append(df_out)
                print(f"  [OK] target={tgt} -> {len(df_out)} rows (M1/M2a/M2b/M3)")
            except Exception as e:
                print(f"  [FAIL] target={tgt} -> {type(e).__name__}: {e}")
                continue

        if model_frames:
            model_all = pd.concat(model_frames, ignore_index=True)
            out_csv = model_dir / "ablation_metrics.csv"
            model_all.to_csv(out_csv, index=False)
            all_frames.append(model_all)
            print(f"  [SAVED] {out_csv}")

    if all_frames:
        global_all = pd.concat(all_frames, ignore_index=True)
        global_csv = parent / "ablation_metrics_all.csv"
        global_all.to_csv(global_csv, index=False)
        print(f"\n[DONE] Global saved: {global_csv}")
    else:
        print("\n[DONE] No results were produced (all targets skipped/failed).")

    return parent

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=1000)

    args = parser.parse_args()

    for i in range(args.runs):
        seed = args.base_seed + i * args.seed_stride
        print(f"\n=== RUN {i+1}/{args.runs} | seed={seed} ===")

        run_ablation_split80_permodel(
            per_model_dir=args.data_dir,
            targets=args.targets,
            random_state=seed,
            test_size=args.test_size,
        )