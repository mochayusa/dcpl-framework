from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from dcpl.framework import baseline_fold_predict
from dcpl.metrics import compute_metrics
from dcpl.blocks import get_blocks_relaxed

from utils.io import save_predictions, save_summary, save_manifest


def _make_unique_dir(path: Path) -> Path:
    """Create a unique directory. If exists, append _1, _2, ..."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
        return path
    k = 1
    while True:
        candidate = Path(f"{path}_{k}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        k += 1


def run_baseline_split80_permodel_nested(
    per_model_dir: str | Path = "data/llm_pilot_data/raw_data/per_model",
    target: str = "Target_throughput_tokens_per_sec",
    test_size: float = 0.20,
    random_state: int = 42,
    model_kind: str = "rf_light",
    results_root: str | Path = "results/runs",
    run_name: str = "baseline_split80",
):
    per_model_dir = Path(per_model_dir)
    files = sorted(per_model_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {per_model_dir}")

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    parent_dir = _make_unique_dir(Path(results_root) / run_id)
    (parent_dir / "per_model").mkdir(parents=True, exist_ok=True)

    all_rows = []

    for csv_path in files:
        model_tag = csv_path.stem

        df = pd.read_csv(csv_path)
        if target not in df.columns:
            print(f"[SKIP] {model_tag}: missing target '{target}'")
            continue

        # Per-model subdir
        tag = f"{run_id}__{run_name}_{model_kind}__{model_tag}"
        model_dir = _make_unique_dir(parent_dir / "per_model" / tag)
        (model_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (model_dir / "figures").mkdir(parents=True, exist_ok=True)

        # Blocks
        X_ai, X_nonai, X_wl = get_blocks_relaxed(df)
        X_all = X_ai.join(X_nonai).join(X_wl)

        # 80/20 split (seeded)
        idx_train, idx_test = train_test_split(
            df.index,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        # Save split mask (recommended)
        pd.DataFrame({"row_index": df.index, "is_test": df.index.isin(idx_test)}).to_csv(
            model_dir / "split_mask.csv", index=False
        )

        X_tr = X_all.loc[idx_train]
        X_te = X_all.loc[idx_test]
        y_tr = df.loc[idx_train, target]
        y_te = df.loc[idx_test, target]

        # Predict (seeded for stochastic learners)
        y_pred = baseline_fold_predict(
            X_tr, X_te, y_tr,
            model_kind=model_kind,
            random_state=random_state,
        )

        pred_df = pd.DataFrame({"row_index": idx_test, "y_true": y_te.values, "y_pred": y_pred})

        for c in ["AI_model", "NonAI_gpu", "NonAI_gpu_type", "model", "gpu"]:
            if c in df.columns and c not in pred_df.columns:
                pred_df[c] = df.loc[idx_test, c].astype(str).values

        pred_df["target"] = target
        pred_df["experiment"] = run_name
        pred_df["learner"] = model_kind
        pred_df["cv"] = "split80_20"
        pred_df["per_model_file"] = model_tag

        out_pred = save_predictions(model_dir, run_name, model_kind, "split80_20", target, pred_df)

        met = compute_metrics(y_te.values, y_pred)
        summary_row = {
            "experiment": run_name,
            "learner": model_kind,
            "cv": "split80_20",
            "target": target,
            "per_model_file": model_tag,
            "n_train": int(len(idx_train)),
            "n_test": int(len(idx_test)),
            **met,
        }
        all_rows.append(summary_row)

        out_summary = save_summary(model_dir, [summary_row])

        save_manifest(
            model_dir,
            {
                "run_id": run_id,
                "run_name": run_name,
                "data_path": str(csv_path),
                "target": target,
                "test_size": test_size,
                "random_state": random_state,
                "model_kind": model_kind,
                "artifacts": {
                    "summary": str(out_summary),
                    "predictions": str(out_pred),
                    "split_mask": str(model_dir / "split_mask.csv"),
                },
            },
        )

        print(f"[OK] {model_tag} -> {out_pred}")

    # Parent-level global summary
    if all_rows:
        global_summary = pd.DataFrame(all_rows)
        global_csv = parent_dir / f"permodel_split80_baseline_summary_{model_kind}.csv"
        global_summary.to_csv(global_csv, index=False)

        save_manifest(
            parent_dir,
            {
                "run_id": run_id,
                "kind": "per_model_baseline_split80",
                "model_kind": model_kind,
                "target": target,
                "per_model_dir": str(per_model_dir),
                "global_summary": str(global_csv),
                "n_models": int(len(global_summary)),
            },
        )

        print(f"[DONE] Global summary saved: {global_csv}")
        print(f"[DONE] Parent run folder: {parent_dir}")

    return parent_dir
