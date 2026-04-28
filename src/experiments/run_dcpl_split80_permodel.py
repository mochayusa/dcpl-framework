# src/erun_dcpl_split80_permodel.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from dcpl.blocks import get_blocks_relaxed
from dcpl.interactions import build_all_interactions
from dcpl.framework import gated_blocks_and_interactions_fold_predict
from dcpl.metrics import compute_metrics

from utils.io import save_predictions, save_summary, save_manifest


def _make_unique_dir(path: Path) -> Path:
    """Create a unique directory; if exists append _1, _2, ..."""
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


def run_dcpl_split80_permodel(
    per_model_dir: str | Path = "data/llm_pilot_data/raw_data/per_model",
    target: str = "Target_throughput_tokens_per_sec",
    test_size: float = 0.20,
    random_state: int = 42,
    gate_kind: str = "ridge",
    inner_splits: int = 5,
    results_root: str | Path = "results/runs",
    run_name: str = "dcpl_split80",
    schema: str | Path | None = None,
):
    """
    DCPL (gated blocks + interactions) per-model evaluation with one 80/20 split per model.

    Folder structure:
      results/runs/<RUN_ID>/per_model/<RUN_ID>__<run_name>_gate=<gate_kind>__<model_tag>/
        - predictions/*.csv
        - split_mask.csv
        - summary.csv
        - manifest.json
        - figures/ (reserved)
    """
    per_model_dir = Path(per_model_dir)
    files = sorted(per_model_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {per_model_dir}")

    # Parent run dir (one per execution)
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

        # blocks (relaxed schema)
        X_ai, X_nonai, X_wl = get_blocks_relaxed(df, schema=schema)

        # 80/20 split (seeded + shuffled)
        idx_train, idx_test = train_test_split(
            df.index,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        # Guard: inner_splits cannot exceed training samples
        if inner_splits < 2:
            raise ValueError("inner_splits must be >= 2")
        if inner_splits > len(idx_train):
            raise ValueError(
                f"inner_splits={inner_splits} > n_train={len(idx_train)} for model '{model_tag}'. "
                "Reduce inner_splits or ensure more training samples."
            )

        # per-model subdir
        tag = f"{run_id}__{run_name}_gate={gate_kind}__{model_tag}"
        model_dir = _make_unique_dir(parent_dir / "per_model" / tag)
        (model_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (model_dir / "figures").mkdir(parents=True, exist_ok=True)

        # Save split mask for auditability (optional but recommended)
        pd.DataFrame({"row_index": df.index, "is_test": df.index.isin(idx_test)}).to_csv(
            model_dir / "split_mask.csv", index=False
        )

        # interactions on the SAME rows
        df_blocks = pd.concat([X_ai, X_nonai, X_wl], axis=1)
        inter_all = build_all_interactions(
            df_blocks,
            X_ai.columns.tolist(),
            X_nonai.columns.tolist(),
            X_wl.columns.tolist(),
        )

        # train/test slices
        X_ai_tr, X_ai_te = X_ai.loc[idx_train], X_ai.loc[idx_test]
        X_n_tr, X_n_te = X_nonai.loc[idx_train], X_nonai.loc[idx_test]
        X_w_tr, X_w_te = X_wl.loc[idx_train], X_wl.loc[idx_test]

        inter_tr = {k: v.loc[idx_train] for k, v in inter_all.items()}
        inter_te = {k: v.loc[idx_test] for k, v in inter_all.items()}

        y_tr = df.loc[idx_train, target]
        y_te = df.loc[idx_test, target]

        # DCPL prediction (CROSS-FITTING + SEEDED)
        y_pred = gated_blocks_and_interactions_fold_predict(
            X_ai_tr, X_ai_te,
            X_n_tr, X_n_te,
            X_w_tr, X_w_te,
            inter_tr, inter_te,
            y_tr,
            inner_splits=inner_splits,
            gate_kind=gate_kind,
            random_state=random_state,  # <<< WAJIB: propagate seed ke cross-fitting
        )

        # prediction DF
        pred_df = pd.DataFrame({"row_index": idx_test, "y_true": y_te.values, "y_pred": y_pred})

        # attach identifiers if present
        for c in ["AI_model", "NonAI_gpu", "NonAI_gpu_type", "model", "gpu"]:
            if c in df.columns and c not in pred_df.columns:
                pred_df[c] = df.loc[idx_test, c].astype(str).values

        pred_df["target"] = target
        pred_df["experiment"] = run_name
        pred_df["learner"] = f"dcpl_gate={gate_kind}"
        pred_df["gate_kind"] = gate_kind
        pred_df["cv"] = "split80_20"
        pred_df["per_model_file"] = model_tag

        out_pred = save_predictions(
            model_dir,
            run_name,
            f"dcpl_gate-{gate_kind}",
            "split80_20",
            target,
            pred_df,
        )

        # metrics
        met = compute_metrics(y_te.values, y_pred)
        summary_row = {
            "experiment": run_name,
            "learner": f"dcpl_gate={gate_kind}",
            "gate_kind": gate_kind,
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
                "schema": str(schema) if schema is not None else None,
                "test_size": test_size,
                "random_state": random_state,
                "gate_kind": gate_kind,
                "inner_splits": inner_splits,
                "artifacts": {
                    "summary": str(out_summary),
                    "predictions": str(out_pred),
                    "split_mask": str(model_dir / "split_mask.csv"),
                },
            },
        )

        print(f"[OK] {model_tag} -> {out_pred}")

    # parent-level global summary
    if all_rows:
        global_df = pd.DataFrame(all_rows)
        global_csv = parent_dir / f"permodel_split80_dcpl_summary_gate-{gate_kind}.csv"
        global_df.to_csv(global_csv, index=False)

        save_manifest(
            parent_dir,
            {
                "run_id": run_id,
                "kind": "per_model_dcpl_split80",
                "gate_kind": gate_kind,
                "inner_splits": inner_splits,
                "target": target,
                "per_model_dir": str(per_model_dir),
                "schema": str(schema) if schema is not None else None,
                "global_summary": str(global_csv),
                "n_models": int(len(global_df)),
            },
        )

        print(f"[DONE] Global DCPL summary saved: {global_csv}")
        print(f"[DONE] Parent run folder: {parent_dir}")

    return parent_dir
