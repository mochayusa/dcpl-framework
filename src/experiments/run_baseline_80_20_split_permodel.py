from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from dcpl.framework import baseline_fold_predict
from dcpl.metrics import compute_metrics

from utils.io import save_predictions, save_summary, save_manifest
from .common import prepare_run


def run_baseline_80_20_split_permodel(
    per_model_dir: str | Path = "data/llm_pilot_data/raw_data/per_model",
    target: str = "Target_throughput_tokens_per_sec",
    test_size: float = 0.20,
    random_state: int = 42,
    model_kind: str = "rf_light",
    results_root: str | Path = "results/runs",
    run_name: str = "baseline_split80_permodel",
):
    """
    For each CSV in per_model_dir:
      - build monolithic X = [AI + NonAI + Workload]
      - do one 80/20 split
      - train baseline model (LR/Ridge/RF/NN via model_kind) and predict on test
      - save predictions + summary per model
    """
    per_model_dir = Path(per_model_dir)
    files = sorted(per_model_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {per_model_dir}")

    all_rows = []

    for csv_path in files:
        model_tag = csv_path.stem
        df = pd.read_csv(csv_path)

        if target not in df.columns:
            print(f"[SKIP] {model_tag}: missing target '{target}'")
            continue

        # Create run dir + blocks
        run_dir, logger, X_ai, X_nonai, X_wl, _ = prepare_run(
            df=df,
            results_root=results_root,
            run_name=f"{run_name}__{model_tag}",
            include_interactions=False,
        )

        # Monolithic features
        X_all = X_ai.join(X_nonai).join(X_wl)

        # 80/20 split (keep indices for traceability)
        idx_train, idx_test = train_test_split(
            df.index,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        X_tr = X_all.loc[idx_train]
        X_te = X_all.loc[idx_test]
        y_tr = df.loc[idx_train, target]
        y_te = df.loc[idx_test, target]

        # Predict
        y_pred = baseline_fold_predict(X_tr, X_te, y_tr, model_kind=model_kind)

        # Build prediction DF
        pred_df = pd.DataFrame(
            {
                "y_true": y_te.values,
                "y_pred": y_pred,
            },
            index=idx_test,
        ).reset_index(names="row_index")

        # Attach identifiers if present
        for c in ["model", "gpu"]:
            if c in df.columns and c not in pred_df.columns:
                pred_df[c] = df.loc[idx_test, c].astype(str).values

        pred_df["target"] = target
        pred_df["experiment"] = "baseline_split80"
        pred_df["learner"] = model_kind
        pred_df["cv"] = "split80_20"
        pred_df["per_model_file"] = model_tag

        # Metrics
        summary = compute_metrics(y_te.values, y_pred)
        summary_row = {
            "experiment": "baseline_split80",
            "learner": model_kind,
            "cv": "split80_20",
            "target": target,
            "per_model_file": model_tag,
            "n_train": int(len(idx_train)),
            "n_test": int(len(idx_test)),
            **summary,
        }
        all_rows.append(summary_row)

        out_pred = save_predictions(run_dir, "baseline_split80", model_kind, "split80_20", target, pred_df)
        logger.info(f"Saved predictions: {out_pred}")

        out_summary = save_summary(run_dir, [summary_row])
        logger.info(f"Saved summary: {out_summary}")

        save_manifest(
            run_dir,
            {
                "run_name": f"{run_name}__{model_tag}",
                "data_path": str(csv_path),
                "target": target,
                "test_size": test_size,
                "random_state": random_state,
                "model_kind": model_kind,
                "artifacts": {
                    "summary": str(out_summary),
                    "predictions_dir": str(run_dir / "predictions"),
                },
            },
        )

    # optional: global summary across all per-model runs
    if all_rows:
        global_csv = Path(results_root) / "permodel_split80_baseline_summary.csv"
        pd.DataFrame(all_rows).to_csv(global_csv, index=False)
        print(f"[OK] Global baseline summary saved: {global_csv}")

    return all_rows
