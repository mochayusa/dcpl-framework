from __future__ import annotations
from pathlib import Path

from dcpl.framework import additive_fold_predict
from evaluation.cv import run_kfold
from evaluation.logo import run_logo
from utils.io import save_predictions, save_summary, save_manifest

from .common import load_dataset, prepare_run


def run_additive_experiment(
    data_path: str | Path,
    targets=("throughput", "median_nttft", "median_itl"),
    cv_mode="kfold10",         # "kfold10" or "logo_model"
    model_kind="ridge",        # ridge recommended
    results_root="results/runs",
    run_name="additive",
    schema: str | Path | None = None,
):
    df = load_dataset(data_path)
    run_dir, logger, X_ai, X_nonai, X_wl, interactions = prepare_run(
        df=df,
        results_root=results_root,
        run_name=run_name,
        include_interactions=False,
        schema=schema,
    )

    all_rows = []
    for target in targets:
        logger.info(f"Running additive | target={target} | cv={cv_mode} | model={model_kind}")

        if cv_mode == "kfold10":
            pred_df, summary = run_kfold(
                fold_fn=additive_fold_predict,
                df=df,
                target=target,
                X_ai=X_ai,
                X_nonai=X_nonai,
                X_wl=X_wl,
                interactions=None,
                model_kind=model_kind,
                n_splits=10,
                random_state=42
            )
            cv_tag = "kfold10"

        elif cv_mode == "logo_model":
            pred_df, summary = run_logo(
                fold_fn=additive_fold_predict,
                df=df,
                target=target,
                groups=df["model"],
                X_ai=X_ai,
                X_nonai=X_nonai,
                X_wl=X_wl,
                interactions=None,
                model_kind=model_kind
            )
            cv_tag = "logoModel"
        else:
            raise ValueError("cv_mode must be 'kfold10' or 'logo_model'")

        for c in ["model", "gpu"]:
            if c in df.columns and c not in pred_df.columns:
                pred_df[c] = df[c].astype(str).values
        pred_df["target"] = target
        pred_df["experiment"] = "additive"
        pred_df["learner"] = model_kind
        pred_df["cv"] = cv_tag

        out_pred = save_predictions(run_dir, "additive", model_kind, cv_tag, target, pred_df)
        logger.info(f"Saved predictions: {out_pred}")

        all_rows.append({"experiment": "additive", "learner": model_kind, "cv": cv_tag, **summary})

    out_summary = save_summary(run_dir, all_rows)
    logger.info(f"Saved summary: {out_summary}")

    save_manifest(run_dir, {
        "run_name": run_name,
        "data_path": str(data_path),
        "schema": str(schema) if schema is not None else None,
        "targets": list(targets),
        "cv_mode": cv_mode,
        "model_kind": model_kind,
        "artifacts": {
            "summary": str(out_summary),
            "predictions_dir": str(run_dir / "predictions")
        }
    })

    return run_dir
