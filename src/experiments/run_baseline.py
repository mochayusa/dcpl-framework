from __future__ import annotations
from pathlib import Path

from dcpl.framework import baseline_fold_predict
from evaluation.cv import run_kfold
from evaluation.logo import run_logo
from utils.io import save_predictions, save_summary, save_manifest

from .common import load_dataset, prepare_run


def run_baseline_experiment(
    data_path: str | Path,
    targets=("throughput", "median_nttft", "median_itl"),
    cv_mode="logo_model",                 # "kfold10" or "logo_model"
    model_kind="rf_light",             # "rf_light" or "ridge" or "nn" etc (see dcpl/models.py)
    results_root="results/runs",
    run_name="baseline",
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

    # Build monolithic X = [AI + NonAI + Workload]
    X_all = X_ai.join(X_nonai).join(X_wl)

    all_rows = []
    for target in targets:
        logger.info(f"Running baseline | target={target} | cv={cv_mode} | model={model_kind}")

        if cv_mode == "kfold10":
            # We adapt baseline_fold_predict signature by passing X_all through X_ai arg slots
            pred_df, summary = run_kfold(
                fold_fn=lambda Xa_tr, Xa_te, Xn_tr, Xn_te, Xw_tr, Xw_te, ytr, model_kind=None:
                    baseline_fold_predict(X_all.iloc[Xa_tr.index], X_all.iloc[Xa_te.index], ytr, model_kind=model_kind),
                df=df,
                target=target,
                X_ai=X_all, X_nonai=X_all, X_wl=X_all,
                interactions=None,
                model_kind=model_kind,
                n_splits=10,
                random_state=42
            )
            cv_tag = "kfold10"

        elif cv_mode == "logo_model":
            pred_df, summary = run_logo(
                fold_fn=lambda Xa_tr, Xa_te, Xn_tr, Xn_te, Xw_tr, Xw_te, ytr, model_kind=None:
                    baseline_fold_predict(X_all.loc[Xa_tr.index], X_all.loc[Xa_te.index], ytr, model_kind=model_kind),
                df=df,
                target=target,
                groups=df["model"],
                X_ai=X_all, X_nonai=X_all, X_wl=X_all,
                interactions=None,
                model_kind=model_kind
            )
            cv_tag = "logoModel"
        else:
            raise ValueError("cv_mode must be 'kfold10' or 'logo_model'")

        # Attach IDs if present
        for c in ["model", "gpu"]:
            if c in df.columns and c not in pred_df.columns:
                pred_df[c] = df[c].astype(str).values
        pred_df["target"] = target
        pred_df["experiment"] = "baseline"
        pred_df["learner"] = model_kind
        pred_df["cv"] = cv_tag

        out_pred = save_predictions(run_dir, "baseline", model_kind, cv_tag, target, pred_df)
        logger.info(f"Saved predictions: {out_pred}")

        all_rows.append({"experiment": "baseline", "learner": model_kind, "cv": cv_tag, **summary})

    out_summary = save_summary(run_dir, all_rows)
    logger.info(f"Saved summary: {out_summary}")

    # Update manifest with run parameters
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
