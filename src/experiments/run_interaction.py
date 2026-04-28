from __future__ import annotations
from pathlib import Path

from dcpl.framework import additive_interaction_residual_fold_predict
from evaluation.cv import run_kfold
from evaluation.logo import run_logo
from utils.io import save_predictions, save_summary, save_manifest

from .common import load_dataset, prepare_run


def run_interaction_experiment(
    data_path: str | Path,
    targets=("throughput", "median_nttft", "median_itl"),
    cv_mode="kfold10",              # "kfold10" or "logo_model"
    base_kind="ridge",              # main effects learner
    inter_kind="ridge",             # residual interaction learner
    results_root="results/runs",
    run_name="interaction",
    schema: str | Path | None = None,
):
    df = load_dataset(data_path)
    run_dir, logger, X_ai, X_nonai, X_wl, interactions = prepare_run(
        df=df,
        results_root=results_root,
        run_name=run_name,
        include_interactions=True,
        schema=schema,
    )

    all_rows = []
    for target in targets:
        logger.info(
            f"Running interaction | target={target} | cv={cv_mode} | base={base_kind} | inter={inter_kind}"
        )

        if cv_mode == "kfold10":
            pred_df, summary = run_kfold(
                fold_fn=additive_interaction_residual_fold_predict,
                df=df,
                target=target,
                X_ai=X_ai,
                X_nonai=X_nonai,
                X_wl=X_wl,
                interactions=interactions,
                model_kind=base_kind,     # evaluation runner passes this to base/inter in our wrapper
                n_splits=10,
                random_state=42
            )
            cv_tag = "kfold10"

        elif cv_mode == "logo_model":
            pred_df, summary = run_logo(
                fold_fn=additive_interaction_residual_fold_predict,
                df=df,
                target=target,
                groups=df["model"],
                X_ai=X_ai,
                X_nonai=X_nonai,
                X_wl=X_wl,
                interactions=interactions,
                model_kind=base_kind
            )
            cv_tag = "logoModel"
        else:
            raise ValueError("cv_mode must be 'kfold10' or 'logo_model'")

        for c in ["model", "gpu"]:
            if c in df.columns and c not in pred_df.columns:
                pred_df[c] = df[c].astype(str).values
        pred_df["target"] = target
        pred_df["experiment"] = "interaction"
        pred_df["learner"] = f"{base_kind}+{inter_kind}"
        pred_df["cv"] = cv_tag

        out_pred = save_predictions(run_dir, "interaction", f"{base_kind}_{inter_kind}", cv_tag, target, pred_df)
        logger.info(f"Saved predictions: {out_pred}")

        all_rows.append({"experiment": "interaction", "learner": f"{base_kind}+{inter_kind}", "cv": cv_tag, **summary})

    out_summary = save_summary(run_dir, all_rows)
    logger.info(f"Saved summary: {out_summary}")

    save_manifest(run_dir, {
        "run_name": run_name,
        "data_path": str(data_path),
        "schema": str(schema) if schema is not None else None,
        "targets": list(targets),
        "cv_mode": cv_mode,
        "base_kind": base_kind,
        "inter_kind": inter_kind,
        "artifacts": {
            "summary": str(out_summary),
            "predictions_dir": str(run_dir / "predictions")
        }
    })

    return run_dir
