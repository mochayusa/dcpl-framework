from __future__ import annotations
from pathlib import Path

from dcpl.framework import gated_blocks_and_interactions_fold_predict
from experiments.common import load_dataset, prepare_run
from evaluation.cv import run_kfold
from evaluation.logo import run_logo
from utils.io import save_predictions, save_summary, save_manifest


def run_gated_interaction_experiment(
    data_path: str | Path,
    targets=("throughput", "median_nttft", "median_itl"),
    cv_mode="logo_model",          # "kfold10" or "logo_model"
    results_root="results/runs",
    run_name="gated_interaction",
    inner_splits: int = 10,
    gate_kind: str = "ridge",
    schema: str | Path | None = None,
):
    """
    Full framework:
      - RF experts on AI, NonAI, Workload
      - RF (light) experts on AIxNonAI, AIxWorkload, NonAIxWorkload
      - Ridge gate on 6 expert predictions
    """

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
            f"Running gated interaction | target={target} | cv={cv_mode} | gate={gate_kind}"
        )

        if cv_mode == "kfold10":
            pred_df, summary = run_kfold(
                fold_fn=lambda Xa_tr, Xa_te, Xn_tr, Xn_te, Xw_tr, Xw_te, inter_tr, inter_te, ytr, base_kind=None, inter_kind=None:
                    gated_blocks_and_interactions_fold_predict(
                        Xa_tr, Xa_te,
                        Xn_tr, Xn_te,
                        Xw_tr, Xw_te,
                        inter_tr, inter_te,
                        ytr,
                        inner_splits=inner_splits,
                        gate_kind=gate_kind,
                    ),
                df=df,
                target=target,
                X_ai=X_ai,
                X_nonai=X_nonai,
                X_wl=X_wl,
                interactions=interactions,
                model_kind="rf_light",  # not used inside, but required by signature
                n_splits=10,
                random_state=42,
            )
            cv_tag = "kfold10"

        elif cv_mode == "logo_model":
            pred_df, summary = run_logo(
                fold_fn=lambda Xa_tr, Xa_te, Xn_tr, Xn_te, Xw_tr, Xw_te, inter_tr, inter_te, ytr, base_kind=None, inter_kind=None:
                    gated_blocks_and_interactions_fold_predict(
                        Xa_tr, Xa_te,
                        Xn_tr, Xn_te,
                        Xw_tr, Xw_te,
                        inter_tr, inter_te,
                        ytr,
                        inner_splits=inner_splits,
                        gate_kind=gate_kind,
                    ),
                df=df,
                target=target,
                groups=df["model"],
                X_ai=X_ai,
                X_nonai=X_nonai,
                X_wl=X_wl,
                interactions=interactions,
                model_kind="rf_light",
            )
            cv_tag = "logoModel"
        else:
            raise ValueError("cv_mode must be 'kfold10' or 'logo_model'")

        # Attach IDs if available
        for c in ["model", "gpu"]:
            if c in df.columns and c not in pred_df.columns:
                pred_df[c] = df[c].astype(str).values

        pred_df["target"] = target
        pred_df["experiment"] = "gated_interaction"
        pred_df["learner"] = f"rf_blocks+rf_interactions+{gate_kind}"
        pred_df["cv"] = cv_tag

        out_pred = save_predictions(run_dir, "gated_interaction", "rf_rf_gate", cv_tag, target, pred_df)
        logger.info(f"Saved predictions: {out_pred}")

        all_rows.append({
            "experiment": "gated_interaction",
            "learner": f"rf_blocks+rf_interactions+{gate_kind}",
            "cv": cv_tag,
            **summary,
        })

    out_summary = save_summary(run_dir, all_rows)
    logger.info(f"Saved summary: {out_summary}")

    save_manifest(run_dir, {
        "run_name": run_name,
        "data_path": str(data_path),
        "schema": str(schema) if schema is not None else None,
        "targets": list(targets),
        "cv_mode": cv_mode,
        "gate_kind": gate_kind,
        "inner_splits": inner_splits,
        "artifacts": {
            "summary": str(out_summary),
            "predictions_dir": str(run_dir / "predictions"),
        },
    })

    return run_dir
