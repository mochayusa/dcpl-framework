from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from dcpl.metrics import compute_metrics


def run_kfold(
    fold_fn,
    df: pd.DataFrame,
    target: str,
    X_ai: pd.DataFrame,
    X_nonai: pd.DataFrame,
    X_wl: pd.DataFrame,
    interactions: dict | None = None,
    model_kind: str = "ridge",
    n_splits: int = 10,
    random_state: int = 42
):
    """
    Generic K-Fold runner.

    fold_fn:
      - baseline_fold_predict
      - additive_fold_predict
      - additive_interaction_residual_fold_predict
    """
    y = df[target].astype(float).to_numpy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_pred = np.full(len(df), np.nan, dtype=float)
    fold_id = np.full(len(df), -1, dtype=int)

    for fold, (tr, te) in enumerate(kf.split(df), 1):
        fold_id[te] = fold

        if interactions is None:
            yhat = fold_fn(
                X_ai.iloc[tr], X_ai.iloc[te],
                X_nonai.iloc[tr], X_nonai.iloc[te],
                X_wl.iloc[tr], X_wl.iloc[te],
                y[tr],
                model_kind=model_kind
            )
        else:
            inter_tr = {k: v.iloc[tr] for k, v in interactions.items()}
            inter_te = {k: v.iloc[te] for k, v in interactions.items()}

            yhat = fold_fn(
                X_ai.iloc[tr], X_ai.iloc[te],
                X_nonai.iloc[tr], X_nonai.iloc[te],
                X_wl.iloc[tr], X_wl.iloc[te],
                inter_tr, inter_te,
                y[tr],
                base_kind=model_kind,
                inter_kind=model_kind
            )

        oof_pred[te] = yhat

    pred_df = pd.DataFrame({
        "fold": fold_id,
        "y_true": y,
        "y_pred": oof_pred
    })

    summary = compute_metrics(pred_df["y_true"], pred_df["y_pred"])
    summary.update({
        "target": target,
        "cv": f"kfold{n_splits}",
        "model": model_kind
    })

    return pred_df, summary
