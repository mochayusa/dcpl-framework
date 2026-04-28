from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from dcpl.metrics import compute_metrics


def run_logo(
    fold_fn,
    df: pd.DataFrame,
    target: str,
    groups: pd.Series,
    X_ai: pd.DataFrame,
    X_nonai: pd.DataFrame,
    X_wl: pd.DataFrame,
    interactions: dict | None = None,
    model_kind: str = "ridge"
):
    """
    LOGO runner (e.g., group by df["model"]).
    """
    y = df[target].astype(float).to_numpy()
    logo = LeaveOneGroupOut()

    oof_pred = np.full(len(df), np.nan, dtype=float)
    heldout_group = np.array([""] * len(df), dtype=object)

    for fold, (tr, te) in enumerate(logo.split(df, y, groups), 1):
        held = groups.iloc[te].iloc[0]
        heldout_group[te] = held

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
        "heldout_model": heldout_group,
        "y_true": y,
        "y_pred": oof_pred
    })

    summary = compute_metrics(pred_df["y_true"], pred_df["y_pred"])
    summary.update({
        "target": target,
        "cv": "LOGO(model)",
        "model": model_kind
    })

    return pred_df, summary
