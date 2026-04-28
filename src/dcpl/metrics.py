from __future__ import annotations
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MRE": float(
            np.mean(
                np.abs(y_pred - y_true) /
                np.maximum(np.abs(y_true), 1e-8)
            ) * 100.0
        ),
    }

