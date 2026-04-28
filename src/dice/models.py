# src/dice/models.py
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    XGBRegressor = None
    _HAS_XGB = False


def make_dice_model(kind: str, random_state: int = 42):
    """
    DICE learner factory.
    ALL MODELS USE DEFAULT HYPERPARAMETERS.
    Only random_state is set for reproducibility.
    """
    kind = kind.lower().strip()

    if kind == "lr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

    if kind == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge()),  # default alpha=1.0
        ])

    if kind == "rf":
        return RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
        )

    if kind == "nn":
        return Pipeline([
            ("scaler", StandardScaler()),
            # default everything else
            ("model", MLPRegressor(
                random_state=random_state,  
            )),
        ])

    if kind == "xgb":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed")
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError("Unknown DICE model kind. Use: lr, ridge, rf, nn, xgb")
