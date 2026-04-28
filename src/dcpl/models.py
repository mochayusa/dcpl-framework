# src/dcpl/models.py
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
    
def make_rf_main(seed: int = 42) -> RandomForestRegressor:
    """
    RF for main blocks (AI, NonAI, Workload).
    Currently using sklearn-like "default" capacity, but seeded.
    """
    return RandomForestRegressor(
        random_state=seed,
        n_jobs=-1,
    )


def make_rf_light(seed: int = 42) -> RandomForestRegressor:
    """
    Light RF for interaction blocks (AIxNonAI, AIxWorkload, NonAIxWorkload).
    Also seeded.
    """
    return RandomForestRegressor(
        random_state=seed,
        n_jobs=-1,
    )


def make_gate(kind: str = "ridge", random_state: int = 42):
    """
    Gate / meta-learner for combining expert predictions.

    Notes:
    - LinearRegression: deterministic (no random_state).
    - Ridge: typically deterministic; random_state is not needed for common solvers.
      We keep it seed-aware by not forcing random_state at all.
    - MLPRegressor: stochastic; must be seeded for reproducibility.
    - RF gate (optional): useful as a non-linear gate alternative.
    """
    kind = kind.lower()

    if kind == "lr":
        return LinearRegression()

    if kind == "ridge":
        # Ridge is deterministic under standard solvers; keep it simple.
        return Ridge(alpha=1.0)

    if kind == "nn":
        return MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=random_state,
        )

    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError("gate kind must be one of: 'lr', 'ridge', 'nn', 'rf'")


def make_llm_pilot(seed: int = 42):
    """
    LLM-Pilot style baseline using XGBoost.
    Seeded (random_state) for reproducibility.
    """
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed, cannot use llm_pilot/xgb models.")
    return XGBRegressor(
        colsample_bytree=1.0,
        learning_rate=0.5,
        max_bin=256,
        max_depth=4,
        n_estimators=5,
        objective="reg:squarederror",
        random_state=seed,
        subsample=0.8,
        tree_method="hist",
        n_jobs=-1,
    )



def make_model(kind: str, random_state: int = 42):
    """
    General model factory used by experiments.

    kind ∈ { 'lr', 'ridge', 'rf_light', 'rf_main', 'nn', 'llm_pilot' }

    random_state is used for stochastic models (RF/NN/XGB).
    For deterministic ones, it is accepted for API consistency.
    """
    kind = kind.lower()

    if kind == "lr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

    if kind == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])

    if kind == "rf_light":
        return make_rf_light(seed=random_state)

    if kind == "rf_main":
        return make_rf_main(seed=random_state)

    if kind == "nn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=random_state,
            )),
        ])

    if kind == "llm_pilot":
        return make_llm_pilot(seed=random_state)
    
    if kind == "xgb":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed")
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError("Unknown model kind. Use: lr, ridge, rf_light, rf_main, nn, llm_pilot")
