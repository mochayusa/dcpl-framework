# src/dcpl/framework.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional
from sklearn.model_selection import KFold

from .models import make_rf_main, make_rf_light, make_gate, make_model


def baseline_fold_predict(
    X_train,
    X_test,
    y_train,
    model_kind: str = "rf_light",
    random_state: int | None = None,
):
    """
    Baseline (monolithic): fit one model on concatenated features.
    Seeded for stochastic learners (RF/NN/XGB).
    """
    seed = 42 if random_state is None else int(random_state)
    model = make_model(model_kind, random_state=seed)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def additive_fold_predict(
    X_ai_train, X_ai_test,
    X_nonai_train, X_nonai_test,
    X_wl_train, X_wl_test,
    y_train,
    model_kind: str = "ridge",
    random_state: Optional[int] = None,
):
    """Additive: yhat = f_ai + f_nonai + f_wl (seeded if supported)."""
    try:
        m_ai = make_model(model_kind, random_state=random_state)
        m_nonai = make_model(model_kind, random_state=random_state)
        m_wl = make_model(model_kind, random_state=random_state)
    except TypeError:
        m_ai = make_model(model_kind)
        m_nonai = make_model(model_kind)
        m_wl = make_model(model_kind)

    m_ai.fit(X_ai_train, y_train)
    m_nonai.fit(X_nonai_train, y_train)
    m_wl.fit(X_wl_train, y_train)

    return (
        m_ai.predict(X_ai_test)
        + m_nonai.predict(X_nonai_test)
        + m_wl.predict(X_wl_test)
    )


def additive_interaction_residual_fold_predict(
    X_ai_train, X_ai_test,
    X_nonai_train, X_nonai_test,
    X_wl_train, X_wl_test,
    inter_train: Dict[str, pd.DataFrame],
    inter_test: Dict[str, pd.DataFrame],
    y_train,
    base_kind: str = "ridge",
    inter_kind: str = "ridge",
    random_state: Optional[int] = None,
):
    """Additive + residual interactions (seeded if supported)."""
    try:
        m_ai = make_model(base_kind, random_state=random_state)
        m_nonai = make_model(base_kind, random_state=random_state)
        m_wl = make_model(base_kind, random_state=random_state)
    except TypeError:
        m_ai = make_model(base_kind)
        m_nonai = make_model(base_kind)
        m_wl = make_model(base_kind)

    m_ai.fit(X_ai_train, y_train)
    m_nonai.fit(X_nonai_train, y_train)
    m_wl.fit(X_wl_train, y_train)

    main_train = (
        m_ai.predict(X_ai_train)
        + m_nonai.predict(X_nonai_train)
        + m_wl.predict(X_wl_train)
    )
    resid = np.asarray(y_train, dtype=float) - np.asarray(main_train, dtype=float)

    try:
        g_aixn = make_model(inter_kind, random_state=random_state)
        g_aixw = make_model(inter_kind, random_state=random_state)
        g_nixw = make_model(inter_kind, random_state=random_state)
    except TypeError:
        g_aixn = make_model(inter_kind)
        g_aixw = make_model(inter_kind)
        g_nixw = make_model(inter_kind)

    if inter_train["AIxNonAI"].shape[1] > 0:
        g_aixn.fit(inter_train["AIxNonAI"], resid)
    if inter_train["AIxWorkload"].shape[1] > 0:
        g_aixw.fit(inter_train["AIxWorkload"], resid)
    if inter_train["NonAIxWorkload"].shape[1] > 0:
        g_nixw.fit(inter_train["NonAIxWorkload"], resid)

    main_test = (
        m_ai.predict(X_ai_test)
        + m_nonai.predict(X_nonai_test)
        + m_wl.predict(X_wl_test)
    )

    rhat = np.zeros(len(X_ai_test), dtype=float)
    if inter_test["AIxNonAI"].shape[1] > 0:
        rhat += g_aixn.predict(inter_test["AIxNonAI"])
    if inter_test["AIxWorkload"].shape[1] > 0:
        rhat += g_aixw.predict(inter_test["AIxWorkload"])
    if inter_test["NonAIxWorkload"].shape[1] > 0:
        rhat += g_nixw.predict(inter_test["NonAIxWorkload"])

    return np.asarray(main_test, dtype=float) + rhat


def _fold_seed(base_seed: int, fold_id: int, offset: int = 0) -> int:
    """Deterministic per-fold seed."""
    return int(base_seed + 10_000 * offset + 101 * fold_id)


def _crossfit_expert_preds(
    X_train: pd.DataFrame,
    y_train,
    model_fn: Callable[[int], object],
    n_splits: int,
    seed: int,
    seed_offset: int,
) -> np.ndarray:
    """Cross-fit OOF predictions with seeded KFold and seeded experts."""
    y_train = np.asarray(y_train, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.full(len(X_train), np.nan, dtype=float)

    for fold_id, (tr, va) in enumerate(kf.split(X_train)):
        m = model_fn(_fold_seed(seed, fold_id, offset=seed_offset))
        m.fit(X_train.iloc[tr], y_train[tr])
        oof[va] = m.predict(X_train.iloc[va])

    if np.isnan(oof).any():
        raise RuntimeError("OOF predictions contain NaN. Check inner CV config.")
    return oof


def gated_blocks_and_interactions_fold_predict(
    X_ai_train, X_ai_test,
    X_nonai_train, X_nonai_test,
    X_wl_train, X_wl_test,
    inter_train: Dict[str, pd.DataFrame],
    inter_test: Dict[str, pd.DataFrame],
    y_train,
    base_kind=None,
    inter_kind=None,
    inner_splits: int = 5,
    gate_kind: str = "ridge",
    random_state: int = 42,
):
    """DCPL split80: fully seeded by random_state."""
    y_train = np.asarray(y_train, dtype=float)
    seed = int(random_state)

    # Expert factories (seeded)
    def rf_main_factory(s: int):
        return make_rf_main(seed=s)

    def rf_light_factory(s: int):
        return make_rf_light(seed=s)

    # OOF for gate (main)
    ai_oof = _crossfit_expert_preds(X_ai_train, y_train, rf_main_factory, inner_splits, seed, seed_offset=1)
    ni_oof = _crossfit_expert_preds(X_nonai_train, y_train, rf_main_factory, inner_splits, seed, seed_offset=2)
    wl_oof = _crossfit_expert_preds(X_wl_train, y_train, rf_main_factory, inner_splits, seed, seed_offset=3)

    # OOF for gate (interactions)
    def maybe_crossfit_inter(block_name: str, offset: int):
        block = inter_train[block_name]
        if block.shape[1] == 0:
            return np.zeros(len(y_train), dtype=float)
        return _crossfit_expert_preds(block, y_train, rf_light_factory, inner_splits, seed, seed_offset=offset)

    aixni_oof = maybe_crossfit_inter("AIxNonAI", offset=4)
    aixwl_oof = maybe_crossfit_inter("AIxWorkload", offset=5)
    nixi_wl_oof = maybe_crossfit_inter("NonAIxWorkload", offset=6)

    Z_train = np.column_stack([ai_oof, ni_oof, wl_oof, aixni_oof, aixwl_oof, nixi_wl_oof])

    # Gate (seeded if make_gate supports it)
    try:
        gate = make_gate(gate_kind, random_state=seed)
    except TypeError:
        gate = make_gate(gate_kind)

    gate.fit(Z_train, y_train)

    # Full-train experts (seeded)
    rf_ai = make_rf_main(seed=seed + 11); rf_ai.fit(X_ai_train, y_train)
    rf_ni = make_rf_main(seed=seed + 12); rf_ni.fit(X_nonai_train, y_train)
    rf_wl = make_rf_main(seed=seed + 13); rf_wl.fit(X_wl_train, y_train)

    def fit_inter_full(block_name: str, add_seed: int):
        block_tr = inter_train[block_name]
        if block_tr.shape[1] == 0:
            return None
        m = make_rf_light(seed=seed + add_seed)
        m.fit(block_tr, y_train)
        return m

    rf_aixni = fit_inter_full("AIxNonAI", add_seed=21)
    rf_aixwl = fit_inter_full("AIxWorkload", add_seed=22)
    rf_nixi_wl = fit_inter_full("NonAIxWorkload", add_seed=23)

    # Predict test
    Z_test_parts = [
        rf_ai.predict(X_ai_test),
        rf_ni.predict(X_nonai_test),
        rf_wl.predict(X_wl_test),
    ]

    def predict_inter_full(model, block_name: str):
        if model is None:
            return np.zeros(len(X_ai_test), dtype=float)
        return model.predict(inter_test[block_name])

    Z_test_parts.append(predict_inter_full(rf_aixni, "AIxNonAI"))
    Z_test_parts.append(predict_inter_full(rf_aixwl, "AIxWorkload"))
    Z_test_parts.append(predict_inter_full(rf_nixi_wl, "NonAIxWorkload"))

    Z_test = np.column_stack(Z_test_parts)
    return gate.predict(Z_test)
