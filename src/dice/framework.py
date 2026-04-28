# src/dice/framework.py
from __future__ import annotations

import pandas as pd
from typing import Dict

from .models import make_dice_model


def build_dice_features(
    X_ai: pd.DataFrame,
    X_nonai: pd.DataFrame,
    X_wl: pd.DataFrame,
    inter_all: Dict[str, pd.DataFrame],
    include_base: bool = True,
    include_interactions: bool = True,
) -> pd.DataFrame:
    """
    DICE feature construction:
      - base: [AI | NonAI | Workload]
      - interactions: [AIxNonAI | AIxWorkload | NonAIxWorkload]
      - concatenate based on flags
    """
    parts = []
    if include_base:
        parts.append(X_ai)
        parts.append(X_nonai)
        parts.append(X_wl)

    if include_interactions:
        # keep stable ordering
        for k in ["AIxNonAI", "AIxWorkload", "NonAIxWorkload"]:
            if k in inter_all and inter_all[k] is not None:
                parts.append(inter_all[k])

    if not parts:
        raise ValueError("DICE: no features selected. Set include_base and/or include_interactions to True.")

    X_all = pd.concat(parts, axis=1)
    # Ensure no duplicate column names (can break some learners)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()].copy()
    return X_all


def dice_fit_predict(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    learner_kind: str = "rf",
    random_state: int = 42,
):
    """
    Fit DICE learner on concatenated features, then predict.
    """
    m = make_dice_model(learner_kind, random_state=int(random_state))
    m.fit(X_train, y_train)
    return m.predict(X_test)
