# src/roofline/model.py

from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from roofline.constants import COL_II, COL_OO, COL_BB
from roofline.roofline_features import add_roofline_proxy_features

FEATURES = [
    COL_BB,           # concurrency
    COL_II,           # input tokens
    COL_OO,           # output tokens
    "roof_thr_tokens_s",
    "roof_time_s",
    "roof_ai",
]

def train_roofline_lr(df_train: pd.DataFrame, target_col: str) -> tuple[LinearRegression, list[str], pd.DataFrame]:
    df_feat = add_roofline_proxy_features(df_train)

    # numeric coercion
    for c in FEATURES + [target_col]:
        df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")

    df_feat = df_feat.dropna(subset=FEATURES + [target_col]).copy()

    X = df_feat[FEATURES].values
    y = df_feat[target_col].values

    model = LinearRegression() # the best model they used on the paper
    model.fit(X, y)
    return model, FEATURES, df_feat

def predict_roofline_lr(model: LinearRegression, df_test: pd.DataFrame, target_col: str) -> np.ndarray:
    df_feat = add_roofline_proxy_features(df_test)
    for c in FEATURES:
        df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")
    df_feat = df_feat.dropna(subset=FEATURES).copy()
    return model.predict(df_feat[FEATURES].values), df_feat.index
