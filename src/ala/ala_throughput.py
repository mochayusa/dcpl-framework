import numpy as np
import pandas as pd
import math

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor
from scipy.optimize import curve_fit
from ala.constants import COL_II, COL_OO, COL_BB, COL_THROUGHPUT


def exp_throughput(bb, a, b, c):
    """
    Saturating exponential throughput model:
        throughput(bb) = a + c * (1 - exp(-b * bb))

    a: base latency (low concurrency)
    c: additional latency due to congestion
    b: sensitivity to concurrency
    """
    return a + c * (1.0 - np.exp(-b * bb))

# ============================================================
# 3. Fit (a, b, c) per (ii, oo) group
# ============================================================

def fit_throughput_group(bb_vals, thr_vals):
    """
    Fit (a, b, c) for a single (ii, oo) group using a saturating exponential:
        thr(bb) = a + c * (1 - exp(-b * bb))

    Robust init + bounds:
      a >= 0, b >= 0, c >= 0
    """
    bb = np.asarray(bb_vals, dtype=float)
    thr = np.asarray(thr_vals, dtype=float)

    # Basic sanity: non-negative throughput
    thr = np.maximum(thr, 0.0)

    if len(np.unique(bb)) < 2:
        # fallback: constant-ish throughput
        a0 = float(np.percentile(thr, 10))
        c0 = float(max(np.percentile(thr, 90) - a0, 1e-3))
        b0 = 0.01
        return (a0, b0, c0)

    thr_p10, thr_p90 = np.percentile(thr, [10, 90])
    bb_p10, bb_p90   = np.percentile(bb,  [10, 90])

    eps = 1e-3
    bb_p90 = max(bb_p90, bb_p10 + eps)

    a0 = float(max(thr_p10, 0.0))
    c0 = float(max(thr_p90 - thr_p10, 1e-3))
    b0 = float(1.0 / max(bb_p90 - bb_p10, 1e-3))

    p0 = (a0, b0, c0)
    bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])

    try:
        popt, _ = curve_fit(
            exp_throughput,
            bb,
            thr,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )
        return tuple(map(float, popt))
    except Exception:
        return (a0, b0, c0)

# ============================================================
# 4. Build parameter DB + XGBoost training table
# ============================================================
def make_param_features(ii, oo):
    ii = float(ii)
    oo = float(oo)

    logii      = np.log1p(ii)
    logoo      = np.log1p(oo)
    logratio   = np.log1p(ii / (oo + 1e-6))
    ii_oo_ratio = ii / (oo + 1.0)
    ii_ii_ratio = ii / (ii + 1.0)

    return np.array([[ii, oo, logii, logoo, logratio,
                      ii_oo_ratio, ii_ii_ratio]], dtype=float)

def build_throughput_db_and_training_params(df_train):
    """
    For each (ii, oo) group in df_train:
      - fit (a, b, c) of the throughput saturating exponential
      - store in param_db[(ii, oo)]
      - create a row in T_df with features + targets (a, b, c)
    """
    groups = df_train.groupby([COL_II, COL_OO], dropna=False)

    param_db = {}
    records = []

    for (ii, oo), g in groups:
        ii_f, oo_f = float(ii), float(oo)

        a_hat, b_hat, c_hat = fit_throughput_group(
            g[COL_BB].values,
            g[COL_THROUGHPUT].values,
        )

        key = (ii_f, oo_f)
        param_db[key] = (a_hat, b_hat, c_hat)

        feat = make_param_features(ii_f, oo_f).ravel()
        # feat order: [ii, oo, logii, logoo, logratio, ii_oo_ratio, ii_ii_ratio]
        records.append({
            "ii": feat[0],
            "oo": feat[1],
            "logii": feat[2],
            "logoo": feat[3],
            "logratio": feat[4],
            "ii_oo_ratio": feat[5],
            "ii_ii_ratio": feat[6],
            "a": a_hat,
            "b": b_hat,
            "c": c_hat,
        })

    T_df = pd.DataFrame(records)
    return param_db, T_df

# ============================================================
# 5. Train XGBoost parameter regressor (multi-output)
# ============================================================

def train_param_regressor(T_df):
    """
    Train MultiOutput XGBoost to predict (a, b, c)
    from (ii, oo)-derived features.
    """
    if T_df.empty:
        return None

    feature_cols = ["ii", "oo", "logii", "logoo",
                    "logratio", "ii_oo_ratio", "ii_ii_ratio"]
    target_cols = ["a", "b", "c"]

    X = T_df[feature_cols].values
    y = T_df[target_cols].values

    base_xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
    )

    model = MultiOutputRegressor(base_xgb)
    model.fit(X, y)
    return model

def ala_predict_throughput(df_rows, param_db, param_regressor, clip_nonneg: bool = True):
    preds = []

    for _, row in df_rows.iterrows():
        ii = float(row[COL_II])
        oo = float(row[COL_OO])
        bb = float(row[COL_BB])

        key = (ii, oo)

        if key in param_db:
            a_hat, b_hat, c_hat = param_db[key]
        else:
            X_feat = make_param_features(ii, oo)  # shape (1,7)
            a_hat, b_hat, c_hat = param_regressor.predict(X_feat)[0]

            # Safety: keep parameters in a sane domain
            a_hat = float(max(a_hat, 0.0))
            b_hat = float(max(b_hat, 0.0))
            c_hat = float(max(c_hat, 0.0))

        y = float(exp_throughput(bb, a_hat, b_hat, c_hat))
        if clip_nonneg:
            y = max(y, 0.0)

        preds.append(y)

    return np.asarray(preds, dtype=float)

# ============================================================
# 7. Metrics function (R2, RMSE, MAE, MAPE, MdAPE)
# ============================================================

def compute_metrics(y_true, y_pred, eps: float = 1e-8) -> dict:
    """
    Regression metrics:
      - R2
      - MAE
      - RMSE
      - MRE (%) = mean(|y_true - y_pred| / max(|y_true|, eps)) * 100
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    denom = np.maximum(np.abs(y_true), eps)
    mre = float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MRE": mre,
    }