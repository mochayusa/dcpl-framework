from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from dcpl.metrics import compute_metrics
from dcpl.models import make_model
from dcpl.blocks import AI_COLS, NONAI_COLS, WORKLOAD_COLS
from dcpl.framework import gated_blocks_and_interactions_fold_predict

# IMPORTANT: adjust this import to whatever your real interactions builder is named.
# In your repo it is in src/dcpl/interactions.py. If your function name differs,
# just change the import + call below accordingly.
from dcpl.interactions import build_interaction_block  # <-- if named differently, edit here


# =============================
# Config
# =============================
PER_MODEL_DIR = Path("data/llm_pilot_data/raw_data/per_model")
OUT_DIR = Path("results/per_model_split_80_20_throughput")
PRED_DIR = OUT_DIR / "predictions"
MANIFEST_DIR = OUT_DIR / "manifests"

TARGET = "throughput"
TEST_SIZE = 0.20
SEED = 42


# =============================
# Helpers
# =============================
def _safe_cols(df: pd.DataFrame, cols):
    """Keep only *string* columns that exist in df (protects against Ellipsis in lists)."""
    out = []
    for c in cols:
        if isinstance(c, str) and c in df.columns:
            out.append(c)
    return out


def safe_get_blocks(df: pd.DataFrame):
    """
    Robust block extraction:
    - intersection with df.columns
    - enforces numeric only (drops non-numeric columns rather than crashing)
    """
    ai_cols = _safe_cols(df, AI_COLS)
    nonai_cols = _safe_cols(df, NONAI_COLS)
    wl_cols = _safe_cols(df, WORKLOAD_COLS)

    # If your per-model CSVs have all these columns, these should be non-empty.
    # If any are empty, we fail loudly (better than silently training on nonsense).
    if len(ai_cols) == 0 or len(nonai_cols) == 0 or len(wl_cols) == 0:
        raise ValueError(
            f"Empty block after intersection. Sizes: AI={len(ai_cols)}, NonAI={len(nonai_cols)}, Workload={len(wl_cols)}. "
            f"Check column names in CSV vs blocks.py lists."
        )

    X_ai = df[ai_cols].apply(pd.to_numeric, errors="coerce")
    X_nonai = df[nonai_cols].apply(pd.to_numeric, errors="coerce")
    X_wl = df[wl_cols].apply(pd.to_numeric, errors="coerce")

    # Fill NaNs conservatively (median) to avoid crashes; you can also do it via pipelines.
    for X in (X_ai, X_nonai, X_wl):
        X[:] = X.fillna(X.median(numeric_only=True))

    return X_ai, X_nonai, X_wl


def concat_blocks(X_ai, X_nonai, X_wl) -> pd.DataFrame:
    return pd.concat([X_ai, X_nonai, X_wl], axis=1)


# =============================
# Main
# =============================
def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    csv_files = sorted(PER_MODEL_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No per-model CSV found under: {PER_MODEL_DIR}")

    baseline_kinds = ["lr", "ridge", "rf_light", "nn"]  # add "llm_pilot" if you want

    for f in csv_files:
        model_name = f.stem
        df = pd.read_csv(f)

        if TARGET not in df.columns:
            print(f"[SKIP] {model_name}: missing target '{TARGET}'")
            continue

        # target cleanup
        y = pd.to_numeric(df[TARGET], errors="coerce")
        keep = np.isfinite(y.to_numpy())
        df = df.loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True).to_numpy(dtype=float)

        # blocks
        X_ai, X_nonai, X_wl = safe_get_blocks(df)
        X_mono = concat_blocks(X_ai, X_nonai, X_wl)

        # split indices (80/20) – same split for all methods
        idx = np.arange(len(df))
        tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, shuffle=True)

        # train/test slices
        X_ai_tr, X_ai_te = X_ai.iloc[tr_idx], X_ai.iloc[te_idx]
        X_ni_tr, X_ni_te = X_nonai.iloc[tr_idx], X_nonai.iloc[te_idx]
        X_wl_tr, X_wl_te = X_wl.iloc[tr_idx], X_wl.iloc[te_idx]
        X_m_tr, X_m_te = X_mono.iloc[tr_idx], X_mono.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # interaction blocks (built on the FULL df then sliced, to avoid mismatch)
        # NOTE: if your build function expects cols, pass X_ai.columns etc.
        inter_full = build_interaction_block(
            df=pd.concat([X_ai, X_nonai, X_wl], axis=1),
            ai_cols=list(X_ai.columns),
            nonai_cols=list(X_nonai.columns),
            wl_cols=list(X_wl.columns),
            include=["AIxNonAI", "AIxWorkload", "NonAIxWorkload"],
        )
        inter_tr = {k: v.iloc[tr_idx] for k, v in inter_full.items()}
        inter_te = {k: v.iloc[te_idx] for k, v in inter_full.items()}

        # manifest
        manifest = {
            "model_file": str(f),
            "model_name": model_name,
            "target": TARGET,
            "n_rows": int(len(df)),
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "seed": SEED,
            "test_size": TEST_SIZE,
            "ai_features": list(X_ai.columns),
            "nonai_features": list(X_nonai.columns),
            "workload_features": list(X_wl.columns),
        }
        (MANIFEST_DIR / f"{model_name}.json").write_text(json.dumps(manifest, indent=2))

        # ----------------------
        # Baselines
        # ----------------------
        out_model_dir = PRED_DIR / model_name
        out_model_dir.mkdir(parents=True, exist_ok=True)

        for kind in baseline_kinds:
            m = make_model(kind)
            m.fit(X_m_tr, y_tr)
            y_pred = m.predict(X_m_te)

            met = compute_metrics(y_te, y_pred)
            rows.append({
                "per_model": model_name,
                "method": f"baseline_{kind}",
                "target": TARGET,
                **met,
            })

            pd.DataFrame({
                "row_index": te_idx,
                "y_true": y_te,
                "y_pred": y_pred,
            }).to_csv(out_model_dir / f"baseline_{kind}_{TARGET}.csv", index=False)

        # ----------------------
        # DCPL (gated blocks + interactions)
        # ----------------------
        y_pred_dcpl = gated_blocks_and_interactions_fold_predict(
            X_ai_train=X_ai_tr, X_ai_test=X_ai_te,
            X_nonai_train=X_ni_tr, X_nonai_test=X_ni_te,
            X_wl_train=X_wl_tr, X_wl_test=X_wl_te,
            inter_train=inter_tr, inter_test=inter_te,
            y_train=y_tr,
            # If your function supports knobs, you can pass:
            # model_kind_main="rf_main", model_kind_inter="rf_light", gate_kind="ridge", inner_folds=5, ...
        )

        met = compute_metrics(y_te, y_pred_dcpl)
        rows.append({
            "per_model": model_name,
            "method": "dcpl_gated_interactions",
            "target": TARGET,
            **met,
        })

        pd.DataFrame({
            "row_index": te_idx,
            "y_true": y_te,
            "y_pred": y_pred_dcpl,
        }).to_csv(out_model_dir / f"dcpl_{TARGET}.csv", index=False)

        print(f"[OK] {model_name}: done")

    # summary
    summary = pd.DataFrame(rows).sort_values(["per_model", "method"])
    out_summary = OUT_DIR / f"summary_{TARGET}.csv"
    summary.to_csv(out_summary, index=False)
    print(f"\nSaved summary: {out_summary}")


if __name__ == "__main__":
    run()
