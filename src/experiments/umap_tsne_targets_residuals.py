"""
UMAP + t-SNE diagnostics for AIware configuration data.

- UMAP (2D) coloured by target:
    * throughput
    * median_nttft
    * median_itl

- t-SNE (2D) coloured by OOF residuals from a monolithic RF:
    residual = y_true - y_pred_oof

This script helps to visually inspect:
  (i) how performance targets vary over the configuration manifold, and
  (ii) where a monolithic model fails systematically (structured residuals),
      which motivates additive + interaction DCPL models.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

# --------------------------
# CONFIG
# --------------------------
DATA_PATH = "data/llm_pilot_data/final_data/historical_performance_data_enriched_final.csv"
OUT_DIR = Path("results/target_residual_manifold")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 10  # LOGO-style generalisation is handled elsewhere; here we focus on manifold structure.

TARGETS = ["throughput", "median_nttft", "median_itl"]

AI_COLS = [
    "model_n_parameters",
    "model_n_layers",
    "model_n_heads",
    "model_n_positions",
    "model_vocabulary_size",
    "model_relative_attention_max_distance",
    "model_relative_attention_n_buckets",
    "model_is_flash_attention",
    "model_is_encoder_decoder",
    "model_type_codegen",
    "model_type_gpt_bigcode",
    "model_type_gpt_neox",
    "model_type_llama",
    "model_type_mpt",
    "model_type_mt5",
    "model_type_t5",
    "model_torch_dtype_bfloat16",
    "model_torch_dtype_float16",
    "model_torch_dtype_float32",
]

NONAI_COLS = [
    "gpu_n_cuda_cores",
    "gpu_n_tensor_cores",
    "gpu_n_rt_cores",
    "gpu_n_sms",
    "gpu_n_rops",
    "gpu_n_tmus",
    "gpu_tflops_cuda_fp32",
    "gpu_tflops_cuda_fp64",
    "gpu_tflops_cuda_mixed",
    "gpu_tflops_tc_fp16",
    "gpu_tflops_tc_bf16",
    "gpu_tflops_tc_fp32",
    "gpu_tflops_tc_tf32",
    "gpu_tflops_tc_fp64",
    "gpu_memory_capacity_gb",
    "gpu_memory_capacity_gb_total",
    "gpu_memory_bandwidth",
    "gpu_system_interface_gen",
    "gpu_compute_capability",
    "gpu_is_sxm",
    "gpu_is_nvlink",
    "gpu_n_gpus",
    "gpu_architecture_Ampere",
    "gpu_architecture_Hopper",
    "gpu_architecture_Turing",
    "gpu_architecture_Volta",
]

WORKLOAD_COLS = [
    "num_users",
]


# --------------------------
# Helpers
# --------------------------
def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build X = [AI | NonAI | Workload], coercing to numeric and filling NaNs.
    """
    cols = [c for c in AI_COLS + NONAI_COLS + WORKLOAD_COLS if c in df.columns]
    X_block = df[cols].copy()
    X_block = X_block.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X_block.to_numpy(dtype=float), cols


def get_oof_predictions_rf(X: np.ndarray, y: np.ndarray,
                           n_splits: int = 10,
                           random_state: int = 42) -> np.ndarray:
    """
    10-fold out-of-fold predictions with a monolithic Random Forest.
    This avoids optimistic bias and lets residuals show real structure.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full_like(y, fill_value=np.nan, dtype=float)

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), 1):
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=random_state + fold,
            n_jobs=-1,
        )
        rf.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = rf.predict(X[te_idx])

    if np.isnan(oof).any():
        raise RuntimeError("OOF predictions contain NaNs – check KFold splitting.")
    return oof


def scatter_embed(df, xcol, ycol, color_col, title, out_path, cmap="viridis"):
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        df[xcol], df[ycol],
        c=df[color_col],
        cmap=cmap,
        s=22,
        alpha=0.85,
        edgecolors="none",
    )
    plt.colorbar(sc, label=color_col)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# --------------------------
# MAIN
# --------------------------
def main():
    print("[INFO] Loading dataset ...")
    df = pd.read_csv(DATA_PATH, index_col=0)

    # ---- Build feature matrix and scale ----
    X, used_cols = build_feature_matrix(df)
    print(f"[INFO] Feature matrix shape: {X.shape} | cols used: {len(used_cols)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Shared UMAP + t-SNE embeddings (same geometry for all targets) ----
    print("[INFO] Computing shared UMAP embedding ...")
    um = umap.UMAP(
        n_neighbors=30,
        min_dist=0.01,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    emb_umap = um.fit_transform(X_scaled)
    df["umap_x"] = emb_umap[:, 0]
    df["umap_y"] = emb_umap[:, 1]

    print("[INFO] Computing shared t-SNE embedding ...")
    perplexity = min(40, max(5, X_scaled.shape[0] // 10))
    ts = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200.0,
        init="pca",
        max_iter=1000,
        n_iter_without_progress=300,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    emb_tsne = ts.fit_transform(X_scaled)
    df["tsne_x"] = emb_tsne[:, 0]
    df["tsne_y"] = emb_tsne[:, 1]

    # Save base embedding
    df[["umap_x", "umap_y", "tsne_x", "tsne_y"]].to_csv(
        OUT_DIR / "embedding_base.csv", index=False
    )

    # ---- For each target: target-coloured UMAP + residual-coloured t-SNE ----
    for target in TARGETS:
        if target not in df.columns:
            print(f"[WARN] Target {target} not found in df; skipping.")
            continue

        print(f"\n[INFO] Processing target = {target}")

        y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        print("[INFO]  - Computing OOF RF residuals ...")
        oof_pred = get_oof_predictions_rf(X_scaled, y, n_splits=N_SPLITS,
                                          random_state=RANDOM_STATE)
        resid = y - oof_pred

        df[f"{target}_oof_pred"] = oof_pred
        df[f"{target}_resid"] = resid

        # Save per-target data (embedding + y + residuals)
        df[["umap_x", "umap_y", "tsne_x", "tsne_y",
            target, f"{target}_oof_pred", f"{target}_resid"]].to_csv(
            OUT_DIR / f"embedding_with_{target}_residuals.csv", index=False
        )

        # ---- UMAP coloured by target ----
        scatter_embed(
            df,
            "umap_x", "umap_y",
            target,
            title=f"UMAP coloured by {target}",
            out_path=OUT_DIR / f"umap_target_{target}.png",
            cmap="viridis",
        )

        # ---- t-SNE coloured by residual (y_true - y_pred_oof) ----
        scatter_embed(
            df,
            "tsne_x", "tsne_y",
            f"{target}_resid",
            title=f"t-SNE coloured by RF residuals (target={target})",
            out_path=OUT_DIR / f"tsne_residual_{target}.png",
            cmap="coolwarm",
        )

    print("\n[INFO] Done. All plots & CSVs in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
