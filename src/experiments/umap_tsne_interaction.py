# src/experiments/umap_tsne_interaction.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# CONFIG: your 3 blocks
# --------------------------------------------------------------
AI_COLS = [
    "model_n_parameters","model_n_layers","model_n_heads","model_n_positions",
    "model_vocabulary_size","model_relative_attention_max_distance",
    "model_relative_attention_n_buckets","model_is_flash_attention",
    "model_is_encoder_decoder","model_type_codegen","model_type_gpt_bigcode",
    "model_type_gpt_neox","model_type_llama","model_type_mpt","model_type_mt5",
    "model_type_t5","model_torch_dtype_bfloat16","model_torch_dtype_float16",
    "model_torch_dtype_float32"
]

NONAI_COLS = [
    "gpu_n_cuda_cores","gpu_n_tensor_cores","gpu_n_rt_cores","gpu_n_sms",
    "gpu_n_rops","gpu_n_tmus","gpu_tflops_cuda_fp32","gpu_tflops_cuda_fp64",
    "gpu_tflops_cuda_mixed","gpu_tflops_tc_fp16","gpu_tflops_tc_bf16",
    "gpu_tflops_tc_fp32","gpu_tflops_tc_tf32","gpu_tflops_tc_fp64",
    "gpu_memory_capacity_gb","gpu_memory_capacity_gb_total",
    "gpu_memory_bandwidth","gpu_system_interface_gen",
    "gpu_compute_capability","gpu_is_sxm","gpu_is_nvlink","gpu_n_gpus",
    "gpu_architecture_Ampere","gpu_architecture_Hopper",
    "gpu_architecture_Turing","gpu_architecture_Volta"
]

WORKLOAD_COLS = ["num_users"]


# --------------------------------------------------------------
# Utility
# --------------------------------------------------------------
def ensure_numeric(df, cols):
    """Convert bool → int and assert numeric."""
    for c in cols:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
        if df[c].dtype == object:
            raise ValueError(f"Column {c} is object dtype. Encode it first.")


# --------------------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------------------
def run_interaction_visualisation_experiment(
    data_path="data/llm_pilot_data/final_data/historical_performance_data_enriched_final.csv",
    out_dir="results/interaction_visualisation",
    random_state=42,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading dataset: {data_path}")
    df = pd.read_csv(data_path, index_col=0)

    # --- Validate blocks ---
    ensure_numeric(df, AI_COLS)
    ensure_numeric(df, NONAI_COLS)
    ensure_numeric(df, WORKLOAD_COLS)

    X_ai = df[AI_COLS].copy().astype(float)
    X_nonai = df[NONAI_COLS].copy().astype(float)
    X_wl = df[WORKLOAD_COLS].copy().astype(float)

    # Concatenate
    X = pd.concat([X_ai, X_nonai, X_wl], axis=1)

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----------------------------------------------------------
    # BLOCK STRENGTHS (for colouring)
    # ----------------------------------------------------------
    df["block_ai_strength"] = np.linalg.norm(X_ai.values, axis=1)
    df["block_nonai_strength"] = np.linalg.norm(X_nonai.values, axis=1)
    df["block_wl_strength"] = np.linalg.norm(X_wl.values, axis=1)

    # ----------------------------------------------------------
    # UMAP
    # ----------------------------------------------------------
    print("[INFO] Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state
    )
    X_umap = reducer.fit_transform(X_scaled)

    df["umap_x"] = X_umap[:, 0]
    df["umap_y"] = X_umap[:, 1]
    df[["umap_x","umap_y"]].to_csv(out_dir / "umap_embedding.csv", index=False)

    # Plot UMAP for each block
    for col in ["block_ai_strength","block_nonai_strength","block_wl_strength"]:
        plt.figure(figsize=(7,5))
        plt.scatter(df["umap_x"], df["umap_y"], c=df[col], cmap="viridis", s=10)
        plt.colorbar(label=col)
        plt.title(f"UMAP coloured by {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"umap_{col}.png", dpi=300)
        plt.close()

    # ----------------------------------------------------------
    # t-SNE
    # ----------------------------------------------------------
    print("[INFO] Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=random_state
    )
    X_tsne = tsne.fit_transform(X_scaled)

    df["tsne_x"] = X_tsne[:, 0]
    df["tsne_y"] = X_tsne[:, 1]
    df[["tsne_x","tsne_y"]].to_csv(out_dir / "tsne_embedding.csv", index=False)

    for col in ["block_ai_strength","block_nonai_strength","block_wl_strength"]:
        plt.figure(figsize=(7,5))
        plt.scatter(df["tsne_x"], df["tsne_y"], c=df[col], cmap="plasma", s=10)
        plt.colorbar(label=col)
        plt.title(f"t-SNE coloured by {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"tsne_{col}.png", dpi=300)
        plt.close()

    print(f"[INFO] All embeddings + plots saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    run_interaction_visualisation_experiment()
