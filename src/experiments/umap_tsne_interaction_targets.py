import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------
# CONFIG
# --------------------------
DATA_PATH = "data/llm_pilot_data/final_data/historical_performance_data_enriched_final.csv"
OUT_DIR = Path("results/interaction_visualisation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["throughput", "median_nttft", "median_itl"]

AI_COLS = [
    "model_n_parameters","model_n_layers","model_n_heads","model_n_positions",
    "model_vocabulary_size","model_relative_attention_max_distance",
    "model_relative_attention_n_buckets","model_is_flash_attention",
    "model_is_encoder_decoder","model_type_codegen","model_type_gpt_bigcode",
    "model_type_gpt_neox","model_type_llama","model_type_mpt","model_type_mt5",
    "model_type_t5","model_torch_dtype_bfloat16","model_torch_dtype_float16",
    "model_torch_dtype_float32",
]

NONAI_COLS = [
    "gpu_n_cuda_cores","gpu_n_tensor_cores","gpu_n_rt_cores","gpu_n_sms",
    "gpu_n_rops","gpu_n_tmus","gpu_tflops_cuda_fp32","gpu_tflops_cuda_fp64",
    "gpu_tflops_cuda_mixed","gpu_tflops_tc_fp16","gpu_tflops_tc_bf16",
    "gpu_tflops_tc_fp32","gpu_tflops_tc_tf32","gpu_tflops_tc_fp64",
    "gpu_memory_capacity_gb","gpu_memory_capacity_gb_total",
    "gpu_memory_bandwidth","gpu_system_interface_gen","gpu_compute_capability",
    "gpu_is_sxm","gpu_is_nvlink","gpu_n_gpus","gpu_architecture_Ampere",
    "gpu_architecture_Hopper","gpu_architecture_Turing","gpu_architecture_Volta",
]

WORKLOAD_COLS = ["num_users"]

# --------------------------
# Helper
# --------------------------
def compute_block_strength(df, cols, block_name: str):
    """
    Compute a simple norm-based strength for a block.

    - Keeps only columns that actually exist.
    - Coerces everything to numeric.
    - Fills NaNs with 0 so norm is well-defined.
    """
    # Keep only columns that exist in df
    existing = [c for c in cols if c in df.columns]
    if not existing:
        raise ValueError(f"No columns from {block_name} found in dataframe.")

    X_block = df[existing].copy()

    # Convert to numeric (bool → 0/1, strings → numeric if possible, else NaN)
    X_block = X_block.apply(pd.to_numeric, errors="coerce")

    # Replace NaNs with 0 for the purpose of computing magnitude
    X_block = X_block.fillna(0.0)

    # Finally ensure float64 array
    X_arr = X_block.to_numpy(dtype=float)

    return np.linalg.norm(X_arr, axis=1)

# --------------------------
# MAIN
# --------------------------
print("[INFO] Loading dataset ...")
df = pd.read_csv(DATA_PATH, index_col=0)

# Compute block strength signals
df["ai_strength"] = compute_block_strength(df, AI_COLS, "AI_COLS")
df["nonai_strength"] = compute_block_strength(df, NONAI_COLS, "NONAI_COLS")
df["wl_strength"] = compute_block_strength(df, WORKLOAD_COLS, "WORKLOAD_COLS")


# Pairwise interactions (magnitude only)
A = df[AI_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
N = df[NONAI_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
W = df[WORKLOAD_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

df["aixn_strength"] = np.linalg.norm(A[:, None, :] * N[:, :, None], axis=(1,2))
df["aixw_strength"] = np.linalg.norm(A[:, None, :] * W[:, :, None], axis=(1,2))
df["nixw_strength"] = np.linalg.norm(N[:, None, :] * W[:, :, None], axis=(1,2))

# --------------------------
# Run embeddings per target
# --------------------------
for target in TARGETS:
    print(f"\n[INFO] Processing target = {target}")

    # X contains ALL features (AI+NonAI+WL)
    X = df[AI_COLS + NONAI_COLS + WORKLOAD_COLS]
    y = df[target]

    # Scale features
    X_scaled = StandardScaler().fit_transform(X)

    # -------- UMAP --------
    um = umap.UMAP(
        n_neighbors=30,
        min_dist=0.01,
        metric="euclidean",
        random_state=42,
    )
    emb_umap = um.fit_transform(X_scaled)
    df["umap_x"] = emb_umap[:, 0]
    df["umap_y"] = emb_umap[:, 1]

    # Save embedding
    df[["umap_x", "umap_y"]].to_csv(OUT_DIR / f"umap_embedding_{target}.csv", index=False)

    # -------- t-SNE --------
    perplexity = min(40, max(5, len(df)//10))
    ts = TSNE(
                n_components=2,
                max_iter=1000,              # replaces n_iter
                perplexity=40,
                learning_rate="auto",
                init="pca",
                n_iter_without_progress=300,
                random_state=42,
            )

    emb_tsne = ts.fit_transform(X_scaled)
    df["tsne_x"] = emb_tsne[:, 0]
    df["tsne_y"] = emb_tsne[:, 1]

    df[["tsne_x", "tsne_y"]].to_csv(OUT_DIR / f"tsne_embedding_{target}.csv", index=False)

    # -------- Plotting utility --------
    def plot_embed(method, xcol, ycol, color_col):
        plt.figure(figsize=(7,5))
        plt.scatter(df[xcol], df[ycol], c=df[color_col], cmap="viridis",
                    s=22, alpha=0.8, edgecolors="none")
        plt.colorbar(label=color_col)
        plt.title(f"{method} coloured by {color_col}  (target={target})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{method}_{color_col}_{target}.png", dpi=300)
        plt.close()

    # Plot block strengths + interaction strengths
    all_strength_cols = [
        "ai_strength","nonai_strength","wl_strength",
        "aixn_strength","aixw_strength","nixw_strength"
    ]

    for c in all_strength_cols:
        plot_embed("umap", "umap_x", "umap_y", c)
        plot_embed("tsne", "tsne_x", "tsne_y", c)

print("\n[INFO] All embeddings & plots generated successfully.")
print(f"Results saved in: {OUT_DIR.resolve()}")

