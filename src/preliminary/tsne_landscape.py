from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


DATA = "data/llm_pilot_data/preliminary/preliminary_sample_100k.csv"
OUTPUT_DIR = "results/preliminary/tsne"

TARGET_COL = "Target_throughput_tokens_per_sec"

AI_COLS = [
    "AI_model_n_parameters",
    "AI_model_n_layers",
    "AI_model_n_heads",
    "AI_model_n_positions",
    "AI_model_vocabulary_size",
    "AI_model_relative_attention_max_distance",
    "AI_model_relative_attention_n_buckets",
    "AI_model_is_flash_attention",
    "AI_model_is_encoder_decoder",
]

NONAI_COLS = [
    "NonAI_gpu_n_cuda_cores",
    "NonAI_gpu_n_tensor_cores",
    "NonAI_gpu_n_rt_cores",
    "NonAI_gpu_n_sms",
    "NonAI_gpu_n_rops",
    "NonAI_gpu_n_tmus",
    "NonAI_gpu_tflops_cuda_fp32",
    "NonAI_gpu_tflops_cuda_fp64",
    "NonAI_gpu_tflops_cuda_mixed",
    "NonAI_gpu_tflops_tc_fp16",
    "NonAI_gpu_tflops_tc_bf16",
    "NonAI_gpu_tflops_tc_fp32",
    "NonAI_gpu_tflops_tc_tf32",
    "NonAI_gpu_tflops_tc_fp64",
    "NonAI_gpu_memory_capacity_gb",
    "NonAI_gpu_memory_capacity_gb_total",
    "NonAI_gpu_memory_bandwidth",
    "NonAI_gpu_system_interface_gen",
    "NonAI_gpu_compute_capability",
    "NonAI_gpu_is_sxm",
    "NonAI_gpu_is_nvlink",
    "NonAI_gpu_n_gpus",
]

WORKLOAD_COLS = [
    "Workload_n_input_tokens",
    "Workload_reqnum",
]

OPTIONAL_CATEGORICALS = [
    "AI_model_type",
    "AI_model_torch_dtype",
    "NonAI_gpu_architecture",
    "NonAI_gpu_type",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = []

    for col in AI_COLS + NONAI_COLS + WORKLOAD_COLS:
        if col in df.columns:
            feature_cols.append(col)

    base = df[feature_cols].copy()

    for col in feature_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    cat_present = [c for c in OPTIONAL_CATEGORICALS if c in df.columns]
    if cat_present:
        dummies = pd.get_dummies(df[cat_present], prefix=cat_present, dummy_na=False)
        base = pd.concat([base, dummies], axis=1)

    base = base.replace([np.inf, -np.inf], np.nan)
    base = base.dropna(axis=0)

    return base, list(base.columns)


def run_tsne(
    X: np.ndarray,
    random_state: int = 42,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: str | float = "auto",
    max_iter: int = 2000,
) -> TSNE:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        max_iter=max_iter,
        init="pca",
        random_state=random_state,
        metric="euclidean",
        verbose=1,
    )
    embedding = tsne.fit_transform(X)
    return tsne, embedding


def make_plot(df_plot: pd.DataFrame, output_png: Path, target_col: str) -> None:
    plt.figure(figsize=(10, 8))

    color_values = np.log10(df_plot[target_col].clip(lower=1e-9))

    sc = plt.scatter(
        df_plot["tsne_1"],
        df_plot["tsne_2"],
        c=color_values,
        s=6,
        alpha=0.65,
        linewidths=0,
    )

    cbar = plt.colorbar(sc)
    cbar.set_label(r"$\log_{10}(\mathrm{Target\_throughput\_tokens\_per\_sec})$")

    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title("Two-dimensional t-SNE projection of LLM inference configuration space")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DATA)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--target", type=str, default=TARGET_COL)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--early-exaggeration", type=float, default=12.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("Loading data...")
    df = load_data(args.input)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found.")

    print("Building feature matrix...")
    X_df, used_features = build_feature_matrix(df)

    # keep only aligned rows
    df_aligned = df.loc[X_df.index].copy()
    df_aligned = df_aligned.dropna(subset=[args.target]).copy()
    X_df = X_df.loc[df_aligned.index].copy()

    print(f"Rows used for t-SNE: {len(X_df):,}")
    print(f"Number of features: {len(used_features)}")

    print("Standardising features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    print("Running t-SNE...")
    tsne_model, embedding = run_tsne(
        X_scaled,
        random_state=args.random_state,
        perplexity=args.perplexity,
        early_exaggeration=args.early_exaggeration,
        max_iter=args.max_iter,
    )

    df_plot = df_aligned.copy()
    df_plot["tsne_1"] = embedding[:, 0]
    df_plot["tsne_2"] = embedding[:, 1]

    csv_path = output_dir / "tsne_embedding.csv"
    png_path = output_dir / "tsne_throughput_landscape.png"
    meta_path = output_dir / "tsne_metadata.json"

    print("Saving outputs...")
    df_plot.to_csv(csv_path, index=False)
    make_plot(df_plot, png_path, args.target)

    meta = {
        "input": args.input,
        "rows_used": int(len(df_plot)),
        "target_col": args.target,
        "n_features": int(len(used_features)),
        "used_features": used_features,
        "perplexity": args.perplexity,
        "early_exaggeration": args.early_exaggeration,
        "max_iter": args.max_iter,
        "random_state": args.random_state,
        "kl_divergence_": float(tsne_model.kl_divergence_),
        "output_csv": str(csv_path),
        "output_png": str(png_path),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Figure saved to: {png_path}")
    print(f"Embedding CSV saved to: {csv_path}")
    print(f"KL divergence: {tsne_model.kl_divergence_:.6f}")


if __name__ == "__main__":
    main()