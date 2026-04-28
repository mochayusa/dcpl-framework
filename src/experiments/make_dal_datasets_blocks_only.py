from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

# Your enriched LLM-pilot dataset
DATA_PATH = Path("data/llm_pilot_data/final_data/historical_performance_data_enriched_final.csv")

# Where to save DaL datasets
OUT_DIR = Path("data/dal_datasets")

# Targets you want to train DaL on (one per dataset)
DAL_TARGETS = ["throughput", "median_nttft", "median_itl"]

# --- Feature blocks (as you specified) ---

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
    # Keep *_list out; DaL expects simple numeric features
]

FEATURE_COLS = AI_COLS + NONAI_COLS + WORKLOAD_COLS


def validate_columns(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")
    return cols


def main():
    # ----------------------------------------------------------
    # 1) Load data
    # ----------------------------------------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input data not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, index_col=0)
    print(f"[INFO] Loaded data: {DATA_PATH} | shape = {df.shape}")

    # ----------------------------------------------------------
    # 2) Validate feature and target columns
    # ----------------------------------------------------------
    validate_columns(df, AI_COLS, "AI_COLS")
    validate_columns(df, NONAI_COLS, "NONAI_COLS")
    validate_columns(df, WORKLOAD_COLS, "WORKLOAD_COLS")
    validate_columns(df, DAL_TARGETS, "DAL_TARGETS")

    # Ensure feature types are numeric/bool (DaL expects numeric)
    non_numeric = [c for c in FEATURE_COLS if df[c].dtype == "object"]
    if non_numeric:
        raise ValueError(f"These feature columns are object dtype; "
                         f"please encode/convert them first: {non_numeric}")

    # Convert bool → int (0/1) for safety
    bool_cols = [c for c in FEATURE_COLS if df[c].dtype == "bool"]
    if bool_cols:
        print(f"[INFO] Converting bool → int for: {bool_cols}")
        df[bool_cols] = df[bool_cols].astype(int)

    # Create output folder
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 3) Build and save one dataset per target
    # ----------------------------------------------------------
    for target in DAL_TARGETS:
        cols_order = FEATURE_COLS + [target]

        df_target = df[cols_order].copy()

        # Drop rows with missing values in features or target
        before = len(df_target)
        df_target = df_target.dropna(axis=0, how="any")
        dropped = before - len(df_target)

        print(
            f"[INFO] Target '{target}': shape before={before}, after_dropna={len(df_target)}, "
            f"dropped={dropped}"
        )

        # Save DaL-style CSV: all features first, target last
        out_path = OUT_DIR / f"dal_blocks_{target}.csv"
        df_target.to_csv(out_path, index=False)
        print(f"[INFO] Saved DaL dataset for target '{target}': {out_path}")

    print("[INFO] Done. All DaL datasets are in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
