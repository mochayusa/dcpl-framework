from __future__ import annotations
from typing import List, Tuple
import pandas as pd


# ----------------------------
# Targets (Y)
# ----------------------------
PERF_COLS = [
    "throughput",
    "median_ttft",
    "median_nttft",
    "median_itl",
    "cost",
    "throughput/$",
]


# ----------------------------
# Curated block columns (X)
# Use your explicit lists for reproducibility
# ----------------------------
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
    # NOTE: keep *_list columns out of regression-ready X
    # you can add workload aggregates like:
    # "workload_n_input_tokens_list_mean", ...
]


def validate_columns(df: pd.DataFrame, cols: List[str], name: str) -> List[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")
    return cols


def get_blocks(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns X_ai, X_nonai, X_workload using curated lists.
    Enforces numeric (no object dtype) for each block.
    """
    ai_cols = validate_columns(df, AI_COLS, "AI block")
    nonai_cols = validate_columns(df, NONAI_COLS, "Non-AI block")
    wl_cols = validate_columns(df, WORKLOAD_COLS, "Workload block")

    X_ai = df[ai_cols].copy()
    X_nonai = df[nonai_cols].copy()
    X_wl = df[wl_cols].copy()

    for X, name in [(X_ai, "AI"), (X_nonai, "Non-AI"), (X_wl, "Workload")]:
        bad = [c for c in X.columns if X[c].dtype == "object"]
        if bad:
            raise ValueError(f"Object dtype columns found in {name} block: {bad}")

    return X_ai, X_nonai, X_wl
