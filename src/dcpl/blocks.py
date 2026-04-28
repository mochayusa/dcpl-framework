from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import pandas as pd


# ============================================================
# Targets (Y)
# ============================================================
PERF_COLS = [
    "throughput",
    "cost",
    "throughput/$",
    # per-model / alternative naming
    "Target_throughput_tokens_per_sec",
    "Target_latency_ms",
    "Target_latency_ms_per_token",
]


# ============================================================
# Curated block columns (X) – canonical names
# ============================================================
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
    # one-hot model_type_*
    "model_type_codegen",
    "model_type_gpt_bigcode",
    "model_type_gpt_neox",
    "model_type_llama",
    "model_type_mpt",
    "model_type_mt5",
    "model_type_t5",
    # one-hot model_torch_dtype_*
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
    # "num_users",
    "n_input_tokens",
    # "n_output_tokens",
    # "smpnum",
    "reqnum",
    # "requests",



   
]
# 1) Alias mapping from alternative column names to canonical names
ALIAS_MAP: Dict[str, str] = {
    # ---- AI ----
    "model_n_parameters": "AI_model_n_parameters",
    "model_n_layers": "AI_model_n_layers",
    "model_n_heads": "AI_model_n_heads",
    "model_n_positions": "AI_model_n_positions",
    "model_vocabulary_size": "AI_model_vocabulary_size",
    "model_relative_attention_max_distance": "AI_model_relative_attention_max_distance",
    "model_relative_attention_n_buckets": "AI_model_relative_attention_n_buckets",
    "model_is_flash_attention": "AI_model_is_flash_attention",
    "model_is_encoder_decoder": "AI_model_is_encoder_decoder",
    
    "model_type": "AI_model_type",
    "model_torch_dtype": "AI_model_torch_dtype",
    "gpu_n_cuda_cores": "NonAI_gpu_n_cuda_cores",
    "gpu_n_tensor_cores": "NonAI_gpu_n_tensor_cores",
    "gpu_n_rt_cores": "NonAI_gpu_n_rt_cores",
    "gpu_architecture": "NonAI_gpu_architecture", 
    "gpu": "NonAI_gpu",
    "gpu_type": "NonAI_gpu_type",

    # ---- Workload ----
    # "num_users": "Workload_num_users",
    "n_input_tokens": "Workload_n_input_tokens",
    # "n_output_tokens": "Workload_n_output_tokens",
    # "smpnum": "Workload_smpnum",
    "reqnum": "Workload_reqnum",
    # "requests": "Workload_requests",
}

MODEL_TYPE_CATS = ["codegen", "gpt_bigcode", "gpt_neox", "llama", "mpt", "mt5", "t5"]
TORCH_DTYPE_CATS = ["bfloat16", "float16", "float32"]

GPU_ARCH_CATS = ["Ampere", "Hopper", "Turing", "Volta"]


@dataclass(frozen=True)
class BlocksConfig:
    """
    strict=True  : require all curated columns, raise if missing
    strict=False : use intersection; will not fail unless a block becomes empty
    """
    strict: bool = True
    min_ai: int = 1
    min_nonai: int = 1
    min_workload: int = 1

    # If True, attempt to create one-hot columns (model_type_*, model_torch_dtype_*, gpu_architecture_*)
    encode_categoricals: bool = True

    # If True, coerce block columns to numeric (booleans->0/1, strings->NaN->fill)
    coerce_numeric: bool = True

    # NaN fill strategy after coercion (median is robust for regression)
    fillna_strategy: str = "median"  # "median" or "zero"


def _apply_aliases(df: pd.DataFrame, alias_map: Dict[str, str]) -> pd.DataFrame:
    """
    Copy alternative columns into canonical names if canonical is missing.
    Does not delete original columns.
    """
    out = df.copy()
    for canonical, alt in alias_map.items():
        if canonical not in out.columns and alt in out.columns:
            out[canonical] = out[alt]
    return out


def _one_hot_ai_type_and_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create canonical one-hot columns required by AI_COLS from:
      - model_type (categorical)
      - model_torch_dtype (categorical)
    """
    out = df.copy()

    # model_type_* one-hot
    if "model_type" in out.columns:
        s = out["model_type"].astype(str).str.lower()
        # common normalisations
        s = s.str.replace("-", "_", regex=False).str.replace(" ", "_", regex=False)
        for cat in MODEL_TYPE_CATS:
            out[f"model_type_{cat}"] = (s == cat).astype(int)

    # model_torch_dtype_* one-hot
    if "model_torch_dtype" in out.columns:
        s = out["model_torch_dtype"].astype(str).str.lower()
        s = s.str.replace("torch.", "", regex=False).str.replace(" ", "", regex=False)
        for cat in TORCH_DTYPE_CATS:
            out[f"model_torch_dtype_{cat}"] = (s == cat).astype(int)

    return out


def _one_hot_gpu_architecture(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: create canonical gpu_architecture_* one-hot columns from a categorical 'gpu_architecture'
    """
    out = df.copy()
    if "gpu_architecture" in out.columns:
        s = out["gpu_architecture"].astype(str)
        for cat in GPU_ARCH_CATS:
            out[f"gpu_architecture_{cat}"] = (s == cat).astype(int)
    return out


def validate_columns(df: pd.DataFrame, cols: List[str], name: str, strict: bool = True, min_required: int = 1) -> List[str]:
    """
    strict=True  : require all columns to exist (raise if missing)
    strict=False : return intersection; raise only if <min_required columns remain
    """
    if strict:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")
        return cols

    present = [c for c in cols if c in df.columns]
    if len(present) < min_required:
        missing = [c for c in cols if c not in df.columns]
        raise ValueError(
            f"{name} has too few columns after intersection (present={len(present)}, required>={min_required}). "
            f"Missing examples: {missing[:20]}"
        )
    return present


def _coerce_and_fill(X: pd.DataFrame, *, fillna_strategy: str = "median") -> pd.DataFrame:
    """
    Convert columns to numeric safely. Booleans become 0/1.
    Objects are coerced to numeric; non-parsable becomes NaN then filled.
    """
    out = X.copy()

    # booleans -> 0/1
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(int)

    # coerce everything to numeric
    out = out.apply(pd.to_numeric, errors="coerce")

    if fillna_strategy == "zero":
        out = out.fillna(0.0)
    else:
        # median per column (robust)
        med = out.median(numeric_only=True)
        out = out.fillna(med)

    return out


def prepare_dataframe_for_blocks(df: pd.DataFrame, cfg: BlocksConfig) -> pd.DataFrame:
    """
    Apply schema adaptation steps so get_blocks can work across:
      - original enriched dataset (canonical cols already present)
      - per-model dataset (AI_model_* etc)
    """
    out = _apply_aliases(df, ALIAS_MAP)

    if cfg.encode_categoricals:
        out = _one_hot_ai_type_and_dtype(out)
        out = _one_hot_gpu_architecture(out)

    return out


def get_blocks(df: pd.DataFrame, cfg: Optional[BlocksConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (X_ai, X_nonai, X_workload).

    - By default: strict=True to preserve reproducibility for your main pipeline.
    - For per-model split runs: call with BlocksConfig(strict=False) or rely on alias+encoding.
    """
    cfg = cfg or BlocksConfig()

    df2 = prepare_dataframe_for_blocks(df, cfg)

    ai_cols = validate_columns(df2, AI_COLS, "AI block", strict=cfg.strict, min_required=cfg.min_ai)
    nonai_cols = validate_columns(df2, NONAI_COLS, "Non-AI block", strict=cfg.strict, min_required=cfg.min_nonai)
    wl_cols = validate_columns(df2, WORKLOAD_COLS, "Workload block", strict=cfg.strict, min_required=cfg.min_workload)

    X_ai = df2[ai_cols].copy()
    X_nonai = df2[nonai_cols].copy()
    X_wl = df2[wl_cols].copy()

    if cfg.coerce_numeric:
        X_ai = _coerce_and_fill(X_ai, fillna_strategy=cfg.fillna_strategy)
        X_nonai = _coerce_and_fill(X_nonai, fillna_strategy=cfg.fillna_strategy)
        X_wl = _coerce_and_fill(X_wl, fillna_strategy=cfg.fillna_strategy)
    else:
        # legacy strict check: no object dtypes
        for X, name in [(X_ai, "AI"), (X_nonai, "Non-AI"), (X_wl, "Workload")]:
            bad = [c for c in X.columns if X[c].dtype == "object"]
            if bad:
                raise ValueError(f"Object dtype columns found in {name} block: {bad}")

    return X_ai, X_nonai, X_wl


# Convenience: relaxed getter for per-model CSVs
def get_blocks_relaxed(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Recommended for per-model split datasets that may not have all curated columns.
    """
    return get_blocks(df, cfg=BlocksConfig(strict=False, encode_categoricals=True, coerce_numeric=True))
