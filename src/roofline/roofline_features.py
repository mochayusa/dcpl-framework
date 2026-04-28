# src/roofline/roofline_features.py

from __future__ import annotations

import numpy as np
import pandas as pd

from roofline.constants import (
    COL_II, COL_OO, COL_BB,
    COL_BW, COL_PARAMS,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        "None of the candidate columns exist.\n"
        f"Tried: {candidates}\n"
        f"Available: {list(df.columns)}"
    )

def _params_to_float(x) -> float:
    """
    Convert parameter counts to float.

    Accepts:
      - numeric values (already counts)
      - strings like '1.3B', '350M', '125m', '7b'
      - strings with commas '7,000,000,000'
    Returns np.nan on failure.
    """
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip().upper().replace(",", "")
    if s == "" or s in {"NAN", "NONE"}:
        return np.nan

    mult = 1.0
    if s.endswith("B"):
        mult = 1e9
        s = s[:-1]
    elif s.endswith("M"):
        mult = 1e6
        s = s[:-1]
    elif s.endswith("K"):
        mult = 1e3
        s = s[:-1]

    try:
        return float(s) * mult
    except Exception:
        return np.nan


# ------------------------------------------------------------
# Roofline proxy features
# ------------------------------------------------------------

def add_roofline_proxy_features(
    df: pd.DataFrame,
    *,
    # Memory: assume weights dominate and stored in fp16 by default
    bytes_per_param: float = 2.0,
    # Compute proxy: scalar multiplier; calibrating LR absorbs scaling anyway
    ops_per_param_token: float = 2.0,
    # Optional: clip pathological values
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Add proxy Roofline features for LLM inference throughput modeling.

    Roofline (conceptual):
      AI = OPs / memory_access
      perf_ops_s = min( AI * bandwidth_bytes_s, peak_ops_s )
      time_s = OPs / perf_ops_s
      throughput_tokens_s ~ tokens / time_s

    In your dataset, we lack exact OPs and memory traffic. We therefore use proxies:
      - OPs ~ ops_per_param_token * params * seq
      - memory_access ~ params * bytes_per_param

    Where:
      - params from AI_model_n_parameters
      - seq from Workload_sequence_length if present, otherwise (ii + oo)

    Requires columns:
      - COL_II, COL_OO
      - COL_BW  (NonAI_gpu_memory_bandwidth)
      - AI_model_n_parameters
      - A peak TFLOPS column (chosen automatically from common candidates)

    Returns df copy with added columns:
      - roof_peak_tflops_col
      - roof_bw_bytes_s
      - roof_peak_ops_s
      - roof_ops
      - roof_mem_bytes
      - roof_ai
      - roof_perf_ops_s
      - roof_time_s
      - roof_thr_tokens_s
    """
    out = df.copy()

    # Required columns
    required = [COL_II, COL_OO, COL_BW, COL_PARAMS]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Missing required columns for roofline features: {missing}")

    # Pick best available peak compute column
    peak_tflops_col = pick_first_existing(out, [
        "NonAI_gpu_tflops_tc_fp16",
        "NonAI_gpu_tflops_tc_bf16",
        "NonAI_gpu_tflops_tc_tf32",
        "NonAI_gpu_tflops_cuda_mixed",
        "NonAI_gpu_tflops_cuda_fp32",
        "NonAI_gpu_tflops_cuda_fp64",
    ])
    out["roof_peak_tflops_col"] = peak_tflops_col

    # Coerce numeric inputs
    ii = _to_float_series(out[COL_II])
    oo = _to_float_series(out[COL_OO])

    bw_gbs = _to_float_series(out[COL_BW])
    # Assume GB/s-ish; convert to bytes/s (GiB-based conversion).
    # If your column is already bytes/s, remove the multiply by (1024**3).
    bw_bytes_s = bw_gbs * (1024**3)

    tflops = _to_float_series(out[peak_tflops_col])
    peak_ops_s = tflops * 1e12  # TFLOPS -> ops/s

    # Params
    params = out[COL_PARAMS].map(_params_to_float).astype(float)

    # Token proxies
    total_tokens = (ii + oo).clip(lower=1.0)

    if "Workload_sequence_length" in out.columns:
        seq = _to_float_series(out["Workload_sequence_length"])
        # fall back to total_tokens if seq missing
        seq = seq.fillna(total_tokens)
    else:
        seq = total_tokens

    seq = seq.clip(lower=1.0)

    # Proxy OPs and memory traffic
    ops = ops_per_param_token * params * seq
    mem_bytes = (params * bytes_per_param)

    # Avoid divide-by-zero / NaNs
    ops = ops.replace([np.inf, -np.inf], np.nan) if isinstance(ops, pd.Series) else ops
    mem_bytes = mem_bytes.replace([np.inf, -np.inf], np.nan) if isinstance(mem_bytes, pd.Series) else mem_bytes

    # Compute AI and perf
    mem_safe = np.maximum(mem_bytes, eps)
    ai = ops / mem_safe

    bw_safe = np.maximum(bw_bytes_s, eps)
    peak_safe = np.maximum(peak_ops_s, eps)

    perf_ops_s = np.minimum(ai * bw_safe, peak_safe)

    # Time proxy
    perf_safe = np.maximum(perf_ops_s, eps)
    time_s = ops / perf_safe

    # Throughput proxy: tokens/time
    thr_tokens_s = total_tokens / np.maximum(time_s, eps)

    # Attach features
    out["roof_bw_bytes_s"] = bw_bytes_s
    out["roof_peak_ops_s"] = peak_ops_s
    out["roof_ops"] = ops
    out["roof_mem_bytes"] = mem_bytes
    out["roof_ai"] = ai
    out["roof_perf_ops_s"] = perf_ops_s
    out["roof_time_s"] = time_s
    out["roof_thr_tokens_s"] = thr_tokens_s

    return out
