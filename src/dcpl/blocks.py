from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .schema import DatasetSchema, load_schema


DEFAULT_SCHEMA = load_schema()


# ============================================================
# Targets (Y)
# ============================================================
PERF_COLS = DEFAULT_SCHEMA.targets


# ============================================================
# Curated block columns (X) – canonical names
# ============================================================
AI_COLS = DEFAULT_SCHEMA.block_columns("ai")
NONAI_COLS = DEFAULT_SCHEMA.block_columns("nonai")
WORKLOAD_COLS = DEFAULT_SCHEMA.block_columns("workload")
ALIAS_MAP = DEFAULT_SCHEMA.alias_map()


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


def _resolve_schema(schema: DatasetSchema | str | None = None) -> DatasetSchema:
    if isinstance(schema, DatasetSchema):
        return schema
    return load_schema(schema)


def _apply_aliases(df: pd.DataFrame, alias_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Copy alternative columns into canonical names if canonical is missing.
    Does not delete original columns.
    """
    out = df.copy()
    for canonical, alternatives in alias_map.items():
        if canonical in out.columns:
            continue
        for alt in alternatives:
            if alt in out.columns:
                out[canonical] = out[alt]
                break
    return out


def _normalize_category_token(value: object) -> str:
    token = str(value).strip().lower()
    token = token.replace("torch.", "")
    token = token.replace("-", "_").replace(" ", "_")
    return token


def _one_hot_configured_categoricals(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """
    Create one-hot columns for any source column listed in schema.categorical_levels.
    Output column names follow <source_column>_<raw_category>.
    """
    out = df.copy()

    for source_col, categories in schema.categorical_levels.items():
        if source_col not in out.columns:
            continue
        series = out[source_col].map(_normalize_category_token)
        for cat in categories:
            out[f"{source_col}_{cat}"] = (series == _normalize_category_token(cat)).astype(int)
    return out


def validate_columns(
    df: pd.DataFrame,
    cols: List[str],
    name: str,
    strict: bool = True,
    min_required: int = 1,
) -> List[str]:
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

    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(int)

    out = out.apply(pd.to_numeric, errors="coerce")

    if fillna_strategy == "zero":
        out = out.fillna(0.0)
    else:
        med = out.median(numeric_only=True)
        out = out.fillna(med)

    return out


def prepare_dataframe_for_blocks(
    df: pd.DataFrame,
    cfg: BlocksConfig,
    schema: DatasetSchema | str | None = None,
) -> pd.DataFrame:
    """
    Apply schema adaptation steps so get_blocks can work across:
      - original enriched dataset (canonical cols already present)
      - per-model dataset (AI_model_* etc)
      - custom datasets described by YAML schemas
    """
    resolved = _resolve_schema(schema)
    out = _apply_aliases(df, resolved.alias_map())

    if cfg.encode_categoricals:
        out = _one_hot_configured_categoricals(out, resolved)

    return out


def get_blocks(
    df: pd.DataFrame,
    cfg: Optional[BlocksConfig] = None,
    schema: DatasetSchema | str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (X_ai, X_nonai, X_workload).

    - By default: strict=True to preserve reproducibility for the main pipeline.
    - For per-model split runs: call with BlocksConfig(strict=False) or use get_blocks_relaxed().
    """
    cfg = cfg or BlocksConfig()
    resolved = _resolve_schema(schema)

    df2 = prepare_dataframe_for_blocks(df, cfg, resolved)

    ai_cols = validate_columns(
        df2,
        resolved.block_columns("ai"),
        "AI block",
        strict=cfg.strict,
        min_required=cfg.min_ai,
    )
    nonai_cols = validate_columns(
        df2,
        resolved.block_columns("nonai"),
        "Non-AI block",
        strict=cfg.strict,
        min_required=cfg.min_nonai,
    )
    wl_cols = validate_columns(
        df2,
        resolved.block_columns("workload"),
        "Workload block",
        strict=cfg.strict,
        min_required=cfg.min_workload,
    )

    X_ai = df2[ai_cols].copy()
    X_nonai = df2[nonai_cols].copy()
    X_wl = df2[wl_cols].copy()

    if cfg.coerce_numeric:
        X_ai = _coerce_and_fill(X_ai, fillna_strategy=cfg.fillna_strategy)
        X_nonai = _coerce_and_fill(X_nonai, fillna_strategy=cfg.fillna_strategy)
        X_wl = _coerce_and_fill(X_wl, fillna_strategy=cfg.fillna_strategy)
    else:
        for X, name in [(X_ai, "AI"), (X_nonai, "Non-AI"), (X_wl, "Workload")]:
            bad = [c for c in X.columns if X[c].dtype == "object"]
            if bad:
                raise ValueError(f"Object dtype columns found in {name} block: {bad}")

    return X_ai, X_nonai, X_wl


def get_blocks_relaxed(
    df: pd.DataFrame,
    schema: DatasetSchema | str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Recommended for per-model split datasets that may not have all curated columns.
    """
    return get_blocks(
        df,
        cfg=BlocksConfig(strict=False, encode_categoricals=True, coerce_numeric=True),
        schema=schema,
    )
