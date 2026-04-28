from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd


def build_interaction_block(df: pd.DataFrame, cols_a, cols_b, prefix: str) -> pd.DataFrame:
    """
    Pairwise interactions: a_i * b_j for all pairs (cols_a x cols_b).

    Robust:
      - Skips any columns not present in df (important for merged/global datasets).
      - Coerces values to numeric (non-numeric -> NaN).
    """
    cols_a = [c for c in cols_a if c in df.columns]
    cols_b = [c for c in cols_b if c in df.columns]

    if len(cols_a) == 0 or len(cols_b) == 0:
        return pd.DataFrame(index=df.index)

    out = {}
    for a in cols_a:
        a_vals = pd.to_numeric(df[a], errors="coerce").to_numpy()
        for b in cols_b:
            b_vals = pd.to_numeric(df[b], errors="coerce").to_numpy()
            out[f"{prefix}_{a}__x__{b}"] = a_vals * b_vals

    return pd.DataFrame(out, index=df.index)


def build_all_interactions(
    df: pd.DataFrame,
    ai_cols: List[str],
    nonai_cols: List[str],
    wl_cols: List[str],
    include: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Returns interaction blocks based on the 'include' list.

    include options:
      - [] -> no interactions (all empty)
      - ["AIxNonAI"]
      - ["AIxNonAI","AIxWorkload"]
      - ["AIxNonAI","AIxWorkload","NonAIxWorkload"] (default)
    """
    if include is None:
        include = ["AIxNonAI", "AIxWorkload", "NonAIxWorkload"]

    blocks: Dict[str, pd.DataFrame] = {}

    # AI × NonAI
    if "AIxNonAI" in include:
        blocks["AIxNonAI"] = build_interaction_block(df, ai_cols, nonai_cols, "AIxNonAI")
    else:
        blocks["AIxNonAI"] = pd.DataFrame(index=df.index)

    # AI × Workload
    if "AIxWorkload" in include:
        blocks["AIxWorkload"] = build_interaction_block(df, ai_cols, wl_cols, "AIxWorkload")
    else:
        blocks["AIxWorkload"] = pd.DataFrame(index=df.index)

    # NonAI × Workload
    if "NonAIxWorkload" in include:
        blocks["NonAIxWorkload"] = build_interaction_block(df, nonai_cols, wl_cols, "NonAIxWorkload")
    else:
        blocks["NonAIxWorkload"] = pd.DataFrame(index=df.index)

    return blocks
