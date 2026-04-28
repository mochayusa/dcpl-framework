from __future__ import annotations

from pathlib import Path
import pandas as pd

from dcpl.blocks import (
    BlocksConfig,
    prepare_dataframe_for_blocks,
    get_blocks_relaxed,
)
from dcpl.interactions import build_all_interactions
from utils.io import make_run_dir, save_manifest
from utils.logging import get_logger


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if data_path.suffix.lower() == ".csv":
        return pd.read_csv(data_path, index_col=0)
    if data_path.suffix.lower() == ".parquet":
        return pd.read_parquet(data_path)
    raise ValueError(f"Unsupported dataset format: {data_path}")


def prepare_run(
    df: pd.DataFrame,
    results_root: str | Path = "results/runs",
    run_name: str = "run",
    include_interactions: bool = False,
    schema: str | Path | None = None,
):
    """
    Creates run directory, extracts blocks, optionally builds interaction blocks,
    and returns (run_dir, logger, X_ai, X_nonai, X_wl, interactions or None).

    IMPORTANT:
    - Blocks use relaxed + alias/encoding support.
    - Interactions must be built from the *prepared* dataframe that contains
      the canonical column names (e.g., model_n_parameters), otherwise merged
      datasets may crash with KeyError.
    """
    logger = get_logger("dcpl")

    run_dir = make_run_dir(results_root)
    logger.info(f"Run directory created: {run_dir}")

    # ---- 1) Build blocks (already relaxed + alias/one-hot + numeric coercion)
    X_ai, X_nonai, X_wl = get_blocks_relaxed(df, schema=schema)

    # ---- 2) Build interactions from prepared dataframe (canonical cols exist)
    interactions = None
    if include_interactions:
        cfg = BlocksConfig(
            strict=False,
            encode_categoricals=True,
            coerce_numeric=True,
            fillna_strategy="median",
        )
        df_prepared = prepare_dataframe_for_blocks(df, cfg, schema=schema)

        # Sanity (optional): ensures the interaction builder won't KeyError
        # If something is missing here, it means alias map needs extension.
        missing_ai = [c for c in X_ai.columns if c not in df_prepared.columns]
        missing_ni = [c for c in X_nonai.columns if c not in df_prepared.columns]
        missing_wl = [c for c in X_wl.columns if c not in df_prepared.columns]
        if missing_ai or missing_ni or missing_wl:
            logger.warning(
                "Prepared df is missing some canonical columns referenced by blocks. "
                f"missing_ai={missing_ai[:5]} missing_nonai={missing_ni[:5]} missing_wl={missing_wl[:5]}"
            )

        interactions = build_all_interactions(
            df_prepared,
            list(X_ai.columns),
            list(X_nonai.columns),
            list(X_wl.columns),
        )

    # ---- manifest skeleton (caller can extend)
    manifest = {
        "run_name": run_name,
        "data_shape": list(df.shape),
        "n_ai": int(X_ai.shape[1]),
        "n_nonai": int(X_nonai.shape[1]),
        "n_workload": int(X_wl.shape[1]),
        "include_interactions": include_interactions,
        "interaction_shapes": {k: list(v.shape) for k, v in interactions.items()} if interactions else None,
    }
    save_manifest(run_dir, manifest)

    return run_dir, logger, X_ai, X_nonai, X_wl, interactions
