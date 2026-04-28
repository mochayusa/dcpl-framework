from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


def make_run_dir(results_root: str | Path = "results/runs", run_id: Optional[str] = None) -> Path:
    """
    Creates a unique run directory under results_root.

    Default ID is timestamp-based (YYYY-mm-dd_HHMMSS). If a directory already exists
    (e.g., multiple runs started within the same second), a suffix _1, _2, ... is appended.

    Example:
      results/runs/2026-01-13_104434/
      results/runs/2026-01-13_104434_1/
    """
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    base_id = run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = results_root / base_id

    # guarantee uniqueness
    if run_dir.exists():
        k = 1
        while (results_root / f"{base_id}_{k}").exists():
            k += 1
        run_dir = results_root / f"{base_id}_{k}"

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    return run_dir


def _atomic_write_csv(df: pd.DataFrame, out: Path, index: bool = False) -> None:
    """
    Write CSV atomically (write temp file then replace).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    os.replace(tmp, out)


def _atomic_write_json(obj: Any, out: Path) -> None:
    """
    Write JSON atomically (write temp file then replace).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, out)


def save_predictions(
    run_dir: Path,
    experiment: str,
    model: str,
    cv: str,
    target: str,
    df: pd.DataFrame,
) -> Path:
    """
    Saves prediction CSV in:
      predictions/{experiment}_{model}_{cv}_{target}.csv
    """
    out = run_dir / "predictions" / f"{experiment}_{model}_{cv}_{target}.csv"
    _atomic_write_csv(df, out, index=False)
    return out


def save_summary(run_dir: Path, rows: List[Dict[str, Any]], append: bool = False) -> Path:
    """
    Saves summary.csv in the run directory.

    - append=False (default): overwrite summary.csv with the provided rows.
    - append=True           : append rows to existing summary.csv if present.

    This is useful when a single run directory contains multiple models/targets.
    """
    out = run_dir / "summary.csv"
    new_df = pd.DataFrame(rows)

    if append and out.exists():
        old_df = pd.read_csv(out)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    _atomic_write_csv(df, out, index=False)
    return out


def save_manifest(run_dir: Path, meta: Dict[str, Any], merge: bool = True) -> Path:
    """
    Saves experiment metadata to manifest.json.

    - merge=True (default): if manifest.json exists, merge keys (new keys override old).
    - merge=False         : overwrite entirely.
    """
    out = run_dir / "manifest.json"

    if merge and out.exists():
        try:
            old = json.loads(out.read_text())
            if isinstance(old, dict):
                merged = {**old, **meta}
            else:
                merged = meta
        except Exception:
            merged = meta
    else:
        merged = meta

    _atomic_write_json(merged, out)
    return out
