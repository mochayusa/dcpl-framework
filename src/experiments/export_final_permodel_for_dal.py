from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

from dcpl.blocks import get_blocks_relaxed


def ensure_target_exists(df: pd.DataFrame, target: str, csv_path: Path) -> None:
    if target not in df.columns:
        raise ValueError(f"[{csv_path.name}] target column not found: '{target}'")


def build_xy(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Build predictor matrix X using blocks (AI+NonAI+Workload),
    then append y (target) as the last column.
    """
    X_ai, X_nonai, X_wl = get_blocks_relaxed(df)

    # Concatenate predictors
    X = pd.concat([X_ai, X_nonai, X_wl], axis=1)

    # Target as numeric (DaL regression expects numeric y)
    y = pd.to_numeric(df[target], errors="coerce")

    # Drop rows where y is NaN (optional but recommended)
    keep = ~y.isna()
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    out = pd.concat([X, y.rename(target)], axis=1)

    # OPTIONAL: sort columns so deterministic + put target at last
    cols = [c for c in out.columns if c != target] + [target]
    out = out[cols]

    return out


def export_per_model(
    per_model_dir: Path,
    out_dir: Path,
    target: str,
    pattern: str = "*.csv",
    keep_original_name: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(per_model_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in: {per_model_dir} with pattern={pattern}")

    summary_rows = []

    for csv_path in files:
        model_tag = csv_path.stem  # e.g., data_EleutherAI_gpt-neox-20b
        print(f"[LOAD] {csv_path}")

        df = pd.read_csv(csv_path)

        ensure_target_exists(df, target, csv_path)

        final_df = build_xy(df, target=target)

        # output file name
        if keep_original_name:
            out_path = out_dir / f"{model_tag}__final_for_dal.csv"
        else:
            out_path = out_dir / f"{model_tag}.csv"

        final_df.to_csv(out_path, index=False)
        print(f"[SAVE] {out_path}  (rows={len(final_df)}, cols={final_df.shape[1]})")

        summary_rows.append({
            "per_model_file": model_tag,
            "in_path": str(csv_path),
            "out_path": str(out_path),
            "rows": int(len(final_df)),
            "n_features": int(final_df.shape[1] - 1),
            "target": target,
        })

    # Write an index/manifest CSV for convenience
    summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "export_manifest.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[DONE] Manifest saved: {summary_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export final per-model datasets (X... + target last) for DaL.")
    p.add_argument("--per-model-dir", type=str, required=True, help="Directory containing per-model CSVs.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for final per-model CSVs.")
    p.add_argument("--target", type=str, default="Target_throughput_tokens_per_sec", help="Target column name.")
    p.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for input files.")
    p.add_argument("--keep-original-name", action="store_true", help="If set, keep original stem + suffix.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_per_model(
        per_model_dir=Path(args.per_model_dir),
        out_dir=Path(args.out_dir),
        target=args.target,
        pattern=args.pattern,
        keep_original_name=bool(args.keep_original_name),
    )


if __name__ == "__main__":
    main()
