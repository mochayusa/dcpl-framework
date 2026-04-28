from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


def merge_permodel_dir(per_model_dir: Path) -> pd.DataFrame:
    per_model_dir = Path(per_model_dir)
    files = sorted(per_model_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {per_model_dir}")

    dfs = []
    all_cols: set[str] = set()

    # First pass: read and collect all columns
    for p in files:
        df = pd.read_csv(p)
        df["per_model_file"] = p.stem
        dfs.append(df)
        all_cols |= set(df.columns)

    # Second pass: align columns (union) so concat never fails
    aligned = []
    all_cols = sorted(all_cols)
    for df in dfs:
        missing = [c for c in all_cols if c not in df.columns]
        for c in missing:
            df[c] = pd.NA
        aligned.append(df[all_cols])

    merged = pd.concat(aligned, ignore_index=True)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-model-dir", required=True, type=str,
                    help="Folder containing per-model CSV files.")
    ap.add_argument("--out", required=True, type=str,
                    help="Output merged CSV path (e.g., data/llm_pilot_data/raw_data/merged_all.csv).")
    args = ap.parse_args()

    per_model_dir = Path(args.per_model_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df_all = merge_permodel_dir(per_model_dir)
    df_all.to_csv(out, index=False)

    print(f"[OK] Merged shape: {df_all.shape}")
    print(f"[OK] Saved merged dataset to: {out}")
    print(df_all[["per_model_file"]].value_counts().head(10))


if __name__ == "__main__":
    main()

# how to run iterpython src/analysis/merge_permodel_to_global.py \
#   --per-model-dir data/llm_pilot_data/raw_data/per_model \
#   --out data/llm_pilot_data/raw_data/merged_all_models.csv