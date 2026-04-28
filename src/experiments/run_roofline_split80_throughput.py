# src/experiments/run_roofline_split80_throughput.py

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from roofline.constants import COL_TARGET
from roofline.model import train_roofline_lr, predict_roofline_lr
from ala.ala_throughput import compute_metrics  # reuse your metric fn (R2/MAE/RMSE/MRE)

def run_one(df: pd.DataFrame, seed: int, test_size: float = 0.20) -> dict:
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)

    model, feat_cols, df_train_used = train_roofline_lr(df_train, target_col=COL_TARGET)
    y_pred, kept_idx = predict_roofline_lr(model, df_test, target_col=COL_TARGET)

    y_true = df_test.loc[kept_idx, COL_TARGET].astype(float).values
    met = compute_metrics(y_true, y_pred)

    return {
        "seed": seed,
        "n_train": int(len(df_train_used)),
        "n_test": int(len(kept_idx)),
        **met,
    }

def run_5runs(data_path: str | Path, seeds: list[int], out_csv: str | Path, test_size: float = 0.20) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    rows = []
    for i, s in enumerate(seeds, start=1):
        print(f"\n=== Roofline-LR split80 | run {i}/{len(seeds)} | seed={s} ===")
        r = run_one(df, seed=s, test_size=test_size)
        rows.append(r)
        print(f"R2={r['R2']:.4f}, RMSE={r['RMSE']:.4f}, MAE={r['MAE']:.4f}, MRE={r['MRE']:.2f}%")

    out = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved: {out_csv}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/roofline/roofline_lr_split80_5runs.csv")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seeds", type=str, default="42,1042,2042,3042,4042")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    run_5runs(args.data, seeds=seeds, out_csv=args.out, test_size=args.test_size)

if __name__ == "__main__":
    main()
