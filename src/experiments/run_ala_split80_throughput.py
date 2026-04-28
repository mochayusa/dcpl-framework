from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ala.ala_throughput import (
    build_throughput_db_and_training_params,
    train_param_regressor,
    ala_predict_throughput,
    compute_metrics,
)
from ala.constants import COL_II, COL_OO, COL_BB, COL_THROUGHPUT


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")


def run_one_split80(df: pd.DataFrame, seed: int, test_size: float = 0.20, clip: bool = True) -> dict:
    """
    One 80/20 split evaluation for ALA throughput.
    Returns a summary row dict with metrics.
    """
    _ensure_cols(df, [COL_II, COL_OO, COL_BB, COL_THROUGHPUT])

    # optional: ensure numeric
    for c in [COL_II, COL_OO, COL_BB, COL_THROUGHPUT]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[COL_II, COL_OO, COL_BB, COL_THROUGHPUT]).copy()

    # split
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)

    # build DB + param training table
    param_db, T_df = build_throughput_db_and_training_params(df_train)

    # train param regressor
    param_regressor = train_param_regressor(T_df)
    if param_regressor is None:
        raise RuntimeError("T_df is empty -> cannot train parameter regressor (check grouping / filters).")

    # predict on test
    y_true = df_test[COL_THROUGHPUT].values.astype(float)
    y_pred = ala_predict_throughput(df_test, param_db, param_regressor, clip_nonneg=True)

    if clip:
        # optional robust clipping based on train distribution
        y_p99 = float(np.percentile(df_train[COL_THROUGHPUT].values.astype(float), 99))
        y_pred = np.clip(y_pred, 0.0, y_p99 * 2.0)

    # metrics
    met = compute_metrics(y_true, y_pred)

    return {
        "seed": seed,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        **met,
    }


def run_ala_split80_5runs(
    data_path: str | Path,
    seeds: list[int],
    test_size: float = 0.20,
    out_csv: str | Path | None = None,
    clip: bool = True,
) -> pd.DataFrame:
    data_path = Path(data_path)
    df = pd.read_csv(data_path)

    rows = []
    for i, s in enumerate(seeds, start=1):
        print(f"\n=== ALA split80 throughput | run {i}/{len(seeds)} | seed={s} ===")
        row = run_one_split80(df, seed=s, test_size=test_size, clip=clip)
        rows.append(row)

        print(
            f"R2={row['R2']:.4f}, RMSE={row['RMSE']:.4f}, "
            f"MAE={row['MAE']:.4f}, MRE={row['MRE']:.2f}%"
        )

    df_runs = pd.DataFrame(rows)

    # print aggregate mean/std
    print("\n=== ALA split80 (aggregate over runs) ===")
    for k in ["R2", "RMSE", "MAE", "MRE"]:
        mean_v = float(df_runs[k].mean())
        std_v = float(df_runs[k].std(ddof=1))  # sample std
        if k == "MRE":
            print(f"{k}: mean={mean_v:.2f}%  std={std_v:.2f}%")
        else:
            print(f"{k}: mean={mean_v:.4f}  std={std_v:.4f}")

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_runs.to_csv(out_csv, index=False)
        print(f"\n[OK] Saved: {out_csv}")

    return df_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to a single CSV (per-model or merged).")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seeds", type=str, default="42,1042,2042,3042,4042")
    ap.add_argument("--out", type=str, default="results/ala/ala_throughput_split80_5runs.csv")
    ap.add_argument("--no-clip", action="store_true", help="Disable throughput clipping.")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    run_ala_split80_5runs(
        data_path=args.data,
        seeds=seeds,
        test_size=args.test_size,
        out_csv=args.out,
        clip=not args.no_clip,
    )


if __name__ == "__main__":
    main()
