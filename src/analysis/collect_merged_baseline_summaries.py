from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, default="results/runs")
    ap.add_argument("--pattern", type=str, default="baseline_split80_merged__*__mean_std.csv")
    ap.add_argument("--out", type=str, default="results/analysis/baseline_merged_mean_std_all_learners.csv")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(runs_root.rglob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern under {runs_root}: {args.pattern}")

    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df.to_csv(out, index=False)

    print(f"[OK] Collected {len(files)} files")
    print(f"[OK] Saved: {out}")
    print(df[["learner", "R2_mean", "R2_std", "MRE_mean", "MRE_std"]].sort_values("R2_mean", ascending=False))


if __name__ == "__main__":
    main()
