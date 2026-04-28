#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare:
- DaL (from dal_summary.csv, mean+std in same file)
- DCPL (nn gate) (from dcpl_gate_summaries_mean/std_across_runs.csv)
- Baselines (llm_pilot + rf_light) (from baseline_split80_permodel_mean_across_runs.csv)
  NOTE: baseline_mean file is actually per-run rows, so we compute mean/std by grouping.
        baseline_std file provided is NaN -> ignored.

Outputs:
- out_dir/compare_long.csv
- out_dir/compare_wide.csv
- out_dir/compare_table.tex (optional)

Run:
python src/analysis/compare_dal_dcpl_llmpilot.py \
  --dal results/DaL/raw_results/dal_summary.csv \
  --dcpl-mean results/runs/.../dcpl_gate_summaries_mean_across_runs.csv \
  --dcpl-std  results/runs/.../dcpl_gate_summaries_std_across_runs.csv \
  --base-mean results/runs/.../baseline_split80_permodel_mean_across_runs.csv \
  --dcpl-gate nn \
  --out-dir results/compare/dal_dcpl_llmpilot \
  --latex
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

METRICS = ["R2", "MAE", "RMSE", "MRE"]
HIGHER_BETTER = {"R2": True, "MAE": False, "RMSE": False, "MRE": False}


# -------------------------
# Utils
# -------------------------
def _norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower())


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        k = _norm_col(cand)
        if k in norm_map:
            return norm_map[k]
    return None


def _dataset_norm(x: str) -> str:
    """
    Align dataset naming between:
    - DaL: EleutherAI_gpt-neox-20b
    - DCPL/Baselines: data_EleutherAI_gpt-neox-20b
    """
    s = str(x).replace(".csv", "")
    if s.startswith("data_"):
        s = s[len("data_"):]
    s = s.replace("__final_for_dal", "")
    return s


def _format_mean_std(mean: float | None, std: float | None, decimals: int = 4) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return ""
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}$\\pm${std:.{decimals}f}"


# -------------------------
# Loaders
# -------------------------
def load_dal_summary(path: Path) -> pd.DataFrame:
    """
    Your schema:
    Dataset,dataset,num_of_run,num_of_success_run,num_of_fail_run,
    R2_mean,R2_std,MAE_mean,MAE_std,RMSE_mean,RMSE_std, MRE_mean,MRE_std
    """
    df = pd.read_csv(path)

    ds_col = _find_col(df, ["dataset", "Dataset"])
    if ds_col is None:
        raise KeyError(f"[DaL] Cannot find dataset column in {path}")

    rows = []
    for metric in METRICS:
        mean_col = _find_col(df, [f"{metric}_mean", f"mean_{metric}"])
        std_col  = _find_col(df, [f"{metric}_std",  f"std_{metric}"])
        if mean_col is None:
            continue  # e.g., MRE may not exist in your dal_summary.csv

        for _, r in df.iterrows():
            rows.append({
                "dataset": _dataset_norm(r[ds_col]),
                "metric": metric,
                "mean": float(r[mean_col]) if pd.notna(r[mean_col]) else np.nan,
                "std": float(r[std_col]) if (std_col and pd.notna(r[std_col])) else np.nan,
                "method": "DaL",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"[DaL] Parsed 0 rows from {path}.")
    return out


def load_dcpl_gate(mean_path: Path, std_path: Path, gate_kind: str, target: str | None, cv: str | None) -> pd.DataFrame:
    m = pd.read_csv(mean_path)
    s = pd.read_csv(std_path)

    # required cols
    pm_m = _find_col(m, ["per_model_file", "dataset"])
    pm_s = _find_col(s, ["per_model_file", "dataset"])
    if pm_m is None or pm_s is None:
        raise KeyError("[DCPL] Missing per_model_file column in DCPL mean/std.")

    # normalize naming
    m = m.rename(columns={pm_m: "per_model_file"}).copy()
    s = s.rename(columns={pm_s: "per_model_file"}).copy()

    # optional filters
    gcol_m = _find_col(m, ["gate_kind", "gate"])
    gcol_s = _find_col(s, ["gate_kind", "gate"])
    if gcol_m is None or gcol_s is None:
        raise KeyError("[DCPL] Missing gate_kind column in DCPL mean/std.")
    m = m[m[gcol_m].astype(str).str.lower() == gate_kind.lower()].copy()
    s = s[s[gcol_s].astype(str).str.lower() == gate_kind.lower()].copy()

    if target is not None:
        tcol_m = _find_col(m, ["target"])
        tcol_s = _find_col(s, ["target"])
        if tcol_m and tcol_s:
            m = m[m[tcol_m].astype(str) == target].copy()
            s = s[s[tcol_s].astype(str) == target].copy()

    if cv is not None:
        ccol_m = _find_col(m, ["cv"])
        ccol_s = _find_col(s, ["cv"])
        if ccol_m and ccol_s:
            m = m[m[ccol_m].astype(str) == cv].copy()
            s = s[s[ccol_s].astype(str) == cv].copy()

    # merge mean/std on keys
    key_cols = ["per_model_file"]
    for extra in ["target", "cv", "gate_kind"]:
        cm = _find_col(m, [extra])
        cs = _find_col(s, [extra])
        if cm and cs:
            m = m.rename(columns={cm: extra})
            s = s.rename(columns={cs: extra})
            key_cols.append(extra)

    merged = m.merge(s, on=key_cols, suffixes=("_mean", "_std"), how="inner")
    if merged.empty:
        raise RuntimeError("[DCPL] mean/std merge produced empty result. Check target/cv/gate filters.")

    rows = []
    for _, r in merged.iterrows():
        ds = _dataset_norm(r["per_model_file"])
        for metric in METRICS:
            mu_col = metric + "_mean"
            sd_col = metric + "_std"
            if mu_col not in merged.columns:
                continue
            rows.append({
                "dataset": ds,
                "metric": metric,
                "mean": float(r[mu_col]) if pd.notna(r[mu_col]) else np.nan,
                "std": float(r[sd_col]) if (sd_col in merged.columns and pd.notna(r[sd_col])) else np.nan,
                "method": f"DCPL({gate_kind}_gate)",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("[DCPL] Extracted 0 rows (after filtering).")
    return out


def load_baselines_from_per_run(base_mean_path: Path, learners_keep: list[str], target: str | None, cv: str | None) -> pd.DataFrame:
    """
    baseline_mean is per-run rows (1500 rows = 10 datasets × 5 learners × 30 runs).
    We compute mean/std across runs:
      groupby(per_model_file, target, cv, learner)
    """
    df = pd.read_csv(base_mean_path)

    pm = _find_col(df, ["per_model_file", "dataset"])
    lr = _find_col(df, ["learner", "model"])
    tg = _find_col(df, ["target"])
    cvcol = _find_col(df, ["cv"])

    if pm is None or lr is None or tg is None or cvcol is None:
        raise KeyError("[Baselines] Missing one of required columns: per_model_file, learner, target, cv.")

    df = df.rename(columns={pm: "per_model_file", lr: "learner", tg: "target", cvcol: "cv"}).copy()

    # filter target/cv if asked
    if target is not None:
        df = df[df["target"].astype(str) == target].copy()
    if cv is not None:
        df = df[df["cv"].astype(str) == cv].copy()

    # keep only specified learners
    df["learner"] = df["learner"].astype(str)
    df = df[df["learner"].isin(learners_keep)].copy()
    if df.empty:
        raise RuntimeError(f"[Baselines] After filtering learners={learners_keep}, got 0 rows.")

    # aggregate mean/std across runs for each dataset×learner
    agg = df.groupby(["per_model_file", "target", "cv", "learner"], as_index=False).agg(
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MRE_mean=("MRE", "mean"),
        MRE_std=("MRE", "std"),
    )

    # to long
    rows = []
    for _, r in agg.iterrows():
        ds = _dataset_norm(r["per_model_file"])
        learner = str(r["learner"])
        method = "LLM_Pilot" if learner == "llm_pilot" else learner  # keep rf_light as-is

        for metric in METRICS:
            rows.append({
                "dataset": ds,
                "metric": metric,
                "mean": float(r[f"{metric}_mean"]) if pd.notna(r[f"{metric}_mean"]) else np.nan,
                "std": float(r[f"{metric}_std"]) if pd.notna(r[f"{metric}_std"]) else np.nan,
                "method": method,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("[Baselines] Extracted 0 rows after aggregation.")
    return out


# -------------------------
# Best + Wide + LaTeX
# -------------------------
def compute_best_flags(df_long: pd.DataFrame) -> pd.DataFrame:
    out = df_long.copy()
    out["is_best"] = False

    for (ds, met), g in out.groupby(["dataset", "metric"], sort=False):
        gg = g.dropna(subset=["mean"])
        if gg.empty:
            continue
        idx = gg["mean"].idxmax() if HIGHER_BETTER.get(met, False) else gg["mean"].idxmin()
        out.loc[idx, "is_best"] = True

    return out


def to_wide_table(df_long_best: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    df = df_long_best.copy()
    df["cell"] = df.apply(lambda r: _format_mean_std(r["mean"], r["std"], decimals=decimals), axis=1)
    df.loc[df["is_best"], "cell"] = df.loc[df["is_best"], "cell"].apply(
        lambda s: f"\\textbf{{{s}}}" if s else s
    )

    wide = (
        df.pivot_table(index=["dataset", "metric"], columns="method", values="cell", aggfunc="first")
          .reset_index()
    )

    preferred = ["DaL", "LLM_Pilot", "rf_light", "DCPL(nn_gate)"]
    cols = ["dataset", "metric"] + [c for c in preferred if c in wide.columns] + \
           [c for c in wide.columns if c not in (["dataset", "metric"] + preferred)]
    return wide[cols]


def wide_to_latex(wide: pd.DataFrame, out_path: Path, caption: str, label: str):
    methods = [c for c in wide.columns if c not in ("dataset", "metric")]
    colspec = "ll" + "c" * len(methods)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        "Dataset & Metric & " + " & ".join(m.replace("_", "\\_") for m in methods) + " \\\\",
        "\\midrule",
    ]

    last_ds = None
    for _, r in wide.iterrows():
        ds = str(r["dataset"]).replace("_", "\\_")
        met = str(r["metric"])
        if last_ds is not None and ds != last_ds:
            lines.append("\\addlinespace")
        row = [ds, met] + [str(r.get(m, "")) if pd.notna(r.get(m, "")) else "" for m in methods]
        lines.append(" & ".join(row) + " \\\\")
        last_ds = ds

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dal", type=str, required=True)
    ap.add_argument("--dcpl-mean", type=str, required=True)
    ap.add_argument("--dcpl-std", type=str, required=True)
    ap.add_argument("--base-mean", type=str, required=True)

    ap.add_argument("--dcpl-gate", type=str, default="nn")
    ap.add_argument("--target", type=str, default="Target_throughput_tokens_per_sec")
    ap.add_argument("--cv", type=str, default="split80_20")

    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--decimals", type=int, default=4)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # DaL
    dal_long = load_dal_summary(Path(args.dal))

    # DCPL nn gate
    dcpl_long = load_dcpl_gate(
        Path(args.dcpl_mean),
        Path(args.dcpl_std),
        gate_kind=args.dcpl_gate,
        target=args.target,
        cv=args.cv,
    )

    # Baselines: llm_pilot + rf_light (computed mean/std from per-run rows)
    base_long = load_baselines_from_per_run(
        Path(args.base_mean),
        learners_keep=["llm_pilot", "rf_light"],
        target=args.target,
        cv=args.cv,
    )

    # method naming alignment for wide table
    dcpl_long["method"] = dcpl_long["method"].replace({f"DCPL({args.dcpl_gate}_gate)": "DCPL(nn_gate)"})
    base_long["method"] = base_long["method"].replace({"rf_light": "rf_light"})

    df_long = pd.concat([dal_long, base_long, dcpl_long], ignore_index=True)
    df_long["metric"] = df_long["metric"].astype(str).str.upper()

    # best + wide + save
    df_best = compute_best_flags(df_long)
    wide = to_wide_table(df_best, decimals=int(args.decimals))

    long_path = out_dir / "compare_long.csv"
    wide_path = out_dir / "compare_wide.csv"
    df_best.to_csv(long_path, index=False)
    wide.to_csv(wide_path, index=False)
    print(f"[OK] {long_path}")
    print(f"[OK] {wide_path}")

    if args.latex:
        tex_path = out_dir / "compare_table.tex"
        caption = (
            "DaL vs LLM Pilot vs rf\\_light vs DCPL (NN gate) comparison using mean$\\pm$std across runs. "
            "Best values are bold per dataset (R2 higher is better; MAE/RMSE/MRE lower is better)."
        )
        label = "tab:dal_llmpilot_rflight_dcpl"
        wide_to_latex(wide, tex_path, caption=caption, label=label)
        print(f"[OK] {tex_path}")


if __name__ == "__main__":
    main()
