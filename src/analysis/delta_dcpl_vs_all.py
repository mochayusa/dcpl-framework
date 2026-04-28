#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["R2", "MAE", "RMSE", "MRE"]
HIGHER_BETTER = {"R2": True, "MAE": False, "RMSE": False, "MRE": False}


# -----------------------------
# Helpers
# -----------------------------
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower())


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    nmap = {norm(c): c for c in df.columns}
    for c in candidates:
        k = norm(c)
        if k in nmap:
            return nmap[k]
    return None


def fmt_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def safe_tex(s: str) -> str:
    return str(s).replace("_", "\\_")


def norm_dataset(s: str) -> str:
    s = str(s).replace(".csv", "")
    if s.startswith("data_"):
        s = s[len("data_"):]
    s = s.replace("__final_for_dal", "")
    return s


# -----------------------------
# Loader: DaL (mean+std in same file)
# -----------------------------
def load_dal_summary(path: Path) -> pd.DataFrame:
    """
    Expected schema like:
    Dataset,dataset,num_of_run,num_of_success_run,num_of_fail_run,
    R2_mean,R2_std,MAE_mean,MAE_std,RMSE_mean,RMSE_std,(optional MRE_mean,MRE_std)
    """
    df = pd.read_csv(path)

    # Prefer 'dataset' (lowercase) then fallback
    ds_col = find_col(df, ["dataset", "Dataset"])
    if ds_col is None:
        raise KeyError(f"[DaL] dataset column not found in {path}")

    rows = []
    for m in METRICS:
        mean_col = find_col(df, [f"{m}_mean", f"mean_{m}", f"{m}_mean_across_runs"])
        std_col  = find_col(df, [f"{m}_std",  f"std_{m}",  f"{m}_std_across_runs"])
        if mean_col is None:
            continue  # allow missing MRE in DaL

        for _, r in df.iterrows():
            rows.append({
                "dataset": norm_dataset(r[ds_col]),
                "metric": m,
                "mean": fmt_float(r[mean_col]),
                "std": fmt_float(r[std_col]) if std_col else np.nan,
                "method": "DaL",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"[DaL] failed to parse metrics from {path}. Check columns.")
    return out


def load_mean_std_pair(mean_path: Path, std_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(mean_path), pd.read_csv(std_path)


# -----------------------------
# Baselines parser (PER-RUN long -> aggregate mean+std)
# -----------------------------
def parse_baselines_per_run(
    base_mean_path: Path,
    *,
    target_filter: str | None,
    cv_filter: str | None,
    learners_keep: list[str] | None,
) -> pd.DataFrame:
    """
    baseline_mean is per-run long, e.g:
    per_model_file,target,learner,cv,experiment,R2,MAE,RMSE,MRE
    We aggregate mean/std across runs per (dataset, learner).
    """
    df = pd.read_csv(base_mean_path)

    ds_col = find_col(df, ["per_model_file", "dataset"])
    lr_col = find_col(df, ["learner", "method", "baseline"])
    tg_col = find_col(df, ["target"])
    cv_col = find_col(df, ["cv"])
    if not (ds_col and lr_col and tg_col and cv_col):
        raise KeyError(
            "[Baselines] expected columns: per_model_file (or dataset), learner, target, cv."
        )

    df = df.rename(columns={ds_col: "dataset", lr_col: "learner", tg_col: "target", cv_col: "cv"}).copy()
    df["dataset"] = df["dataset"].map(norm_dataset)
    df["learner"] = df["learner"].astype(str).str.strip().replace({"RF_light": "rf_light"})

    # filter target/cv
    if target_filter is not None:
        df = df[df["target"].astype(str) == target_filter].copy()
    if cv_filter is not None:
        df = df[df["cv"].astype(str) == cv_filter].copy()

    # filter learners if requested
    if learners_keep is not None:
        df = df[df["learner"].isin(learners_keep)].copy()

    if df.empty:
        raise RuntimeError("[Baselines] empty after filtering. Check target/cv/learners.")

    # Aggregate across runs
    agg = df.groupby(["dataset", "learner"], as_index=False).agg(
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MRE_mean=("MRE", "mean"), MRE_std=("MRE", "std"),
    )

    # to long (dataset, metric, mean, std, method)
    rows = []
    for _, r in agg.iterrows():
        learner = str(r["learner"])
        method = "LLM_Pilot" if learner == "llm_pilot" else learner  # nicer name
        for met in METRICS:
            rows.append({
                "dataset": r["dataset"],
                "metric": met,
                "mean": fmt_float(r.get(f"{met}_mean", np.nan)),
                "std": fmt_float(r.get(f"{met}_std", np.nan)),
                "method": method,
            })

    out = pd.DataFrame(rows)
    out = out[out["metric"].isin(METRICS)].copy()
    if out.empty:
        raise RuntimeError("[Baselines] produced 0 rows after aggregation.")
    return out


# -----------------------------
# DCPL parser (per_model_file + gate_kind + mean/std files)
# -----------------------------
def parse_dcpl(
    mean_path: Path,
    std_path: Path,
    *,
    gate_name: str = "nn",
    target_filter: str | None,
    cv_filter: str | None,
) -> pd.DataFrame:
    mean_df, std_df = load_mean_std_pair(mean_path, std_path)

    ds_m = find_col(mean_df, ["per_model_file", "dataset"])
    ds_s = find_col(std_df,  ["per_model_file", "dataset"])
    if not (ds_m and ds_s):
        raise KeyError("[DCPL] dataset/per_model_file missing in dcpl files.")

    gate_m = find_col(mean_df, ["gate_kind", "gate"])
    gate_s = find_col(std_df,  ["gate_kind", "gate"])
    if not (gate_m and gate_s):
        raise KeyError("[DCPL] gate_kind missing in dcpl files.")

    mean_df = mean_df.rename(columns={ds_m: "dataset", gate_m: "gate_kind"}).copy()
    std_df  = std_df.rename(columns={ds_s: "dataset", gate_s: "gate_kind"}).copy()

    mean_df["dataset"] = mean_df["dataset"].map(norm_dataset)
    std_df["dataset"]  = std_df["dataset"].map(norm_dataset)

    # gate filter
    mean_df = mean_df[mean_df["gate_kind"].astype(str).str.lower() == gate_name.lower()].copy()
    std_df  = std_df[std_df["gate_kind"].astype(str).str.lower() == gate_name.lower()].copy()

    # target/cv filters (optional)
    if target_filter is not None:
        t_m = find_col(mean_df, ["target"])
        t_s = find_col(std_df,  ["target"])
        if t_m and t_s:
            mean_df = mean_df[mean_df[t_m].astype(str) == target_filter].copy()
            std_df  = std_df[std_df[t_s].astype(str) == target_filter].copy()

    if cv_filter is not None:
        c_m = find_col(mean_df, ["cv"])
        c_s = find_col(std_df,  ["cv"])
        if c_m and c_s:
            mean_df = mean_df[mean_df[c_m].astype(str) == cv_filter].copy()
            std_df  = std_df[std_df[c_s].astype(str) == cv_filter].copy()

    # merge on keys
    key_cols = ["dataset"]
    for k in ["target", "cv", "gate_kind"]:
        km = find_col(mean_df, [k])
        ks = find_col(std_df,  [k])
        if km and ks:
            mean_df = mean_df.rename(columns={km: k})
            std_df  = std_df.rename(columns={ks: k})
            key_cols.append(k)

    merged = mean_df.merge(std_df, on=key_cols, suffixes=("_mean", "_std"), how="inner")
    if merged.empty:
        raise RuntimeError("[DCPL] mean/std merge empty after filters. Check target/cv/gate.")

    rows = []
    for _, r in merged.iterrows():
        for met in METRICS:
            if f"{met}_mean" not in merged.columns:
                continue
            rows.append({
                "dataset": r["dataset"],
                "metric": met,
                "mean": fmt_float(r[f"{met}_mean"]),
                "std": fmt_float(r.get(f"{met}_std", np.nan)),
                "method": f"DCPL({gate_name}_gate)",
            })

    out = pd.DataFrame(rows)
    out = out[out["metric"].isin(METRICS)].copy()
    if out.empty:
        raise RuntimeError("[DCPL] produced 0 rows; check columns R2/MAE/RMSE/MRE exist.")
    return out


# -----------------------------
# Delta computation
# -----------------------------
def compute_delta(df_long: pd.DataFrame, dcpl_method: str) -> pd.DataFrame:
    piv = df_long.pivot_table(index=["dataset", "metric"], columns="method", values="mean", aggfunc="first")
    if dcpl_method not in piv.columns:
        raise KeyError(f"DCPL method '{dcpl_method}' not found. Available: {list(piv.columns)}")

    dcpl = piv[dcpl_method]
    out_rows = []

    for baseline in piv.columns:
        if baseline == dcpl_method:
            continue

        base = piv[baseline]
        for (ds, met), base_val in base.items():
            if (ds, met) not in dcpl.index:
                continue
            dcpl_val = dcpl.loc[(ds, met)]
            if pd.isna(base_val) or pd.isna(dcpl_val):
                continue

            if HIGHER_BETTER.get(met, False):  # R2
                d = float(dcpl_val - base_val)
                pct = np.nan
            else:  # errors
                d = float(base_val - dcpl_val)
                pct = float((base_val - dcpl_val) / max(abs(base_val), 1e-12) * 100.0)

            out_rows.append({
                "dataset": ds,
                "metric": met,
                "baseline": baseline,
                "dcpl_method": dcpl_method,
                "delta": d,
                "delta_pct": pct,
                "dcpl_wins": bool(d > 0),
            })

    return pd.DataFrame(out_rows)


# -----------------------------
# LaTeX Writers
# -----------------------------
def write_delta_wide_latex(delta_wide: pd.DataFrame, out_tex: Path, caption: str, label: str, decimals: int = 4):
    df = delta_wide.copy()
    baselines = [c for c in df.columns if c not in ("dataset", "metric")]

    # bold max delta per row
    for i in range(len(df)):
        vals = []
        for b in baselines:
            v = df.at[i, b]
            vals.append(v if pd.notna(v) else -np.inf)
        j = int(np.argmax(vals)) if vals else None

        for k, b in enumerate(baselines):
            v = df.at[i, b]
            if pd.isna(v):
                df.at[i, b] = ""
                continue
            s = f"{float(v):.{decimals}f}"
            if j is not None and k == j:
                s = f"\\textbf{{{s}}}"
            df.at[i, b] = s

    colspec = "ll" + "c" * len(baselines)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        "Dataset & Metric & " + " & ".join(safe_tex(b) for b in baselines) + " \\\\",
        "\\midrule",
    ]

    last_ds = None
    for _, r in df.iterrows():
        ds = safe_tex(r["dataset"])
        met = str(r["metric"])
        if last_ds is not None and ds != last_ds:
            lines.append("\\addlinespace")
        row = [ds, met] + [str(r[b]) for b in baselines]
        lines.append(" & ".join(row) + " \\\\")
        last_ds = ds

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def write_compare_means_wide_latex(df_long: pd.DataFrame, out_tex: Path, caption: str, label: str, decimals: int = 4):
    df = df_long.copy()

    mean_p = df.pivot_table(index=["dataset", "metric"], columns="method", values="mean", aggfunc="first")
    std_p  = df.pivot_table(index=["dataset", "metric"], columns="method", values="std",  aggfunc="first")
    methods = list(mean_p.columns)

    rows = []
    for (ds, met) in mean_p.index:
        means = mean_p.loc[(ds, met)]
        stds  = std_p.loc[(ds, met)] if (ds, met) in std_p.index else pd.Series(index=methods, dtype=float)

        valid = means.dropna()
        best_method = None
        if not valid.empty:
            best_method = valid.idxmax() if HIGHER_BETTER.get(met, False) else valid.idxmin()

        row = {"dataset": ds, "metric": met}
        for m in methods:
            mu = means.get(m, np.nan)
            sd = stds.get(m, np.nan)
            if pd.isna(mu):
                cell = ""
            else:
                cell = f"{float(mu):.{decimals}f}" if pd.isna(sd) else f"{float(mu):.{decimals}f}$\\pm${float(sd):.{decimals}f}"
            if best_method == m and cell:
                cell = f"\\textbf{{{cell}}}"
            row[m] = cell

        rows.append(row)

    wide = pd.DataFrame(rows)
    colspec = "ll" + "c" * len(methods)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        "Dataset & Metric & " + " & ".join(safe_tex(m) for m in methods) + " \\\\",
        "\\midrule",
    ]

    last_ds = None
    for _, r in wide.iterrows():
        ds = safe_tex(r["dataset"])
        met = str(r["metric"])
        if last_ds is not None and ds != last_ds:
            lines.append("\\addlinespace")
        row = [ds, met] + [str(r[m]) for m in methods]
        lines.append(" & ".join(row) + " \\\\")
        last_ds = ds

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dal", type=str, required=True, help="DaL dal_summary.csv (mean+std in same file)")
    ap.add_argument("--dcpl-mean", type=str, required=True)
    ap.add_argument("--dcpl-std", type=str, required=True)
    ap.add_argument("--base-mean", type=str, required=True, help="Baseline per-run CSV (mean_across_runs file)")
    ap.add_argument("--dcpl-gate", type=str, default="nn")
    ap.add_argument("--target", type=str, default="Target_throughput_tokens_per_sec")
    ap.add_argument("--cv", type=str, default="split80_20")
    ap.add_argument("--learners", type=str, default="", help="Optional comma-list learners to keep (e.g. lr,ridge,nn,rf,llm_pilot,rf_light)")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--decimals", type=int, default=4)
    ap.add_argument("--latex", action="store_true", help="Write LaTeX tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    learners_keep = None
    if args.learners.strip():
        learners_keep = [x.strip() for x in args.learners.split(",") if x.strip()]
        # normalize possible RF_light typo
        learners_keep = ["rf_light" if x == "RF_light" else x for x in learners_keep]

    # Load
    dal_long = load_dal_summary(Path(args.dal))

    base_long = parse_baselines_per_run(
        Path(args.base_mean),
        target_filter=args.target,
        cv_filter=args.cv,
        learners_keep=learners_keep,
    )

    dcpl_long = parse_dcpl(
        Path(args.dcpl_mean),
        Path(args.dcpl_std),
        gate_name=args.dcpl_gate,
        target_filter=args.target,
        cv_filter=args.cv,
    )

    # ensure naming
    base_long["method"] = base_long["method"].astype(str).replace({"RF_light": "rf_light"})
    dcpl_method = f"DCPL({args.dcpl_gate}_gate)"

    df_long = pd.concat([dal_long, base_long, dcpl_long], ignore_index=True)
    df_long["metric"] = df_long["metric"].astype(str).str.upper()
    df_long = df_long[df_long["metric"].isin(METRICS)].copy()

    # Save compare long
    compare_long_path = out_dir / "compare_long.csv"
    df_long.to_csv(compare_long_path, index=False)

    # Delta
    delta_long = compute_delta(df_long, dcpl_method=dcpl_method)
    delta_long_path = out_dir / "delta_dcpl_vs_all_long.csv"
    delta_long.to_csv(delta_long_path, index=False)

    delta_wide = (
        delta_long.pivot_table(index=["dataset", "metric"], columns="baseline", values="delta", aggfunc="first")
                  .reset_index()
    )
    delta_wide_path = out_dir / "delta_dcpl_vs_all_wide.csv"
    delta_wide.to_csv(delta_wide_path, index=False)

    delta_summary = (
        delta_long.groupby(["baseline", "metric"], as_index=False)
                  .agg(
                      n=("delta", "count"),
                      delta_mean=("delta", "mean"),
                      delta_std=("delta", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
                      win_rate=("dcpl_wins", "mean"),
                      delta_pct_mean=("delta_pct", "mean"),
                      delta_pct_std=("delta_pct", lambda x: float(np.std(x.dropna(), ddof=1)) if x.dropna().shape[0] > 1 else 0.0),
                  )
                  .sort_values(["metric", "delta_mean"], ascending=[True, False])
    )
    delta_summary_path = out_dir / "delta_dcpl_vs_all_summary.csv"
    delta_summary.to_csv(delta_summary_path, index=False)

    print(f"[OK] {compare_long_path}")
    print(f"[OK] {delta_long_path}")
    print(f"[OK] {delta_wide_path}")
    print(f"[OK] {delta_summary_path}")

    if args.latex:
        tex_delta = out_dir / "delta_dcpl_vs_all_wide.tex"
        cap_delta = (
            "Selisih (\\textit{delta}) DCPL terhadap setiap baseline/existing study. "
            "Definisi: untuk R2, $\\Delta=\\text{DCPL}-\\text{baseline}$; "
            "untuk MAE/RMSE/MRE, $\\Delta=\\text{baseline}-\\text{DCPL}$. "
            "Dengan demikian $\\Delta>0$ berarti DCPL lebih baik. "
            "Nilai $\\Delta$ terbesar per dataset--metric dicetak tebal."
        )
        lab_delta = "tab:delta_dcpl_vs_all"
        write_delta_wide_latex(delta_wide, tex_delta, caption=cap_delta, label=lab_delta, decimals=args.decimals)
        print(f"[OK] {tex_delta}")

        tex_cmp = out_dir / "compare_means_wide.tex"
        cap_cmp = (
            "Perbandingan mean$\\pm$std antar metode (DaL, baseline/existing study, dan DCPL). "
            "Nilai terbaik per dataset--metric dicetak tebal (R2 lebih besar lebih baik; MAE/RMSE/MRE lebih kecil lebih baik)."
        )
        lab_cmp = "tab:compare_means_all"
        write_compare_means_wide_latex(df_long, tex_cmp, caption=cap_cmp, label=lab_cmp, decimals=args.decimals)
        print(f"[OK] {tex_cmp}")


if __name__ == "__main__":
    main()
