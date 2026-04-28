from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


# ------------------------------------------------------------
# Config (edit paths if needed)
# ------------------------------------------------------------

DCPL_ROOT = Path("results/runs/dcpl_split80_global")  # contains seed_42, seed_1042, ...

# Your baseline global runs are these timestamp folders
BASELINE_RUN_DIRS = [
    Path("results/runs/2026-01-22_114637"),
    Path("results/runs/2026-01-22_114645"),
    Path("results/runs/2026-01-22_114653"),
    Path("results/runs/2026-01-22_114735"),
    Path("results/runs/2026-01-22_115101"),
]

OUT_DIR = Path("results/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_STACKED = OUT_DIR / "global_runs_stacked_baseline_and_dcpl.csv"
OUT_AGG = OUT_DIR / "global_mean_std_baseline_vs_dcpl.csv"

OUT_TEX_R2 = OUT_DIR / "table_global_R2_mean_std_separate_cols.tex"
OUT_TEX_MRE = OUT_DIR / "table_global_MRE_mean_std_separate_cols.tex"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _escape_latex(s: str) -> str:
    return str(s).replace("_", r"\_")


def _infer_seed_from_path(p: Path) -> int | None:
    m = re.search(r"seed[_\-]?(\d+)", str(p))
    return int(m.group(1)) if m else None


def _find_summary_csv(run_dir: Path) -> Path:
    """
    Heuristic: find the most likely summary CSV in run_dir.
    Prefer files containing 'summary', else newest CSV.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    candidates = list(run_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found under: {run_dir}")

    summary_like = [p for p in candidates if "summary" in p.name.lower()]
    use = summary_like if summary_like else candidates
    return max(use, key=lambda p: p.stat().st_mtime)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to columns: learner, cv, target, R2, MRE, seed(optional)
    Accepts various column naming conventions.
    """
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]

    # learner
    if "learner" not in out.columns:
        for alt in ["model_kind", "method", "algo"]:
            if alt in out.columns:
                out["learner"] = out[alt].astype(str)
                break
    if "learner" not in out.columns:
        out["learner"] = "unknown"

    # cv/target
    if "cv" not in out.columns:
        out["cv"] = "split80_20"
    if "target" not in out.columns:
        out["target"] = "Target_throughput_tokens_per_sec"

    # R2
    r2_candidates = ["R2", "R2_mean", "mean_R2"]
    r2_col = next((c for c in r2_candidates if c in out.columns), None)
    if r2_col is None:
        raise KeyError(f"Could not find R2 column. Available: {list(out.columns)}")

    # MRE
    mre_candidates = ["MRE", "MRE%", "MRE(%)", "MRE_mean", "mean_MRE(%)", "mean_MRE"]
    mre_col = next((c for c in mre_candidates if c in out.columns), None)
    if mre_col is None:
        raise KeyError(f"Could not find MRE column. Available: {list(out.columns)}")

    out["R2"] = pd.to_numeric(out[r2_col], errors="coerce")
    out["MRE"] = pd.to_numeric(out[mre_col], errors="coerce")

    # seed if present
    if "seed" in out.columns:
        out["seed"] = pd.to_numeric(out["seed"], errors="coerce")

    return out


# ------------------------------------------------------------
# Collectors
# ------------------------------------------------------------

def _load_dcpl_one_seed_run(seed_dir: Path) -> pd.DataFrame:
    """
    DCPL: each seed folder -> one run (one row summary or a small CSV).
    We reduce to one row (global) if needed.
    """
    csv_path = _find_summary_csv(seed_dir)
    df = pd.read_csv(csv_path)
    df = _standardize_columns(df)

    seed = _infer_seed_from_path(seed_dir) or _infer_seed_from_path(csv_path)

    # Reduce to single row if multiple rows exist (should not for global, but safe)
    if len(df) > 1:
        df = pd.DataFrame([{
            "learner": "DCPL",
            "cv": str(df["cv"].iloc[0]),
            "target": str(df["target"].iloc[0]),
            "R2": float(df["R2"].mean()),
            "MRE": float(df["MRE"].mean()),
        }])
    else:
        # force learner label
        df.loc[:, "learner"] = "DCPL"

    df["method_group"] = "DCPL"
    df["seed"] = seed
    df["source_dir"] = str(seed_dir)
    df["source_csv"] = str(csv_path)

    return df[["method_group", "seed", "learner", "cv", "target", "R2", "MRE", "source_dir", "source_csv"]]


def _collect_dcpl(dcpl_root: Path) -> pd.DataFrame:
    seed_dirs = sorted([p for p in Path(dcpl_root).glob("seed_*") if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* folders found under: {dcpl_root}")
    return pd.concat([_load_dcpl_one_seed_run(d) for d in seed_dirs], ignore_index=True)


def _load_baseline_5runs_raw(run_dir: Path) -> pd.DataFrame:
    """
    Baseline: each timestamp folder contains files like:
      baseline_split80_merged__rf_light__5runs_raw.csv
    That file contains 5 rows => we must keep all 5 to compute std.
    """
    run_dir = Path(run_dir)
    raw_files = sorted(run_dir.glob("baseline_split80_merged__*__5runs_raw.csv"))
    if not raw_files:
        raise FileNotFoundError(
            f"No baseline 5runs raw file found in {run_dir}. "
            f"Expected: baseline_split80_merged__<learner>__5runs_raw.csv"
        )

    dfs = []
    for csv_path in raw_files:
        df = pd.read_csv(csv_path)
        df = _standardize_columns(df)

        # infer learner from filename (robust)
        m = re.search(r"baseline_split80_merged__(.+?)__5runs_raw\.csv", csv_path.name)
        learner_from_name = m.group(1) if m else None
        if learner_from_name:
            df["learner"] = learner_from_name

        # ensure seed exists: many raw files include seed; if not, index them 1..n
        if "seed" not in df.columns:
            df["seed"] = list(range(1, len(df) + 1))

        df["method_group"] = "Baseline"
        df["source_dir"] = str(run_dir)
        df["source_csv"] = str(csv_path)

        dfs.append(df[["method_group", "seed", "learner", "cv", "target", "R2", "MRE", "source_dir", "source_csv"]])

    return pd.concat(dfs, ignore_index=True)


def _collect_baselines(run_dirs: list[Path]) -> pd.DataFrame:
    return pd.concat([_load_baseline_5runs_raw(d) for d in run_dirs], ignore_index=True)


# ------------------------------------------------------------
# Aggregation + LaTeX
# ------------------------------------------------------------

def _agg_mean_std(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate mean/std across runs for each (method, learner).
    For DCPL learner is 'DCPL'. Baseline learners: lr/ridge/rf_light/nn/llm_pilot.
    """
    g = df_runs.groupby(["method_group", "learner"], dropna=False)

    out = g.agg(
        n_runs=("R2", "count"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", lambda x: x.std(ddof=1)),
        MRE_mean=("MRE", "mean"),
        MRE_std=("MRE", lambda x: x.std(ddof=1)),
    ).reset_index()

    return out


def _build_structured_table(agg: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Returns a single-row display table for global setting:
      Dataset | Baseline_mean | Baseline_std | DCPL_mean | DCPL_std

    For Baseline, we will use BEST baseline learner per metric:
      - R2: max mean
      - MRE: min mean
    This keeps the global table compact and readable.

    If you prefer “one column per baseline learner”, tell me and I will generate that variant.
    """
    metric = metric.upper()
    assert metric in {"R2", "MRE"}

    # separate baseline vs dcpl
    base = agg[agg["method_group"] == "Baseline"].copy()
    dcpl = agg[agg["method_group"] == "DCPL"].copy()

    if base.empty:
        raise RuntimeError("No baseline rows found in aggregated table.")
    if dcpl.empty:
        raise RuntimeError("No DCPL rows found in aggregated table.")

    # choose best baseline learner based on mean
    if metric == "R2":
        best_base = base.sort_values(f"{metric}_mean", ascending=False).iloc[0]
    else:
        best_base = base.sort_values(f"{metric}_mean", ascending=True).iloc[0]

    # DCPL (should be single row)
    dcpl_row = dcpl.sort_values(f"{metric}_mean", ascending=False).iloc[0]

    disp = pd.DataFrame([{
        "Dataset": "Merged-All",
        "Baseline_learner": str(best_base["learner"]),
        "Baseline_mean": float(best_base[f"{metric}_mean"]),
        "Baseline_std": float(best_base[f"{metric}_std"]) if pd.notna(best_base[f"{metric}_std"]) else 0.0,
        "DCPL_mean": float(dcpl_row[f"{metric}_mean"]),
        "DCPL_std": float(dcpl_row[f"{metric}_std"]) if pd.notna(dcpl_row[f"{metric}_std"]) else 0.0,
    }])

    return disp


def _to_camera_ready_latex_separate_cols(disp: pd.DataFrame, metric: str, out_path: Path) -> None:
    """
    Camera-ready LaTeX with separate mean/std columns (grouped header).
    """
    metric = metric.upper()
    assert metric in {"R2", "MRE"}

    # Formatting
    df = disp.copy()
    df["Dataset"] = df["Dataset"].map(_escape_latex)
    df["Baseline_learner"] = df["Baseline_learner"].map(_escape_latex)

    if metric == "R2":
        df["Baseline_mean"] = df["Baseline_mean"].map(lambda x: f"{x:.3f}")
        df["Baseline_std"]  = df["Baseline_std"].map(lambda x: f"{x:.3f}")
        df["DCPL_mean"]     = df["DCPL_mean"].map(lambda x: f"{x:.3f}")
        df["DCPL_std"]      = df["DCPL_std"].map(lambda x: f"{x:.3f}")
        metric_title = r"R$^2$"
        caption = r"Global throughput prediction performance on the merged dataset (80/20 split, 5 runs). Higher is better."
        label = "tab:global_r2_baseline_vs_dcpl"
    else:
        df["Baseline_mean"] = df["Baseline_mean"].map(lambda x: f"{x:.1f}")
        df["Baseline_std"]  = df["Baseline_std"].map(lambda x: f"{x:.1f}")
        df["DCPL_mean"]     = df["DCPL_mean"].map(lambda x: f"{x:.1f}")
        df["DCPL_std"]      = df["DCPL_std"].map(lambda x: f"{x:.1f}")
        metric_title = r"MRE (\%)"
        caption = r"Global throughput prediction performance on the merged dataset (80/20 split, 5 runs). Lower is better."
        label = "tab:global_mre_baseline_vs_dcpl"

    # Build LaTeX manually for a structured header
    lines = []
    for _, r in df.iterrows():
        lines.append(
            f"{r['Dataset']} & {r['Baseline_learner']} & "
            f"{r['Baseline_mean']} & {r['Baseline_std']} & "
            f"{r['DCPL_mean']} & {r['DCPL_std']} \\\\"
        )

    table = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\renewcommand{\\arraystretch}{1.2}\n"
        "\\begin{tabular}{llcccc}\n"
        "\\toprule\n"
        f"\\textbf{{Dataset}} & \\textbf{{Best Baseline}} & "
        f"\\multicolumn{{4}}{{c}}{{\\textbf{{{metric_title}}}}} \\\\\n"
        "\\cmidrule(lr){3-6}\n"
        " &  & \\multicolumn{2}{c}{Baseline} & \\multicolumn{2}{c}{DCPL} \\\\\n"
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}\n"
        " &  & mean & std & mean & std \\\\\n"
        "\\midrule\n"
        + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )

    out_path.write_text(table, encoding="utf-8")


def main():
    # 1) collect runs (stacked)
    dcpl_runs = _collect_dcpl(DCPL_ROOT)
    base_runs = _collect_baselines(BASELINE_RUN_DIRS)

    runs = pd.concat([base_runs, dcpl_runs], ignore_index=True)
    runs.to_csv(OUT_STACKED, index=False)
    print(f"[OK] Saved stacked runs: {OUT_STACKED}")

    # 2) aggregate mean/std per learner
    agg = _agg_mean_std(runs)
    agg.to_csv(OUT_AGG, index=False)
    print(f"[OK] Saved mean/std: {OUT_AGG}")

    # 3) Build compact structured display tables (global)
    disp_r2 = _build_structured_table(agg, metric="R2")
    disp_mre = _build_structured_table(agg, metric="MRE")

    # 4) LaTeX output with separate mean/std columns
    _to_camera_ready_latex_separate_cols(disp_r2, metric="R2", out_path=OUT_TEX_R2)
    print(f"[OK] Saved LaTeX (R2): {OUT_TEX_R2}")

    _to_camera_ready_latex_separate_cols(disp_mre, metric="MRE", out_path=OUT_TEX_MRE)
    print(f"[OK] Saved LaTeX (MRE): {OUT_TEX_MRE}")

    # preview
    print("\n=== Aggregated mean/std per method/learner ===")
    print(agg.sort_values(["method_group", "learner"]).to_string(index=False))


if __name__ == "__main__":
    main()
