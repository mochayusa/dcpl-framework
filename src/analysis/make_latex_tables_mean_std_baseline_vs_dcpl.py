from __future__ import annotations

from pathlib import Path
import pandas as pd

# =========================
# INPUT PATHS
# =========================
BASELINE_STACKED_CSV = Path(
    "results/runs/baseline_split80__MULTI_BASELINES__5x_base42/"
    "baseline_split80_permodel_ALL_learners_ALL_runs_stacked.csv"
)

# DCPL mean/std (already computed)
DCPL_MEAN_CSV = Path(
    "results/runs/dcpl_split80__multirun_5x_base42/dcpl_split80_permodel_mean.csv"
)
DCPL_STD_CSV = Path(
    "results/runs/dcpl_split80__multirun_5x_base42/dcpl_split80_permodel_std.csv"
)

OUT_DIR = Path("results/latex_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FILTER = "Target_throughput_tokens_per_sec"  # or None
CV_FILTER = "split80_20"                            # or None

# Column order (baseline learners + dcpl at end)
BASELINE_ORDER = ["lr", "ridge", "rf_light", "nn", "llm_pilot"]


# =========================
# Helpers
# =========================
def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if TARGET_FILTER is not None and "target" in out.columns:
        out = out[out["target"] == TARGET_FILTER]
    if CV_FILTER is not None and "cv" in out.columns:
        out = out[out["cv"] == CV_FILTER]
    return out


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")


def _latex_booktabs(df: pd.DataFrame, caption: str, label: str) -> str:
    # Note: pandas to_latex uses booktabs rules if you include \usepackage{booktabs}
    colfmt = "l" + "r" * (df.shape[1] - 1)
    return df.to_latex(
        index=False,
        escape=True,
        column_format=colfmt,
        caption=caption,
        label=label,
    )


def _short_dataset_name(s: str) -> str:
    """
    Turn: data_EleutherAI_gpt-neox-20b -> EleutherAI_gpt-neox-20b
    Keep underscores safe for LaTeX by escaping later via pandas.
    """
    if isinstance(s, str) and s.startswith("data_"):
        return s[len("data_"):]
    return str(s)


# =========================
# 1) BASELINE mean/std from STACKED
# =========================
baseline = pd.read_csv(BASELINE_STACKED_CSV)
baseline = _apply_filters(baseline)
_require_cols(baseline, ["per_model_file", "learner", "R2", "MRE"])

baseline["R2"] = pd.to_numeric(baseline["R2"], errors="coerce")
baseline["MRE"] = pd.to_numeric(baseline["MRE"], errors="coerce")

baseline["dataset"] = baseline["per_model_file"].map(_short_dataset_name)

baseline_stats = (
    baseline
    .groupby(["dataset", "learner"], dropna=False)[["R2", "MRE"]]
    .agg(["mean", "std"])
    .reset_index()
)
baseline_stats.columns = ["dataset", "learner", "R2_mean", "R2_std", "MRE_mean", "MRE_std"]

# enforce learner ordering if present
baseline_stats["learner"] = pd.Categorical(baseline_stats["learner"], categories=BASELINE_ORDER, ordered=True)


def _pivot(stats: pd.DataFrame, value_col: str, col_order: list[str]) -> pd.DataFrame:
    wide = stats.pivot_table(index="dataset", columns="learner", values=value_col, aggfunc="first").reset_index()
    # reorder columns
    cols = ["dataset"] + [c for c in col_order if c in wide.columns]
    # keep any unexpected learners at the end
    extras = [c for c in wide.columns if c not in cols]
    wide = wide[cols + extras]
    return wide


baseline_r2_mean = _pivot(baseline_stats, "R2_mean", BASELINE_ORDER)
baseline_r2_std  = _pivot(baseline_stats, "R2_std",  BASELINE_ORDER)
baseline_mre_mean = _pivot(baseline_stats, "MRE_mean", BASELINE_ORDER)
baseline_mre_std  = _pivot(baseline_stats, "MRE_std",  BASELINE_ORDER)


# =========================
# 2) DCPL mean/std from provided files
# =========================
dcpl_mean = _apply_filters(pd.read_csv(DCPL_MEAN_CSV))
dcpl_std  = _apply_filters(pd.read_csv(DCPL_STD_CSV))

_require_cols(dcpl_mean, ["per_model_file", "learner", "R2", "MRE"])
_require_cols(dcpl_std,  ["per_model_file", "learner", "R2", "MRE"])

for df in (dcpl_mean, dcpl_std):
    df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
    df["MRE"] = pd.to_numeric(df["MRE"], errors="coerce")
    df["dataset"] = df["per_model_file"].map(_short_dataset_name)

# merge mean+std to keep consistent row keys
id_cols = [c for c in ["dataset", "learner"]]
dcpl_merged = (
    dcpl_mean[id_cols + ["R2", "MRE"]].rename(columns={"R2": "R2_mean", "MRE": "MRE_mean"})
    .merge(dcpl_std[id_cols + ["R2", "MRE"]].rename(columns={"R2": "R2_std", "MRE": "MRE_std"}),
           on=id_cols, how="inner")
)

# In your output, learner looks like "dcpl_gate=ridge" — we want a single column name "dcpl"
# If you later have multiple dcpl variants, you can keep them; for now we collapse into one.
dcpl_col_name = "dcpl"

dcpl_r2_mean = dcpl_merged.groupby("dataset", as_index=False)["R2_mean"].first().rename(columns={"R2_mean": dcpl_col_name})
dcpl_r2_std  = dcpl_merged.groupby("dataset", as_index=False)["R2_std"].first().rename(columns={"R2_std": dcpl_col_name})
dcpl_mre_mean = dcpl_merged.groupby("dataset", as_index=False)["MRE_mean"].first().rename(columns={"MRE_mean": dcpl_col_name})
dcpl_mre_std  = dcpl_merged.groupby("dataset", as_index=False)["MRE_std"].first().rename(columns={"MRE_std": dcpl_col_name})


# =========================
# 3) Combine baseline + dcpl (wide tables)
# =========================
def _add_dcpl(wide_baseline: pd.DataFrame, dcpl_wide: pd.DataFrame) -> pd.DataFrame:
    out = wide_baseline.merge(dcpl_wide, on="dataset", how="left")
    # reorder to put dcpl at the end
    cols = list(out.columns)
    if dcpl_col_name in cols:
        cols = [c for c in cols if c != dcpl_col_name] + [dcpl_col_name]
        out = out[cols]
    return out


r2_mean_table = _add_dcpl(baseline_r2_mean, dcpl_r2_mean)
r2_std_table  = _add_dcpl(baseline_r2_std,  dcpl_r2_std)
mre_mean_table = _add_dcpl(baseline_mre_mean, dcpl_mre_mean)
mre_std_table  = _add_dcpl(baseline_mre_std,  dcpl_mre_std)

def format_decimal(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    """
    Format all numeric columns to fixed decimal places (as strings),
    except the identifier column 'dataset' / 'learner'.
    """
    out = df.copy()
    for c in out.columns:
        if c not in {"dataset", "learner"}:
            out[c] = out[c].apply(
                lambda x: f"{x:.{digits}f}" if pd.notna(x) else ""
            )
    return out


# === Force fixed 2-decimal formatting ===
r2_mean_table  = format_decimal(r2_mean_table,  digits=3)
r2_std_table   = format_decimal(r2_std_table,   digits=3)
mre_mean_table = format_decimal(mre_mean_table, digits=3)
mre_std_table  = format_decimal(mre_std_table,  digits=3)



# =========================
# 4) Export LaTeX
# =========================
r2_mean_tex = _latex_booktabs(
    r2_mean_table,
    caption="R\\textsuperscript{2} mean across 5 splits (baseline learners vs. DCPL).",
    label="tab:r2_mean_baseline_vs_dcpl",
)
r2_std_tex = _latex_booktabs(
    r2_std_table,
    caption="R\\textsuperscript{2} standard deviation across 5 splits (baseline learners vs. DCPL).",
    label="tab:r2_std_baseline_vs_dcpl",
)
mre_mean_tex = _latex_booktabs(
    mre_mean_table,
    caption="MRE mean across 5 splits (baseline learners vs. DCPL).",
    label="tab:mre_mean_baseline_vs_dcpl",
)
mre_std_tex = _latex_booktabs(
    mre_std_table,
    caption="MRE standard deviation across 5 splits (baseline learners vs. DCPL).",
    label="tab:mre_std_baseline_vs_dcpl",
)

(OUT_DIR / "r2_mean_table.tex").write_text(r2_mean_tex, encoding="utf-8")
(OUT_DIR / "r2_std_table.tex").write_text(r2_std_tex, encoding="utf-8")
(OUT_DIR / "mre_mean_table.tex").write_text(mre_mean_tex, encoding="utf-8")
(OUT_DIR / "mre_std_table.tex").write_text(mre_std_tex, encoding="utf-8")

print("[OK] Wrote:")
print(" -", OUT_DIR / "r2_mean_table.tex")
print(" -", OUT_DIR / "r2_std_table.tex")
print(" -", OUT_DIR / "mre_mean_table.tex")
print(" -", OUT_DIR / "mre_std_table.tex")
