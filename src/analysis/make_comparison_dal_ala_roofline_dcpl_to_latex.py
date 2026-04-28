# src/analysis/make_comparison_dal_ala_roofline_dcpl_to_latex.py
"""
Create unified comparison tables (DaL vs ALA vs Roofline+LR vs DCPL)
from the following CSVs:

- DaL:       results/DaL/raw_results/dal_results_summary_flat.csv
- ALA:       results/ala/ala_permodel_split80_summary.csv
- Roofline:  results/roofline/roofline_lr_permodel_split80_summary.csv
- DCPL mean: results/runs/dcpl_split80__multirun_5x_base42/dcpl_split80_permodel_mean.csv
- DCPL std:  results/runs/dcpl_split80__multirun_5x_base42/dcpl_split80_permodel_std.csv

Outputs:
- results/analysis/comparison_merged_mean_std.csv
- results/analysis/table_R2_mean_std.tex
- results/analysis/table_MRE_mean_std.tex

Notes:
- Only R2 and MRE are included (mean/std in separate columns).
- LaTeX cells are formatted as mean±std.
- Best per row is bolded: max R2, min MRE.
- Display headers show "Roofline+LR" but internal prefix is "RooflineLR".
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


# ---------------------------
# Paths (edit if needed)
# ---------------------------

DAL_PATH = Path("results/DaL/raw_results/dal_results_summary_flat.csv")
ALA_PATH = Path("results/ala/ala_permodel_split80_summary.csv")
ROOF_PATH = Path("results/roofline/roofline_lr_permodel_split80_summary.csv")

DCPL_MEAN_PATH = Path("results/runs/dcpl_split80__multirun_5x_base42/dcpl_split80_permodel_mean.csv")
DCPL_STD_PATH  = Path("results/runs/dcpl_split80__multirun_5x_base42/dcpl_split80_permodel_std.csv")

OUT_DIR = Path("results/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MERGED_CSV = OUT_DIR / "comparison_merged_mean_std.csv"
OUT_TEX_R2  = OUT_DIR / "table_R2_mean_std.tex"
OUT_TEX_MRE = OUT_DIR / "table_MRE_mean_std.tex"


# ---------------------------
# Helpers
# ---------------------------

def _normalize_model_tag(x: str) -> str:
    """
    Normalize per-model identifiers so all methods join correctly.

    Examples:
      - "data_EleutherAI_gpt-neox-20b" -> "EleutherAI_gpt-neox-20b"
      - "EleutherAI_gpt-neox-20b"      -> "EleutherAI_gpt-neox-20b"
    """
    s = str(x).strip()
    s = re.sub(r"^data_", "", s)
    return s

def _escape_latex(s: str) -> str:
    """Minimal LaTeX escaping for underscores."""
    return str(s).replace("_", r"\_")

def _fmt_mean_std(mean: float, std: float, metric: str) -> str:
    """
    Format for LaTeX cell: mean±std with sensible decimals.
    R2: 3 decimals; MRE: 1 decimal (percentage-like).
    """
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        std = 0.0

    if metric.lower() == "r2":
        return f"{mean:.3f}$\\pm${std:.3f}"
    else:  # MRE
        return f"{mean:.1f}$\\pm${std:.1f}"

def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{name}] Missing columns {missing}. Available: {list(df.columns)}")

def _standardize_to_mean_std(
    df: pd.DataFrame,
    *,
    name: str,
    model_col: str,
    r2_mean_col: str,
    r2_std_col: str,
    mre_mean_col: str,
    mre_std_col: str,
) -> pd.DataFrame:
    _require_cols(df, [model_col, r2_mean_col, r2_std_col, mre_mean_col, mre_std_col], name)
    out = df[[model_col, r2_mean_col, r2_std_col, mre_mean_col, mre_std_col]].copy()
    out.rename(
        columns={
            model_col: "model_tag",
            r2_mean_col: "R2_mean",
            r2_std_col: "R2_std",
            mre_mean_col: "MRE_mean",
            mre_std_col: "MRE_std",
        },
        inplace=True,
    )
    out["model_tag"] = out["model_tag"].map(_normalize_model_tag)

    # numeric conversion
    for c in ["R2_mean", "R2_std", "MRE_mean", "MRE_std"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # drop duplicated model tags if any (keep first)
    out = out.drop_duplicates(subset=["model_tag"], keep="first").reset_index(drop=True)
    return out


# ---------------------------
# Load + standardize each method
# ---------------------------

def load_dal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _standardize_to_mean_std(
        df,
        name="DaL",
        model_col="model_tag",
        r2_mean_col="R2_mean",
        r2_std_col="R2_std",
        mre_mean_col="MRE_mean",
        mre_std_col="MRE_std",
    )

def load_ala(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _standardize_to_mean_std(
        df,
        name="ALA",
        model_col="per_model_file",
        r2_mean_col="R2_mean",
        r2_std_col="R2_std",
        mre_mean_col="MRE_mean",
        mre_std_col="MRE_std",
    )

def load_roofline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _standardize_to_mean_std(
        df,
        name="Roofline+LR",
        model_col="per_model_file",
        r2_mean_col="R2_mean",
        r2_std_col="R2_std",
        mre_mean_col="MRE_mean",
        mre_std_col="MRE_std",
    )

def load_dcpl(mean_path: Path, std_path: Path) -> pd.DataFrame:
    mean_df = pd.read_csv(mean_path)
    std_df  = pd.read_csv(std_path)

    _require_cols(mean_df, ["per_model_file", "R2", "MRE"], "DCPL(mean)")
    _require_cols(std_df,  ["per_model_file", "R2", "MRE"], "DCPL(std)")

    mean_df = mean_df[["per_model_file", "R2", "MRE"]].copy()
    std_df  = std_df[["per_model_file", "R2", "MRE"]].copy()

    mean_df.rename(columns={"per_model_file": "model_tag", "R2": "R2_mean", "MRE": "MRE_mean"}, inplace=True)
    std_df.rename(columns={"per_model_file": "model_tag", "R2": "R2_std",  "MRE": "MRE_std"}, inplace=True)

    out = pd.merge(mean_df, std_df, on="model_tag", how="inner")
    out["model_tag"] = out["model_tag"].map(_normalize_model_tag)

    for c in ["R2_mean", "R2_std", "MRE_mean", "MRE_std"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.drop_duplicates(subset=["model_tag"], keep="first").reset_index(drop=True)
    return out


# ---------------------------
# Merge into one wide table
# ---------------------------

def build_merged_table(dal: pd.DataFrame, ala: pd.DataFrame, roof: pd.DataFrame, dcpl: pd.DataFrame) -> pd.DataFrame:
    """
    Output columns (internal prefixes):
      model_tag
      DaL_R2_mean, DaL_R2_std, DaL_MRE_mean, DaL_MRE_std,
      RooflineLR_R2_mean, ...
      ALA_R2_mean, ...
      DCPL_R2_mean, ...
    """
    def prefix(df: pd.DataFrame, p: str) -> pd.DataFrame:
        df = df.copy()
        df.rename(
            columns={
                "R2_mean": f"{p}_R2_mean",
                "R2_std":  f"{p}_R2_std",
                "MRE_mean": f"{p}_MRE_mean",
                "MRE_std":  f"{p}_MRE_std",
            },
            inplace=True,
        )
        return df

    dal_p  = prefix(dal,  "DaL")
    roof_p = prefix(roof, "RooflineLR")
    ala_p  = prefix(ala,  "ALA")
    dcpl_p = prefix(dcpl, "DCPL")

    tags = sorted(
        set(dal_p["model_tag"])
        | set(roof_p["model_tag"])
        | set(ala_p["model_tag"])
        | set(dcpl_p["model_tag"])
    )
    base = pd.DataFrame({"model_tag": tags})

    out = (
        base.merge(dal_p,  on="model_tag", how="left")
            .merge(roof_p, on="model_tag", how="left")
            .merge(ala_p,  on="model_tag", how="left")
            .merge(dcpl_p, on="model_tag", how="left")
    )

    return out


# ---------------------------
# LaTeX table builders
# ---------------------------

def to_latex_metric_table(
    merged: pd.DataFrame,
    metric: str,
    out_path: Path,
    method_order_internal: list[str],
    method_display_map: dict[str, str],
    caption: str,
    label: str,
) -> None:
    """
    metric: "R2" or "MRE"
    Produces a LaTeX table where each cell is mean±std.

    method_order_internal: internal prefixes used in merged dataframe (e.g., RooflineLR)
    method_display_map: mapping for header display names (e.g., RooflineLR -> Roofline+LR)
    """
    assert metric in {"R2", "MRE"}

    # Build display rows
    rows = []
    for _, r in merged.iterrows():
        row = {"Model": _escape_latex(r["model_tag"])}
        for m in method_order_internal:
            mean = r.get(f"{m}_{metric}_mean", pd.NA)
            std  = r.get(f"{m}_{metric}_std", pd.NA)
            col_name = method_display_map.get(m, m)
            row[col_name] = _fmt_mean_std(mean, std, metric)
        rows.append(row)

    disp = pd.DataFrame(rows)

    # Determine displayed method columns (after mapping)
    display_methods = [method_display_map.get(m, m) for m in method_order_internal]

    # Bold best per row: R2 max, MRE min
    for i in range(len(disp)):
        vals = []
        for m_disp in display_methods:
            cell = disp.loc[i, m_disp]
            if cell == "-":
                vals.append((m_disp, None))
            else:
                mean_str = cell.split("$\\pm$")[0]
                try:
                    vals.append((m_disp, float(mean_str)))
                except Exception:
                    vals.append((m_disp, None))

        available = [(m, v) for (m, v) in vals if v is not None]
        if not available:
            continue

        if metric == "R2":
            best_m, _ = max(available, key=lambda t: t[1])
        else:
            best_m, _ = min(available, key=lambda t: t[1])

        disp.loc[i, best_m] = r"\textbf{" + disp.loc[i, best_m] + "}"

    # Produce LaTeX with booktabs
    latex = disp.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * len(display_methods),
    )

    table = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"{latex}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )

    out_path.write_text(table, encoding="utf-8")


def main() -> None:
    # Load standardized
    dal = load_dal(DAL_PATH)
    ala = load_ala(ALA_PATH)
    roof = load_roofline(ROOF_PATH)
    dcpl = load_dcpl(DCPL_MEAN_PATH, DCPL_STD_PATH)

    # Merge
    merged = build_merged_table(dal, ala, roof, dcpl)

    merged.to_csv(OUT_MERGED_CSV, index=False)
    print(f"[OK] Saved merged table: {OUT_MERGED_CSV}")

    # Internal prefixes (must match build_merged_table)
    methods_internal = ["DaL", "RooflineLR", "ALA", "DCPL"]

    # Display mapping for table header
    display_map = {
        "DaL": "DaL",
        "RooflineLR": "Roofline+LR",
        "ALA": "ALA",
        "DCPL": "DCPL",
    }

    # LaTeX tables
    to_latex_metric_table(
        merged=merged,
        metric="R2",
        out_path=OUT_TEX_R2,
        method_order_internal=methods_internal,
        method_display_map=display_map,
        caption="R$^2$ comparison (mean$\\pm$std) for throughput prediction (80/20 split). Higher is better.",
        label="tab:r2_dal_roofline_ala_dcpl",
    )
    print(f"[OK] Saved LaTeX (R2): {OUT_TEX_R2}")

    to_latex_metric_table(
        merged=merged,
        metric="MRE",
        out_path=OUT_TEX_MRE,
        method_order_internal=methods_internal,
        method_display_map=display_map,
        caption="MRE (\\%) comparison (mean$\\pm$std) for throughput prediction (80/20 split). Lower is better.",
        label="tab:mre_dal_roofline_ala_dcpl",
    )
    print(f"[OK] Saved LaTeX (MRE): {OUT_TEX_MRE}")

    # Safe preview (never raises KeyError)
    print("\n=== Preview (first 10 rows; mean/std columns) ===")
    desired = ["model_tag"]
    for m in methods_internal:
        desired += [f"{m}_R2_mean", f"{m}_R2_std", f"{m}_MRE_mean", f"{m}_MRE_std"]
    show_cols = [c for c in desired if c in merged.columns]
    missing = [c for c in desired if c not in merged.columns]
    if missing:
        print("[WARN] Missing preview columns (safe to ignore):")
        print("  " + ", ".join(missing))
    print(merged[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
