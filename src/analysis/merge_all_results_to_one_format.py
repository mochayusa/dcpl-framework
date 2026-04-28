#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FULL MERGE SCRIPT (Baseline + DCPL Gates + DaL raw .txt logs) -> ONE unified CSV.

What this script guarantees for DaL rows:
- method          = "baseline_DaL"
- per_model_file  = normalised to match baseline/DCPL (e.g., data_EleutherAI_gpt-neox-20b)
- cv              = DEFAULT_CV (because DaL logs usually don't store CV)
- target          = parsed from filename if present (Target_...), else DEFAULT_TARGET
- seed            = DEFAULT_SEED (unless you later enhance parsing)
- n_train/n_test  = parsed from header "N_train=... N_test=..." or filename "_NtrainX_NtestY"
                   (NOT from the "..._800_0-30_..." token, which is not N_train)

Baseline CSV (your schema):
  experiment,learner,cv,target,per_model_file,n_train,n_test,R2,MAE,RMSE,MRE,iteration,seed,run_name_iter,parent_dir,summary_csv

Gate CSV (your schema):
  experiment,learner,gate_kind,cv,target,per_model_file,n_train,n_test,R2,MAE,RMSE,MRE,run_folder,summary_path
(or similar; this supports missing 'iteration' by generating it)

DaL log example path:
  results/DaL/raw_results/data_bigcode_starcoder__final_for_dal/
    data_bigcode_starcoder__final_for_dal_Ntrain13477_Ntest20000_02-04_19-37-31.txt

DaL log example filename containing target/budget/time:
  data_EleutherAI_gpt-neox-20b_Target_throughput_tokens_per_sec_800_0-30_01-16_09-18-26.txt
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG (EDIT ONCE)
# ============================================================
BASELINE_CSV = Path(
    "results/runs/20260130_104106/30x_baselines/"
    "baseline_split80__MULTI_BASELINES__30x_base42/"
    "baseline_split80_permodel_ALL_learners_ALL_runs_stacked.csv"
)

GATE_CSV = Path(
    "results/runs/20260130_104106/30x_dcpl/"
    "dcpl_split80__multirun_30x_gates_summary/"
    "dcpl_gate_summaries_ALL_runs_stacked.csv"
)

DAL_LOG_DIR = Path("results/DaL/raw_results")   # <-- change if needed
DAL_GLOB = "**/*.txt"

OUT_DIR = Path("results/scott_knott_merged_input")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MERGED_CSV = OUT_DIR / "merged_baseline_dcpl_gate_dal.csv"

# Defaults used to fill missing DaL fields (and any missing fields overall)
DEFAULT_CV = "split80_20"
DEFAULT_TARGET = "Target_throughput_tokens_per_sec"
DEFAULT_SEED = -1

# DaL parsing behaviour
DAL_DEPTH_TO_USE = 1
DAL_MAX_RUNS = 30  # keep first 30 runs; set None to keep all

# Fairness cap: keep first N iterations per (LLM, method)
CAP_RUNS_PER_METHOD = 30  # set None to disable


# ============================================================
# Helpers
# ============================================================
def to_numeric_safe(x):
    return pd.to_numeric(x, errors="coerce")


def normalise_per_model_file(x: str) -> str:
    """
    Baseline/DCPL per_model_file might be a filename or a path.
    Convert to a stable ID. Example:
      'data_bigcode_starcoder.csv' -> 'data_bigcode_starcoder'
      '/path/to/data_llama-7b.csv' -> 'data_llama-7b'
    """
    s = str(x)
    return Path(s).stem


# ---------------- DaL: normalise dataset id / cv / target / n_train n_test ----------------
def dal_llm_from_path(p: Path) -> str:
    """
    Normalise DaL dataset id to match baseline/DCPL.

    Examples:
      data_bigcode_starcoder__final_for_dal_Ntrain... -> data_bigcode_starcoder
      data_EleutherAI_gpt-neox-20b_Target_throughput_tokens_per_sec_800_... -> data_EleutherAI_gpt-neox-20b
    """
    s = p.stem  # filename without extension

    # 1) data_... then __
    m = re.search(r"(data_[A-Za-z0-9\-_]+?)(?=__)", s)
    if m:
        return m.group(1)

    # 2) prefix before _Target_
    m = re.search(r"^(data_[A-Za-z0-9\-_]+?)(?=_Target_)", s)
    if m:
        return m.group(1)

    # 3) just "data_...." at beginning
    m = re.search(r"^(data_[A-Za-z0-9\-_]+)", s)
    if m:
        return m.group(1)

    return s


def dal_target_from_path(p: Path) -> str:
    s = p.stem
    m = re.search(r"(Target_[A-Za-z0-9\-_]+)", s)
    return m.group(1) if m else DEFAULT_TARGET


def dal_cv_from_path(_: Path) -> str:
    # DaL logs usually don't include CV; enforce your experiment CV
    return DEFAULT_CV


def dal_ntrain_ntest_from_path_or_text(p: Path, text: str):
    """
    Return (n_train, n_test)
    Priority:
      1) header: N_train=... N_test=...
      2) filename: _Ntrain123_Ntest456
      else NaN
    """
    n_train = np.nan
    n_test = np.nan

    m = re.search(r"N_train=(\d+)\s+N_test=(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))

    m = re.search(r"_Ntrain(\d+)_Ntest(\d+)", p.name)
    if m:
        return int(m.group(1)), int(m.group(2))

    # IMPORTANT: do NOT treat "..._800_0-30_..." as N_train.
    return n_train, n_test


# ============================================================
# 1) Baseline -> unified
# ============================================================
def load_baseline_unified(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {path}")

    df = pd.read_csv(path)

    required = ["experiment", "learner", "cv", "target", "per_model_file",
                "n_train", "n_test", "R2", "MAE", "RMSE", "MRE", "iteration"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Baseline CSV missing column: {c}. Found: {list(df.columns)}")

    out = pd.DataFrame()
    out["experiment"] = df["experiment"].astype(str)
    out["method"] = "baseline_" + df["learner"].astype(str)
    out["cv"] = df["cv"].astype(str).fillna(DEFAULT_CV)
    out["target"] = df["target"].astype(str).fillna(DEFAULT_TARGET)
    out["per_model_file"] = df["per_model_file"].map(normalise_per_model_file).astype(str)

    out["n_train"] = to_numeric_safe(df["n_train"])
    out["n_test"] = to_numeric_safe(df["n_test"])

    for m in ["R2", "MAE", "RMSE", "MRE"]:
        out[m] = to_numeric_safe(df[m])

    out["iteration"] = to_numeric_safe(df["iteration"])
    out["seed"] = to_numeric_safe(df["seed"]) if "seed" in df.columns else DEFAULT_SEED

    out["source"] = "baseline"
    out["run_folder"] = ""
    out["summary_path"] = df["summary_csv"] if "summary_csv" in df.columns else ""
    out["parent_dir"] = df["parent_dir"] if "parent_dir" in df.columns else ""

    out["dal_depth"] = np.nan  # keep column consistent

    return out


# ============================================================
# 2) DCPL gate -> unified
# ============================================================
def load_gate_unified(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Gate CSV not found: {path}")

    df = pd.read_csv(path)

    required = ["experiment", "learner", "cv", "target", "per_model_file",
                "n_train", "n_test", "R2", "MAE", "RMSE", "MRE"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Gate CSV missing column: {c}. Found: {list(df.columns)}")

    out = pd.DataFrame()
    out["experiment"] = df["experiment"].astype(str)

    if "gate_kind" in df.columns:
        out["method"] = "dcpl_gate_" + df["gate_kind"].astype(str)
    else:
        out["method"] = "dcpl_gate_" + df["learner"].astype(str)

    out["cv"] = df["cv"].astype(str).fillna(DEFAULT_CV)
    out["target"] = df["target"].astype(str).fillna(DEFAULT_TARGET)
    out["per_model_file"] = df["per_model_file"].map(normalise_per_model_file).astype(str)

    out["n_train"] = to_numeric_safe(df["n_train"])
    out["n_test"] = to_numeric_safe(df["n_test"])

    for m in ["R2", "MAE", "RMSE", "MRE"]:
        out[m] = to_numeric_safe(df[m])

    # seed (if absent, try parse from experiment)
    if "seed" in df.columns:
        out["seed"] = to_numeric_safe(df["seed"])
    else:
        out["seed"] = to_numeric_safe(df["experiment"].astype(str).str.extract(r"seed(\d+)", expand=False)).fillna(DEFAULT_SEED)

    out["source"] = "dcpl_gate"
    out["run_folder"] = df["run_folder"] if "run_folder" in df.columns else ""
    out["summary_path"] = df["summary_path"] if "summary_path" in df.columns else ""
    out["parent_dir"] = ""

    # iteration: if missing, generate stable iteration per (LLM, method)
    if "iteration" in df.columns:
        out["iteration"] = to_numeric_safe(df["iteration"])
    else:
        order_col = None
        if "run_folder" in df.columns:
            order_col = "run_folder"
        elif "summary_path" in df.columns:
            order_col = "summary_path"
        elif "experiment" in df.columns:
            order_col = "experiment"

        if order_col:
            out = out.sort_values(["per_model_file", "method", order_col])
        else:
            out = out.sort_values(["per_model_file", "method"])

        out["iteration"] = out.groupby(["per_model_file", "method"]).cumcount() + 1

    out["dal_depth"] = np.nan

    return out


# ============================================================
# 3) DaL txt logs -> unified
# ============================================================
RUN_RE = re.compile(r"^\s*Run\s+(\d+)\s*$")
METRIC_RE = re.compile(
    r"^depth(?P<depth>\d+)\s+DNN_DaL\s+(?P<metric>R2|MAE|RMSE|MRE)\s*:\s*(?P<value>[-+0-9.eE]+)"
)

def parse_one_dal_log(txt_path: Path) -> pd.DataFrame:
    llm_id = dal_llm_from_path(txt_path)
    target = dal_target_from_path(txt_path)
    cv = dal_cv_from_path(txt_path)

    raw_text = txt_path.read_text(errors="ignore")
    lines = raw_text.splitlines()

    n_train, n_test = dal_ntrain_ntest_from_path_or_text(txt_path, raw_text)

    current_run = None
    run_data = {}

    for raw in lines:
        line = raw.strip()

        mrun = RUN_RE.match(line)
        if mrun:
            current_run = int(mrun.group(1))
            run_data[current_run] = {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "MRE": np.nan}
            continue

        if current_run is None:
            continue

        mm = METRIC_RE.match(line)
        if mm:
            depth = int(mm.group("depth"))
            if depth != DAL_DEPTH_TO_USE:
                continue
            metric = mm.group("metric")
            value = float(mm.group("value"))
            run_data[current_run][metric] = value

    rows = []
    for run_id in sorted(run_data.keys()):
        metrics = run_data[run_id]
        rows.append({
            "experiment": f"DaL_depth{DAL_DEPTH_TO_USE}",
            "method": "baseline_DaL",
            "cv": cv,
            "target": target,
            "per_model_file": llm_id,
            "n_train": n_train,
            "n_test": n_test,
            "R2": metrics["R2"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MRE": metrics["MRE"],
            "iteration": run_id,
            "seed": DEFAULT_SEED,
            "source": "dal",
            "run_folder": str(txt_path),
            "summary_path": "",
            "parent_dir": "",
            "dal_depth": float(DAL_DEPTH_TO_USE),
        })

    out = pd.DataFrame(rows)

    # drop runs where all metrics are missing (parsing failed)
    out = out.dropna(subset=["R2", "MAE", "RMSE", "MRE"], how="all")

    if DAL_MAX_RUNS is not None:
        out = out[out["iteration"] <= DAL_MAX_RUNS].copy()

    return out


def load_dal_unified(log_dir: Path) -> pd.DataFrame:
    if not log_dir.exists():
        print(f"[WARN] DaL log dir not found: {log_dir} (skip DaL)")
        return pd.DataFrame()

    txts = sorted(log_dir.glob(DAL_GLOB))
    if not txts:
        print(f"[WARN] No DaL logs matched: {log_dir}/{DAL_GLOB} (skip DaL)")
        return pd.DataFrame()

    frames = []
    for p in txts:
        df_one = parse_one_dal_log(p)
        if not df_one.empty:
            frames.append(df_one)

    if not frames:
        print("[WARN] DaL logs found but none parsed into rows.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ============================================================
# Post-processing: run cap & fill defaults
# ============================================================
def cap_runs(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    df = df.sort_values(["per_model_file", "method", "iteration"])
    df["_rank"] = df.groupby(["per_model_file", "method"]).cumcount() + 1
    df = df[df["_rank"] <= cap].drop(columns=["_rank"])
    return df


def fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure schema & fill missing text columns
    for col in ["experiment", "method", "cv", "target", "per_model_file", "source", "run_folder", "summary_path", "parent_dir"]:
        if col not in df.columns:
            df[col] = ""

    df["cv"] = df["cv"].replace({None: "", np.nan: ""})
    df["target"] = df["target"].replace({None: "", np.nan: ""})

    # Fill empty strings with defaults for cv/target
    df.loc[df["cv"].astype(str).str.strip().eq(""), "cv"] = DEFAULT_CV
    df.loc[df["target"].astype(str).str.strip().eq(""), "target"] = DEFAULT_TARGET

    # Seeds
    if "seed" not in df.columns:
        df["seed"] = DEFAULT_SEED
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").fillna(DEFAULT_SEED).astype(int)

    # Numeric columns
    for col in ["n_train", "n_test", "iteration", "R2", "MAE", "RMSE", "MRE", "dal_depth"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # per_model_file normalisation (ensure stable)
    df["per_model_file"] = df["per_model_file"].map(lambda x: str(x).strip())
    df["per_model_file"] = df["per_model_file"].map(normalise_per_model_file)

    return df


# ============================================================
# MAIN
# ============================================================
def main():
    print("=== Loading baseline ===")
    baseline = load_baseline_unified(BASELINE_CSV)
    print("Baseline rows:", len(baseline),
          "| LLM:", baseline["per_model_file"].nunique(),
          "| methods:", baseline["method"].nunique())

    print("\n=== Loading DCPL gates ===")
    gate = load_gate_unified(GATE_CSV)
    print("Gate rows:", len(gate),
          "| LLM:", gate["per_model_file"].nunique(),
          "| methods:", gate["method"].nunique())

    print("\n=== Parsing DaL logs ===")
    dal = load_dal_unified(DAL_LOG_DIR)
    if dal.empty:
        print("DaL rows: 0")
    else:
        print("DaL rows:", len(dal),
              "| LLM:", dal["per_model_file"].nunique(),
              "| methods:", dal["method"].nunique())
        print("DaL method names:", sorted(dal["method"].unique())[:10])

    print("\n=== Concatenating ===")
    frames = [baseline, gate]
    if not dal.empty:
        frames.append(dal)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = fill_defaults(merged)

    if CAP_RUNS_PER_METHOD is not None:
        merged = cap_runs(merged, CAP_RUNS_PER_METHOD)

    # Save
    merged.to_csv(OUT_MERGED_CSV, index=False)
    print("\n[OK] Saved:", OUT_MERGED_CSV)

    # Sanity checks
    print("\n=== Sanity summary ===")
    print("Total rows:", len(merged))
    print("Unique LLM:", merged["per_model_file"].nunique())
    print("Unique methods:", merged["method"].nunique())
    print("\nRows by source:\n", merged["source"].value_counts())

    # Confirm DaL alignment
    if (merged["source"] == "dal").any():
        dsub = merged[merged["source"] == "dal"].copy()
        print("\nDaL sanity:")
        print("  DaL rows:", len(dsub))
        print("  DaL unique LLM:", dsub["per_model_file"].nunique())
        print("  DaL method names:", sorted(dsub["method"].unique()))
        print("  DaL cv unique:", sorted(dsub["cv"].astype(str).unique())[:10])
        print("  DaL target unique:", sorted(dsub["target"].astype(str).unique())[:10])
        print("  DaL metric non-null counts:\n", dsub[["R2", "MAE", "RMSE", "MRE"]].notna().sum())

        # show a few DaL rows
        print("\nDaL sample rows:")
        print(dsub[["per_model_file", "iteration", "R2", "MAE", "RMSE", "MRE", "cv", "target"]].head(10).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
