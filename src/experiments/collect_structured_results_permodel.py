from __future__ import annotations
from pathlib import Path
import json
import re
import pandas as pd

RUNS_ROOT = Path("results/runs")
OUT_DIR = Path("results/structured")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# subfolder pattern:
# 2026-01-13_104433__baseline_split80_rf_light__data_llama-7b
SUBDIR_RE = re.compile(r"(?P<run_id>\d{4}-\d{2}-\d{2}_\d{6})__(?P<exp>.+?)_(?P<learner>[^_]+)__(?P<model_tag>.+)$")

# prediction file pattern:
# baseline_split80_rf_light_split80_20_Target_throughput_tokens_per_sec.csv
PRED_RE = re.compile(r"(?P<experiment>.+?)_(?P<learner>.+?)_(?P<cv>.+?)_(?P<target>.+)\.csv$")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def collect():
    rows_metrics = []
    rows_runs = []
    preds_all = []

    # parent run folders like results/runs/2026-01-13_104433
    for parent in sorted([p for p in RUNS_ROOT.iterdir() if p.is_dir()]):
        run_id = parent.name
        per_model_root = parent / "per_model"
        parent_manifest = read_json(parent / "manifest.json")

        rows_runs.append({
            "run_id": run_id,
            "run_dir": str(parent),
            **{f"meta_{k}": v for k, v in parent_manifest.items()}
        })

        if not per_model_root.exists():
            continue

        for sub in sorted([s for s in per_model_root.iterdir() if s.is_dir()]):
            m = SUBDIR_RE.match(sub.name)
            if not m:
                continue

            exp = m.group("exp")
            learner = m.group("learner")
            model_tag = m.group("model_tag")

            # summary.csv (per model)
            summ_path = sub / "summary.csv"
            if summ_path.exists():
                df_s = pd.read_csv(summ_path)
                # add identifiers
                df_s["run_id"] = run_id
                df_s["experiment_folder"] = exp
                df_s["learner_folder"] = learner
                df_s["model_tag"] = model_tag
                rows_metrics.append(df_s)

            # predictions/*.csv
            pred_dir = sub / "predictions"
            if pred_dir.exists():
                for pred_file in pred_dir.glob("*.csv"):
                    pm = PRED_RE.match(pred_file.name)
                    if not pm:
                        continue
                    target = pm.group("target")
                    cv = pm.group("cv")

                    df_p = pd.read_csv(pred_file)
                    df_p["run_id"] = run_id
                    df_p["experiment_folder"] = exp
                    df_p["learner_folder"] = learner
                    df_p["model_tag"] = model_tag
                    df_p["target_from_file"] = target
                    df_p["cv_from_file"] = cv
                    df_p["pred_file"] = str(pred_file)
                    preds_all.append(df_p)

    # concat and write outputs
    if rows_metrics:
        df_metrics = pd.concat(rows_metrics, ignore_index=True)
        df_metrics.to_csv(OUT_DIR / "metrics_long.csv", index=False)
    else:
        df_metrics = pd.DataFrame()
        df_metrics.to_csv(OUT_DIR / "metrics_long.csv", index=False)

    if preds_all:
        df_preds = pd.concat(preds_all, ignore_index=True)
        # parquet is faster and smaller; keep csv if you prefer
        df_preds.to_parquet(OUT_DIR / "predictions_long.parquet", index=False)
        df_preds.sample(min(5000, len(df_preds))).to_csv(OUT_DIR / "predictions_sample.csv", index=False)
    else:
        df_preds = pd.DataFrame()
        df_preds.to_csv(OUT_DIR / "predictions_sample.csv", index=False)

    df_runs = pd.DataFrame(rows_runs)
    df_runs.to_csv(OUT_DIR / "runs_index.csv", index=False)

    print("[DONE] Structured outputs written to:", OUT_DIR)
    print(" - metrics_long.csv")
    print(" - predictions_long.parquet (and predictions_sample.csv)")
    print(" - runs_index.csv")


if __name__ == "__main__":
    collect()
