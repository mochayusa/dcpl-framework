# src/experiments/run_dice_split80_permodel.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from dcpl.blocks import get_blocks_relaxed
from dcpl.interactions import build_all_interactions
# from dice.framework import build_dice_features, dice_fit_predict
from dcpl.metrics import compute_metrics

from utils.io import save_predictions, save_summary, save_manifest


def _make_unique_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
        return path
    k = 1
    while True:
        candidate = Path(f"{path}_{k}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        k += 1


def run_dice_split80_permodel(
    
    per_model_dir: str | Path = "data/llm_pilot_data/raw_data/per_model",
    target: str = "Target_throughput_tokens_per_sec",
    test_size: float = 0.20,
    random_state: int = 42,
    learner_kind: str = "rf",
    include_base: bool = True,
    include_interactions: bool = True,
    results_root: str | Path = "results/runs",
    run_name: str = "dice_split80",
):
    
    from dice.framework import build_dice_features, dice_fit_predict
    per_model_dir = Path(per_model_dir)
    files = sorted(per_model_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {per_model_dir}")

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    parent_dir = _make_unique_dir(Path(results_root) / run_id)
    (parent_dir / "per_model").mkdir(parents=True, exist_ok=True)

    all_rows = []

    for csv_path in files:
        model_tag = csv_path.stem
        df = pd.read_csv(csv_path)
        if target not in df.columns:
            print(f"[SKIP] {model_tag}: missing target '{target}'")
            continue

        tag = f"{run_id}__{run_name}_{learner_kind}__{model_tag}"
        model_dir = _make_unique_dir(parent_dir / "per_model" / tag)
        (model_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (model_dir / "figures").mkdir(parents=True, exist_ok=True)

        # blocks
        X_ai, X_nonai, X_wl = get_blocks_relaxed(df)

        # interactions
        df_blocks = pd.concat([X_ai, X_nonai, X_wl], axis=1)
        inter_all = build_all_interactions(
            df_blocks,
            X_ai.columns.tolist(),
            X_nonai.columns.tolist(),
            X_wl.columns.tolist(),
        )

        # build DICE features
        X_all = build_dice_features(
            X_ai, X_nonai, X_wl,
            inter_all,
            include_base=include_base,
            include_interactions=include_interactions,
        )

        idx_train, idx_test = train_test_split(
            df.index, test_size=test_size, random_state=random_state, shuffle=True
        )

        X_tr = X_all.loc[idx_train]
        X_te = X_all.loc[idx_test]
        y_tr = df.loc[idx_train, target]
        y_te = df.loc[idx_test, target]

        y_pred = dice_fit_predict(
            X_tr, X_te, y_tr,
            learner_kind=learner_kind,
            random_state=random_state,
        )

        pred_df = pd.DataFrame({"row_index": idx_test, "y_true": y_te.values, "y_pred": y_pred})
        for c in ["AI_model", "NonAI_gpu", "NonAI_gpu_type", "model", "gpu"]:
            if c in df.columns and c not in pred_df.columns:
                pred_df[c] = df.loc[idx_test, c].astype(str).values

        pred_df["target"] = target
        pred_df["experiment"] = run_name
        pred_df["learner"] = f"dice_{learner_kind}"
        pred_df["cv"] = "split80_20"
        pred_df["per_model_file"] = model_tag
        pred_df["dice_include_base"] = int(include_base)
        pred_df["dice_include_interactions"] = int(include_interactions)

        out_pred = save_predictions(model_dir, run_name, f"dice_{learner_kind}", "split80_20", target, pred_df)

        met = compute_metrics(y_te.values, y_pred)
        summary_row = {
            "experiment": run_name,
            "learner": f"dice_{learner_kind}",
            "cv": "split80_20",
            "target": target,
            "per_model_file": model_tag,
            "n_train": int(len(idx_train)),
            "n_test": int(len(idx_test)),
            "dice_include_base": int(include_base),
            "dice_include_interactions": int(include_interactions),
            **met,
        }
        all_rows.append(summary_row)

        out_summary = save_summary(model_dir, [summary_row])
        save_manifest(
            model_dir,
            {
                "run_id": run_id,
                "run_name": run_name,
                "data_path": str(csv_path),
                "target": target,
                "test_size": test_size,
                "random_state": random_state,
                "learner_kind": learner_kind,
                "dice": {
                    "include_base": include_base,
                    "include_interactions": include_interactions,
                },
                "artifacts": {"summary": str(out_summary), "predictions": str(out_pred)},
            },
        )

        print(f"[OK] {model_tag} -> {out_pred}")

    if all_rows:
        global_df = pd.DataFrame(all_rows)
        global_csv = parent_dir / f"permodel_split80_dice_summary_{learner_kind}.csv"
        global_df.to_csv(global_csv, index=False)

        save_manifest(
            parent_dir,
            {
                "run_id": run_id,
                "kind": "per_model_dice_split80",
                "learner_kind": learner_kind,
                "target": target,
                "global_summary": str(global_csv),
                "n_models": int(len(global_df)),
            },
        )

        print(f"[DONE] Global DICE summary saved: {global_csv}")
        print(f"[DONE] Parent run folder: {parent_dir}")

    return parent_dir
