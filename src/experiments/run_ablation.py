import pandas as pd
from pathlib import Path
from evaluation.logo import run_logo
from dcpl.metrics import compute_metrics
from dcpl.framework import gated_blocks_and_interactions_fold_predict

def run_ablation_experiments(
    df,
    X_ai,
    X_nonai,
    X_wl,
    inter,
    targets=("throughput", "median_nttft", "median_itl"),
    inner_splits=5,
    gate_kind="rf",
):
    """
    Runs M1, M2a, M2b, M3 ablation experiments.
    Returns a dataframe with full metrics + delta relative to M1.
    """

    ablations = {
        "M1_no_interaction":   {"AIxNonAI": False, "AIxWorkload": False, "NonAIxWorkload": False},
        "M2a_add_AIxNonAI":    {"AIxNonAI": True,  "AIxWorkload": False, "NonAIxWorkload": False},
        "M2b_add_AIxN_and_AIxW":{"AIxNonAI": True,  "AIxWorkload": True,  "NonAIxWorkload": False},
        "M3_full":             {"AIxNonAI": True,  "AIxWorkload": True,  "NonAIxWorkload": True},
    }

    def filter_interactions(config):
        """Build interaction dict only containing enabled blocks."""
        return {
            name: block if config[name] else block.iloc[:, :0]  # empty block if disabled
            for name, block in inter.items()
        }

    results = []

    for target in targets:
        print(f"\n[INFO] Running ablations for target = {target}")
        m1_summary = None

        for name, config in ablations.items():
            print(f"  → {name}")

            pred_df, summary = run_logo(
                fold_fn=lambda X_ai_tr, X_ai_te,
                            X_na_tr, X_na_te,
                            X_wl_tr, X_wl_te,
                            inter_tr, inter_te,
                            y_tr, base_kind=None, inter_kind=None:
                    gated_blocks_and_interactions_fold_predict(
                        X_ai_tr, X_ai_te,
                        X_na_tr, X_na_te,
                        X_wl_tr, X_wl_te,
                        inter_tr, inter_te,
                        y_tr,
                        base_kind=None,
                        inter_kind=None,
                        inner_splits=inner_splits,
                        gate_kind=gate_kind
                    ),
                df=df,
                target=target,
                groups=df["model"],
                X_ai=X_ai,
                X_nonai=X_nonai,
                X_wl=X_wl,
                interactions=filter_interactions(config),
                model_kind="ridge"
            )

            summary["model"] = name
            summary["target"] = target
            results.append(summary)

            if name == "M1_no_interaction":
                m1_summary = summary  # baseline for computing deltas

    df_res = pd.DataFrame(results)

    # ---- Compute deltas relative to M1 ----
    out = []
    for target in targets:
        df_t = df_res[df_res["target"] == target].copy()
        m1 = df_t[df_t["model"] == "M1_no_interaction"].iloc[0]
        mre_key = "MRE" if "MRE" in df_t.columns else "MRE(%)"


        for _, row in df_t.iterrows():
            deltas = {
                "experiment": row["model"],
                "target": target,
                "R2": row["R2"],
                "ΔR2": row["R2"] - m1["R2"],
                "MAE": row["MAE"],
                "ΔMAE": row["MAE"] - m1["MAE"],
                "RMSE": row["RMSE"],
                "ΔRMSE": row["RMSE"] - m1["RMSE"],
                "MRE": row[mre_key],
                "ΔMRE": row[mre_key] - m1[mre_key],
            }
            out.append(deltas)

    return pd.DataFrame(out)
