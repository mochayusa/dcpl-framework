import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------
# Load your ablation result table
# ------------------------------
df = pd.read_csv("results/ablation_results.csv")

# Ensure proper ordering of experiments
order = [
    "M1_no_interaction",
    "M2a_add_AIxNonAI",
    "M2b_add_AIxN_and_AIxW",
    "M3_full",
]

df["experiment"] = pd.Categorical(df["experiment"], categories=order, ordered=True)

# Create output directory
outdir = Path("results/plots/ablation")
outdir.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Plotting helper
# ------------------------------
def plot_delta(metric, ylabel):
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="experiment",
        y=f"Δ{metric}",
        hue="target",
        marker="o",
        linewidth=2.5
    )
    plt.title(f"Ablation: Change in {ylabel} vs Adding Interaction Blocks")
    plt.xlabel("Experiment Stage")
    plt.ylabel(f"Δ{ylabel}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Target")
    plt.tight_layout()

    outfile = outdir / f"delta_{metric}.png"
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[SAVED] {outfile}")

# ------------------------------
# Generate all four Δ-metric plots
# ------------------------------
plot_delta("R2", "R²")
plot_delta("MAE", "MAE")
plot_delta("RMSE", "RMSE")
plot_delta("MRE", "MRE (%)")

print("\n[Done] Ablation delta plots saved.\n")
