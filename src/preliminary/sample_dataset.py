import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

DATA = "data/llm_pilot_data/final_data/full_raw_data_800k_clean.csv"

OUTPUT = "data/llm_pilot_data/preliminary/preliminary_sample_100k.csv"

TARGET = "Target_throughput_tokens_per_sec"

MODEL_COL = "AI_model"

TOTAL_SAMPLE = 100_000
MIN_RATIO_PER_MODEL = 0.10

RANDOM_SEED = 42


# ============================================================
# Sampling function
# ============================================================

def stratified_model_sampling(df: pd.DataFrame) -> pd.DataFrame:

    rng = np.random.default_rng(RANDOM_SEED)

    grouped = df.groupby(MODEL_COL)

    samples = []

    # --------------------------------------------------------
    # Step 1: guarantee minimum 10% per model
    # --------------------------------------------------------

    for model, g in grouped:

        n_total = len(g)

        n_min = int(np.ceil(n_total * MIN_RATIO_PER_MODEL))

        n_take = min(n_min, len(g))

        sampled = g.sample(n=n_take, random_state=RANDOM_SEED)

        samples.append(sampled)

    sampled_df = pd.concat(samples)

    remaining = TOTAL_SAMPLE - len(sampled_df)

    if remaining <= 0:
        return sampled_df.sample(n=TOTAL_SAMPLE, random_state=RANDOM_SEED)

    # --------------------------------------------------------
    # Step 2: fill remaining samples randomly
    # --------------------------------------------------------

    remaining_pool = df.drop(sampled_df.index)

    extra = remaining_pool.sample(
        n=min(remaining, len(remaining_pool)),
        random_state=RANDOM_SEED
    )

    sampled_df = pd.concat([sampled_df, extra])

    return sampled_df.sample(frac=1, random_state=RANDOM_SEED)


# ============================================================
# Main
# ============================================================

def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA)

    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} not found in dataset")

    if MODEL_COL not in df.columns:
        raise ValueError(f"{MODEL_COL} not found in dataset")

    print("Dataset size:", len(df))

    sampled = stratified_model_sampling(df)

    print("Sample size:", len(sampled))

    print("\nSamples per model:")

    print(sampled[MODEL_COL].value_counts())

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

    sampled.to_csv(OUTPUT, index=False)

    print("\nSaved to:", OUTPUT)


if __name__ == "__main__":
    main()