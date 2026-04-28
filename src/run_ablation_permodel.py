from experiments.run_ablation_split80_permodel import run_ablation_split80_permodel

if __name__ == "__main__":
    run_ablation_split80_permodel(
        per_model_dir="data/llm_pilot_data/raw_data/per_model",
        targets=[
            "Target_throughput_tokens_per_sec",
            "Target_latency_ms"
           
        ],
        out_root="results/runs",
        run_name="ablation_split80_permodel",
        test_size=0.20,
        random_state=42,
        inner_splits=5,
        gate_kind="ridge",  
        base_kind="ridge",
        inter_kind="ridge",
    )
