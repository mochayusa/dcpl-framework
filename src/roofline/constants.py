# src/roofline/constants.py

# Workload
COL_II = "Workload_n_input_tokens"
COL_OO = "Workload_n_output_tokens"
COL_BB = "Workload_num_users"

# Target
COL_TARGET = "Target_throughput_tokens_per_sec"

# Model proxy
COL_PARAMS = "AI_model_n_parameters"

# Roofline hardware knobs (present in your dataset)
COL_BW = "NonAI_gpu_memory_bandwidth"     # (expected GB/s; see note below)

# Choose ONE peak compute column (TFLOPS). FP16 tensor cores is a good default.
COL_TFLOPS = "NonAI_gpu_tflops_tc_fp16"

# If you prefer BF16 instead, switch to:
# COL_TFLOPS = "NonAI_gpu_tflops_tc_bf16"
