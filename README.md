# DCPL Framework

Dataset-driven experimentation framework for baseline models, DCPL, and DICE.

The framework now separates:
- experiment logic in `src/`
- dataset schemas in `configs/schemas/`
- dataset files in `data/`

This lets you evaluate different datasets and redefine AI, Non-AI, and Workload blocks without editing Python source.

## Core Idea

Each dataset is described by a schema YAML file. The schema defines:
- target columns
- AI feature columns
- Non-AI feature columns
- Workload feature columns
- alias mappings from dataset-specific names to canonical names
- categorical levels for one-hot expansion

The default schema for the current project is:
- [`configs/schemas/llm_pilot.yaml`](configs/schemas/llm_pilot.yaml)

## Expected Data Layout

The main runners expect a directory of per-model CSV files:

```text
data/my_dataset/per_model/
├── model_a.csv
├── model_b.csv
└── model_c.csv
```

Each file should contain:
- the target column you want to predict
- the columns referenced by the schema, either directly or through aliases

## Add A New Dataset

1. Create a new data directory, for example `data/my_dataset/per_model/`.
2. Put one CSV per model or subgroup into that directory.
3. Copy [`configs/schemas/example_custom.yaml`](configs/schemas/example_custom.yaml) to a new schema file.
4. Edit the schema so its `blocks`, `targets`, `aliases`, and `categorical_levels` match your dataset.
5. Run the CLI with `--per-model-dir` and `--schema`.

## Schema Format

```yaml
name: my_dataset

targets:
  - target_runtime_ms

blocks:
  ai:
    - model_params
    - model_layers
  nonai:
    - gpu_memory_gb
    - gpu_bandwidth_gbps
  workload:
    - prompt_tokens
    - batch_size

aliases:
  model_params:
    - AI_model_n_parameters
  prompt_tokens:
    - Workload_n_input_tokens

categorical_levels:
  model_type:
    - llama
    - mistral
```

Notes:
- `blocks.ai`, `blocks.nonai`, and `blocks.workload` define the features used by the framework.
- `aliases` let one canonical feature read from one or more dataset-specific columns.
- `categorical_levels` controls one-hot feature creation for source categorical columns such as `model_type`.

## Run Baselines

Run all baselines:

```bash
python src/project_main.py baseline all 30 \
  --per-model-dir data/llm_pilot_data/raw_data/per_model \
  --schema configs/schemas/llm_pilot.yaml \
  --target Target_throughput_tokens_per_sec
```

Run one baseline on a custom dataset:

```bash
python src/project_main.py baseline ridge 10 \
  --per-model-dir data/my_dataset/per_model \
  --schema configs/schemas/my_dataset.yaml \
  --target target_runtime_ms
```

## Run DCPL

```bash
python src/project_main.py dcpl 30 \
  --per-model-dir data/llm_pilot_data/raw_data/per_model \
  --schema configs/schemas/llm_pilot.yaml \
  --target Target_throughput_tokens_per_sec \
  --gate-kind ridge \
  --inner-splits 5
```

Custom dataset:

```bash
python src/project_main.py dcpl 10 \
  --per-model-dir data/my_dataset/per_model \
  --schema configs/schemas/my_dataset.yaml \
  --target target_runtime_ms
```

## Run DICE

```bash
python src/project_dice_main.py rf 30 \
  --per-model-dir data/llm_pilot_data/raw_data/per_model \
  --schema configs/schemas/llm_pilot.yaml \
  --target Target_throughput_tokens_per_sec
```

Ablation-style run:

```bash
python src/project_dice_main.py ridge 10 \
  --per-model-dir data/my_dataset/per_model \
  --schema configs/schemas/my_dataset.yaml \
  --target target_runtime_ms \
  --no-interactions
```

## Supported Dataset Formats

- `.csv`
- `.parquet` for single-dataset loaders in `src/experiments/common.py`

The per-model batch runners currently scan `*.csv` files inside `--per-model-dir`.

## Practical Rules

- Keep one schema file per dataset family.
- Treat AI, Non-AI, and Workload definitions as experimental design choices, not source code constants.
- Prefer aliases when the dataset has different naming but the same concept.
- Create a new schema when the conceptual meaning of the blocks changes.
- Keep targets explicit in the schema even if the CLI still takes `--target`.

## Key Files

- [`src/dcpl/schema.py`](src/dcpl/schema.py)
- [`src/dcpl/blocks.py`](src/dcpl/blocks.py)
- [`src/project_main.py`](src/project_main.py)
- [`src/project_dice_main.py`](src/project_dice_main.py)
- [`configs/schemas/llm_pilot.yaml`](configs/schemas/llm_pilot.yaml)
- [`configs/schemas/example_custom.yaml`](configs/schemas/example_custom.yaml)

## Backward Compatibility

Legacy scripts that call `get_blocks()` directly still work because the default schema remains `llm_pilot.yaml`. The new CLI path is schema-driven; for a new dataset, pass a new schema file instead of editing `src/dcpl/blocks.py`.
