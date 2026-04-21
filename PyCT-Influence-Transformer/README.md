# PyCT Influence Transformer

An experimental framework for concolic testing on image classifiers, built on top of PyCT, with SHAP-guided constraint prioritization and multi-stage pixel attacks.

## What is implemented
- SHAP-guided and random baselines for pixel-level concolic attacks.
- Multi-stage attack flow via `--pixel-search` (for example `1,2,4,8,16,32`).
- Per-case experiment artifacts and statistics under `exp/`.
- Constraint solving through SMT (`cvc5` is the default solver).
- Post-run aggregation via `python -m pyct.stats` (including split by status).

## Current defaults and key behavior
- Entrypoint: `python -m pyct`
- Default solver: `cvc5` (`engine/executor.py`, `libct/explore.py`)
- `--solver-run-timeout` default: `60` seconds
- `--score-alpha` is required
- `--symbolic-path-threshold` default: `8000`
- Supported attack modes: `shap`, `random`, `random-assign`, `queue`
- Supported datasets: `fashion_mnist`, `cifar10`, `mnist`

## Repository layout
- `pyct/`: canonical application entrypoints for attacks, SHAP preprocessing, and stats
- `orchestration/`: task scheduling, stage progression, multiprocessing, runners
- `tasks/`: task payload schemas, output paths, dataset-family builders
- `datasets/`: dataset adapters and tensor-to-payload conversion
- `engine/`: exploration engine wiring and model execution bootstrap
- `modeling/`: shared Keras/custom-layer loading compatibility helpers
- `explainability/`: SHAP calculation and pixel-selection utilities
- `libct/`: concolic engine, solver, recorder
- `dnnct/`: pure-Python DNN execution path
- `reporting/`: metrics aggregation over `stats.json`
- `pyct/shap.py`: SHAP map generation CLI

## Prerequisites
- Linux environment (or WSL)
- Python 3.9
- `cvc5` installed and available on `PATH`
- `uv` installed (`https://docs.astral.sh/uv/`)

Verify solver installation:
```bash
cvc5 --version
```

### Install CVC5
Choose one method:

```bash
# Ubuntu/Debian (if package is available)
sudo apt update
sudo apt install -y cvc5
```

```bash
# Build from source (official repository)
git clone https://github.com/cvc5/cvc5.git
cd cvc5
./configure.sh --auto-download --production
cd build
make -j"$(nproc)"
sudo make install
```

## Environment setup

This repository now uses `uv` as the source of truth for dependency management.
Python compatibility is declared in `pyproject.toml` as `>=3.9,<3.10`, and `.python-version` pins local development to `3.9`.

### Create the environment
```bash
uv sync
```

This creates `.venv/` automatically and installs both runtime and dev dependencies.

### Run commands inside the managed environment
```bash
uv run python -m pyct --help
uv run pytest
```

Dataset cache policy:
- By default, MNIST / Fashion-MNIST / CIFAR10 must already exist in the local Keras cache under `~/.keras/datasets`.
- To point at a different local cache, set `PYCT_KERAS_HOME=/path/to/keras-home`.
- To allow on-demand dataset downloads, set `PYCT_ALLOW_DATASET_DOWNLOAD=1`.

### Export `requirements.txt` for pip-compatible tooling
If you still need a flat `requirements.txt`, export it from the `uv` project metadata:

```bash
uv export --format requirements.txt --no-hashes -o requirements.txt
```

## Models
Current model artifacts in `model/`:
- `cifar10_concolic_transformer.h5`
- `transformer_fashion_mnist.h5`
- `transformer_fashion_mnist_two_mha.h5`
- `simple_mnist_m6_09585.h5`
- `mnist_sep_act_m6_9628.h5`

Local-only SHAP artifacts:
- `shap_value/` and `shap_value_all_layer/` are cache/artifact directories and should remain untracked.

## 1) Generate SHAP maps
Run SHAP preprocessing before SHAP-guided attacks:

```bash
python3 -m pyct.shap \
  --dataset cifar10 \
  --model-name cifar10_concolic_transformer \
  --first-n 100 \
  --background-per-class 3 \
  --background-seed 2233 \
  --force-refresh
```

Important SHAP options:
- `--background-per-class` (default `3`)
- `--background-seed` (default `2233`)
- `--explainer-type` (`gradient` or `kernel`)

## 2) Run attacks

### SHAP-guided run
```bash
python3 -m pyct \
  --dataset cifar10 \
  --model-name cifar10_concolic_transformer \
  --attack-mode shap \
  --timeout 1800 \
  --solver-run-timeout 60 \
  --pixel-search 1 \
  --num-process 2 \
  --score-alpha 0.8 \
  --symbolic-path-threshold 2000
```

### Queue mode (FIFO constraint collection)
```bash
python3 -m pyct \
  --dataset cifar10 \
  --model-name cifar10_concolic_transformer \
  --attack-mode queue \
  --timeout 1800 \
  --solver-run-timeout 60 \
  --pixel-search 1 \
  --num-process 1 \
  --score-alpha 0.8 \
  --symbolic-path-threshold 2000
```

### Random baselines
```bash
# random
python3 -m pyct --attack-mode random --dataset cifar10 --model-name cifar10_concolic_transformer --pixel-search 1 --first-n 100 --score-alpha 0.8

# random-assign (pixel source: random or shap)
python3 -m pyct --attack-mode random-assign --pixel-source random --dataset cifar10 --model-name cifar10_concolic_transformer --pixel-search 1 --first-n 100 --score-alpha 0.8
```

## CLI reference (core flags)
| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--model-name` | str | `transformer_fashion_mnist` | Target model in `model/` without `.h5` suffix |
| `--dataset` | choice | `fashion_mnist` | `fashion_mnist`, `cifar10`, `mnist` |
| `--attack-mode` | choice | `shap` | `shap`, `random`, `random-assign`, `queue` |
| `--pixel-search` | csv-int | `1,2,4,8,16,32` | Ton/stage sequence |
| `--num-process` | int | `1` | Worker processes |
| `--timeout` | int | `3600` | Per-stage timeout (seconds) |
| `--solver-run-timeout` | int | `60` | Timeout per SMT solve call (`0` disables wrapper timeout) |
| `--no-constraint-build-timeout` | flag | disabled | Disable the 30s formula-build timeout |
| `--score-alpha` | float | required | Priority score weight for path length term |
| `--symbolic-path-threshold` | int | `8000` | Disable symbolic tracking after threshold |
| `--first-n` | int | `100` | Number of dataset items starting from index 0 |
| `--random-seed` | int | `2024` | Used by random baselines |
| `--pixel-source` | choice | `random` | For `random-assign`: `random` or `shap` |
| `--spawn-delay` | float | `1.0` | Delay between spawning worker processes |
| `--force-refresh` | flag | disabled | Recompute even if outputs already exist |

## Output layout
Experiment outputs are stored under the repository root:

```text
<repo_root>/exp/<model_name>_<attack_mode>_<timeout>_<constraint_build_timeout_seconds>_<alpha_tag>_<threshold>/case_<idx>/
```

Notes:
- `alpha_tag` format is `a00`, `a05`, `a08`, `a10`, etc.
- `constraint_build_timeout_seconds` is numeric; `0` means build timeout is disabled.
- If `--solver-run-timeout > 0`, attack mode in path includes suffix, for example `shap_solver60s`.
- Output paths are resolved against the repo root, not the runtime working directory. This avoids artifacts drifting into `engine/exp/` during execution.
- Common files per case:
  - `stats.json`
  - `stats_history.jsonl` (stage snapshots)
  - `sat_inputs.npy` (when SAT inputs are recorded)

## Analyze results

### Human-readable summary
```bash
python3 -m pyct.stats --path exp/<your_experiment_dir>
```

### Split by status (`success` / `timeout` / ...)
```bash
python3 -m pyct.stats --path exp/<your_experiment_dir> --split-by-status
```

### JSON output for scripting
```bash
python3 -m pyct.stats --path exp/<your_experiment_dir> --json --split-by-status
```

## Common issues
- `cvc5: command not found`:
  - Install `cvc5` and make sure it is on `PATH`.
- Missing `--score-alpha`:
  - This flag is required by the current CLI.
- Long wall time with low solver time:
  - Usually indicates forward/constraint-generation overhead dominates.
- Inconsistent SHAP behavior:
  - Regenerate maps with `--force-refresh` and fixed background settings.
- Dataset cache missing in offline environments:
  - Pre-populate `~/.keras/datasets`, or set `PYCT_KERAS_HOME` to a local cache. Use `PYCT_ALLOW_DATASET_DOWNLOAD=1` only when network downloads are acceptable.

## Quick sanity checklist
- `cvc5 --version` works
- Correct Python env is activated
- SHAP maps generated for the target model/dataset
- Attack command includes required `--score-alpha`
- Output directory appears under `exp/` with expected naming
