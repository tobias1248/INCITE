# PyCT Influence Transformer

An experimental framework for concolic testing on image classifiers, built on top of PyCT, with SHAP-guided constraint prioritization and multi-stage pixel attacks.

## What is implemented
- SHAP-guided and random baselines for pixel-level concolic attacks.
- Multi-stage attack flow via `--pixel-search` (for example `1,2,4,8,16,32`).
- Per-case experiment artifacts and statistics under `exp/`.
- Constraint solving through SMT (`cvc5` is the default solver).
- Post-run aggregation via `statistic.py` (including split by status).

## Current defaults and key behavior
- Entrypoint: `start_test.py`
- Default solver: `cvc5` (`run_dnnct.py`, `libct/explore.py`)
- `--solver-run-timeout` default: `60` seconds
- `--score-alpha` is required
- `--symbolic-path-threshold` default: `8000`
- Supported attack modes: `shap`, `random`, `random-assign`, `queue`
- Supported datasets: `fashion_mnist`, `cifar10`, `mnist`

## Repository layout
- `start_test.py`: main CLI wrapper
- `start_cli.py`: argument parsing and logging setup
- `start_launch.py`: task scheduling, stage progression, multiprocessing
- `run_dnnct.py`: model loading + exploration engine wiring
- `libct/`: concolic engine, solver, recorder
- `dnnct/`: pure-Python DNN execution path
- `utils/experiment_task_specs.py`: task generation + output directory naming
- `statistic.py`: metrics aggregation over `stats.json`
- `shap_map_calculator.py`: SHAP map generation pipeline

## Prerequisites
- Linux environment (or WSL)
- Python 3.9
- `cvc5` installed and available on `PATH`

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

### Option A: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate shap-concolic
```

### Option B: pip/venv (for other package managers)
A `requirements.txt` is provided and mirrors the `pip:` section in `environment.yml`.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Regenerate `requirements.txt` from `environment.yml`
If you update the `pip:` section in `environment.yml`, regenerate with:

```bash
python3 - <<'PY'
from pathlib import Path

lines = Path('environment.yml').read_text(encoding='utf-8').splitlines()
packages = []
in_pip = False

for raw in lines:
    s = raw.strip()
    if s == '- pip:':
        in_pip = True
        continue
    if in_pip:
        if s.startswith('- '):
            pkg = s[2:].strip()
            if pkg:
                packages.append(pkg)

Path('requirements.txt').write_text('\n'.join(packages) + '\n', encoding='utf-8')
print(f'wrote {len(packages)} packages to requirements.txt')
PY
```

## Models
Current model artifacts in `model/`:
- `cifar10_concolic_transformer.h5`
- `transformer_fashion_mnist.h5`
- `transformer_fashion_mnist_two_mha.h5`
- `simple_mnist_m6_09585.h5`
- `mnist_sep_act_m6_9628.h5`

## 1) Generate SHAP maps
Run SHAP preprocessing before SHAP-guided attacks:

```bash
python3 shap_map_calculator.py \
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
python3 start_test.py \
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
python3 start_test.py \
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
python3 start_test.py --attack-mode random --dataset cifar10 --model-name cifar10_concolic_transformer --pixel-search 1 --first-n 100 --score-alpha 0.8

# random-assign (pixel source: random or shap)
python3 start_test.py --attack-mode random-assign --pixel-source random --dataset cifar10 --model-name cifar10_concolic_transformer --pixel-search 1 --first-n 100 --score-alpha 0.8
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
Experiment outputs are stored under:

```text
exp/<model_name>_<attack_mode>_<timeout>_<alpha_tag>_<threshold>/case_<idx>/
```

Notes:
- `alpha_tag` format is `a00`, `a05`, `a08`, `a10`, etc.
- If `--solver-run-timeout > 0`, attack mode in path includes suffix, for example `shap_solver60s`.
- Common files per case:
  - `stats.json`
  - `stats_history.jsonl` (stage snapshots)
  - `sat_inputs.npy` (when SAT inputs are recorded)

## Analyze results

### Human-readable summary
```bash
python3 statistic.py --path exp/<your_experiment_dir>
```

### Split by status (`success` / `timeout` / ...)
```bash
python3 statistic.py --path exp/<your_experiment_dir> --split-by-status
```

### JSON output for scripting
```bash
python3 statistic.py --path exp/<your_experiment_dir> --json --split-by-status
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

## Quick sanity checklist
- `cvc5 --version` works
- Correct Python env is activated
- SHAP maps generated for the target model/dataset
- Attack command includes required `--score-alpha`
- Output directory appears under `exp/` with expected naming
