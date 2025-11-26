# SHAP-based Concolic Testing for Transformers

This project extends **[PyCT](https://github.com/kupl/PyCT)** by implementing SHAP-based influence-guided concolic testing on Transformer models. SHAP values act as a priority-queue influence matrix to improve robustness evaluation and adversarial case discovery.

---

## Environment Setup  

The project environment consists of three main steps:

### 1. CVC4 Setup
This project currently requires **[CVC4](https://github.com/cvc5/cvc4)** (we have not migrated to CVC5 yet). 

> **Note:** Building CVC4 requires a **Linux system** (e.g., Ubuntu) or **WSL** on Windows.  

Run the following commands in your terminal:
```bash
git clone https://github.com/cvc5/cvc4.git
cd cvc4
contrib/get-antlr-3.4
contrib/get-sources.sh
./configure.sh --optimized
cd build
make -j$(nporc)
make check
sudo make install
```
Verify installation:
```bash
cvc4 --version
```

### 2. Create a Python 3.9 Virtual Environment
Make sure you are using Python 3.9. You can create a clean virtual environment with either Conda or pipenv (see step 3).

### 3. Install Dependencies
We currently rely on conda, and will provide a requirements.txt file in the future.

**Conda**

```bash
conda env create -f environment.yml
conda activate shap-concolic
```
## Project Layout
  ```graphql
.
├─ dnnct/ 
├─ libct/  
├─ model/ 
├─ popped_constraint_position/ 
├─ shap_value/ 
├─ shap_value_all_layer/ 
├─ utils/
├─ start_cli.py
├─ start_config.py
├─ start_launch.py
├─ .gitignore
├─ dnnct_predict_common.py
├─ shap_map_calculator.py
├─ start_test.py
├─ environment.yml
├─ README.md
└─ run_dnnct.py
  ```

### Note

- popped_constraint_position/ should exist (create it if missing).

- exp/ is generated at runtime.
---

## Running the Attack
The entrypoint is now split: `start_cli.py` parses CLI flags, `start_launch.py` schedules work, and `start_test.py` is the thin wrapper you execute.

Execute a SHAP-guided concolic test on a Transformer model (default pixel search `1,2,4,8,16,32`):

```bash
python3 start_test.py \
  --model-name transformer_fashion_mnist \
  --attack-mode shap
```
If the setup is successful, you should see logs similar to:

```markdown
self.x_test.shape: (10000, 28, 28, 1)
[DEBUG] built inputs=26, skipped=4
######################################## number of inputs: 26 #############################################
{'input_name': 'fashion_mnist_test_4', 'exp_name': 'shap_1'}
/home/tobias/soslab/incite/PyCT-Influence-Transformer/dnn_predict_common.py
./dnn_predict_common.py
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to
==================================================================================================
 input_1 (InputLayer)        [(None, 28, 28, 1)]          0         []
```
---

### CLI flags of interest

| Flag | Description |
| ---- | ----------- |
| `--model-name` | Selects which saved model (under `model/`) to attack, e.g. `transformer_fashion_mnist` or `mnist_sep_act_m6_9628`. |
| `--attack-mode` | `shap`, `random`, or `random-assign`. |
| `--pixel-search` | Comma-separated ton sequence to try per input (default `1,2,4,8,16,32`). |
| `--pixel-source` | Only used with `--attack-mode random-assign`; choose `random` for RNG pixels or `shap` to reuse the SHAP ranking. |
| `--first-n` | Upper bound on dataset indices to enqueue (default 100). Combined with resume logic to skip finished inputs. |
| `--num-process` | Number of worker processes to spawn (default 1). |
| `--force-refresh` | Regenerate experiment folders even if `exp/<model>/<queue>/<exp_name>/fashion_mnist_test_*` already exists. |
| `--timeout`, `--spawn-delay`, `--random-seed`, `--norm-01`/`--no-norm-01`, `--log-level`, `--explore-log-level`, `--solver-log-level`, `--log-file` | Miscellaneous runtime/logging knobs. See `python start_test.py --help` for full details. |

Example random baseline with multi-pixel perturbations:

```bash
python3 start_test.py \
  --model-name transformer_fashion_mnist \
  --attack-mode random \
  --pixel-search 8 \
  --num-process 4 \
  --first-n 200
```

Example SHAP-guided run that resumes from prior progress:

```bash
python3 start_test.py \
  --model-name mnist_sep_act_m6_9628 \
  --attack-mode shap \
  --pixel-search 2 \
  --first-n 50
```
