# SHAP-based Concolic Testing for Transformers

This project extends **[PyCT](https://github.com/kupl/PyCT)** by implementing SHAP-based influence-guided concolic testing on Transformer models. SHAP values act as a priority-queue influence matrix to improve robustness evaluation and adversarial case discovery.

---

## Environment Setup  

The project environment consists of three main steps:

### 1. CVC4 Setup
This project requires **[CVC4](https://github.com/CVC4/CVC4)**, commit **[d1f3225e26b9d64f065048885053392b10994e71](https://github.com/cvc5/cvc5/blob/d1f3225e26b9d64f065048885053392b10994e71/INSTALL.md)**.  

⚠️ **Note:** Compiling CVC4 requires a **Linux system** (e.g., Ubuntu) or **WSL** on Windows.  

Build with Python bindings:
```bash
./contrib/get-antlr-3.4
./configure.sh --language-bindings=python --python3
cd <build_dir>
make -j$(nproc)
```
Ensure the CLI is on PATH:

```bash
cvc4 --version
```
And let Python find the bindings (add to ~/.bashrc):
```bash
export PYTHONPATH={path-to-CVC4-build}/src/bindings/python:$PYTHONPATH
```
### 2. Create a Python 3.9 Virtual Environment
Make sure you are using Python 3.9. You can create a clean virtual environment with either Conda or pipenv (see step 3).

### 3. Install Dependencies
We currently rely on conda, and will provide a requirements.txt file in the future.

**Conda**

```bash
conda env create -f environments.yml
conda activate shap-concolic
```
## Project Layout (key folders)
  ```graphql
.
├─ dnnct/ 
├─ libct/  
├─ model/ 
├─ popped_constraint_position/ 
├─ shap_value/ 
├─ shap_value_all_layer/ 
└─ utils/ 
  ```

### Note

- popped_constraint_position/ should exist (create it if missing).

- exp/ is generated at runtime.

- Ensure CVC4 is correctly built with Python bindings and visible to both the shell (cvc4) and Python (PYTHONPATH).

---

## Running the Attack
Execute the SHAP-guided concolic test on a Transformer model:

```bash
python3 dnnct_transformer_multi.py
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


