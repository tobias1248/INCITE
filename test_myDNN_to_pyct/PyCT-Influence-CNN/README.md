# INCITE: SHAP-guided Concolic Testing for CNNs

This project is based on **[PyCT](https://github.com/kupl/PyCT)** and extends it by applying SHAP-guided concolic testing on CNN models (e.g., MNIST). The system combines SHAP feature attribution with concolic execution to generate adversarial test cases.

---
### 0. Install Conda
This project requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.  
Make sure you have `conda` available before proceeding:

```bash
conda --version

```
## Environment Setup

The project setup consists of three main steps:

⚠️ Note: Compiling CVC4 requires a Linux system (e.g., Ubuntu) or WSL on Windows.

### 1. Install CVC4

This project depends on **[CVC4](https://github.com/CVC4/CVC4)**, commit **d1f3225e26b9d64f065048885053392b10994e715**.

We provide a helper script `install_cvc4.sh` that automatically clones, builds, and installs the correct version with Python bindings.

```bash
# From the root of the repository
bash install_cvc4.sh
```

After installation, **reload your environment** so that `PYTHONPATH` is updated:

```bash
source ~/.bashrc
```

You can verify the installation with:

```bash
cvc4 --version
```

---

### 2. Create a Python 3.9 Environment

We provide a Conda environment specification (`environment.yml`).

```bash
conda env create -f environment.yml
conda activate pyct39
```

---

### 3. Project Layout

Key folders in this repository:

```plaintext
INCITE/
└── PyCT-Influence-CNN/
    ├── dnnct/  
    ├── libct/  
    ├── model/  
    ├── utils_out/  
    ├── shap_value/  
    ├── exp/               # generated at runtime (stores adversarial examples & stats.json)  
    ├── run_dnnct.py  
    ├── dnnct_cnn_multi.py  
    ├── install_cvc4.sh  
    └── environment.yml  
```


⚠️ **Note:**

* Make sure you are inside the `INCITE/PyCT-Influence-CNN/` directory when running experiments.
* The folders `exp/` and `shap_value/` will be created automatically when attacks are executed.
* If you are not already in the correct directory, please run:

  ```bash
  cd INCITE/PyCT-Influence-CNN
  ```


## Running CNN Experiments

Run the concolic attack on CNNs (MNIST example):

```bash
python dnnct_cnn_multi.py
```

If setup is correct, you should see logs like:

```plaintext
######################################## number of inputs: 200 ########################################
{'input_name': 'mnist_test_82', 'exp_name': 'shap_1'}
Model: "sequential"
...
```

Results (adversarial examples and statistics) will appear under:

* `exp/.../adv_*.jpg` → adversarial samples
* `exp/.../stats.json` → detailed solver & constraint logs

