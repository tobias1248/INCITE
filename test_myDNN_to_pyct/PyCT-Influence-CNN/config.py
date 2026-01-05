# config.py
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXP_RESULT_DIR = os.path.join(PROJECT_ROOT, "exp_result")
SHAP_VALUE_DIR = os.path.join(PROJECT_ROOT, "utils_out", "shap_value")
