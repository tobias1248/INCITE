# test_path.py
import os
from config import PROJECT_ROOT, EXP_RESULT_DIR, SHAP_VALUE_DIR


def check_path(path, name):
    print(f"{name}: {path}")
    if os.path.exists(path):
        print(f"  ✅ Exists")
    else:
        print(f"  ❌ Not found")


if __name__ == "__main__":
    print("=== Path Check ===")
    check_path(PROJECT_ROOT, "PROJECT_ROOT")
    check_path(EXP_RESULT_DIR, "EXP_RESULT_DIR")
    check_path(SHAP_VALUE_DIR, "SHAP_VALUE_DIR")
