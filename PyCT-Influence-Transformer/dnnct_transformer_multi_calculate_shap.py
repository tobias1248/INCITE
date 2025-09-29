import os
import time
from typing import List, Dict, Any
from multiprocessing import Process

# ----------------------------
# 主函數
# ----------------------------
def main():
    model_name = "transformer_fashion_mnist_two_mha"  # 你想使用的模型名稱
    NUM_PROCESS = 1      # 未使用，保留
    TIMEOUT = 3600       # 未使用，保留
    NORM_01 = False      # 未使用，保留
    first_n_img = 100

    # 匯入所需模組
    from utils.pyct_attack_exp_research_question import fashion_mnist_transformer_shap_calculate_all
    from libct.shapInfl import ShapValuesComparator

    # 取得輸入資料與模型
    inputs, model = fashion_mnist_transformer_shap_calculate_all(model_name, first_n_img=first_n_img)
    print("#" * 40, f"number of inputs: {len(inputs)}", "#" * 45)
    time.sleep(3)

    # 確保 SHAP 儲存目錄存在
    os.makedirs(f"./shap_value_all_layer/{model_name}", exist_ok=True)

    # 逐一處理每張圖片
    for input_dict in inputs:
        idx = input_dict['idx']
        shap_json_path = f"./shap_value_all_layer/{model_name}/shap_value_{idx}.json"
        shap_pre_calc = os.path.exists(shap_json_path)

        if shap_pre_calc:
            print(f"[SKIP] SHAP already exists for idx={idx}")
        else:
            print(f"[CALC] Calculating SHAP for idx={idx}")

        # 呼叫 SHAP 計算器（自動跳過已存在）
        ShapValuesComparator(
            model_path=f'./model/{model_name}.h5',
            background_dataset=input_dict['background_dataset_for_shap'],
            input=input_dict['input_for_shap'],
            idx=idx,
            shap_value_pre_calculated=shap_pre_calc,
            explainer_type="gradient"  # 或 "kernel"
        )

    print("✅ All SHAP calculations completed.")


# ----------------------------
# 入口點
# ----------------------------
if __name__ == "__main__":
    main()