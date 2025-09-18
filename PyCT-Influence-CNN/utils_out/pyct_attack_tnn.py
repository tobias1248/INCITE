import os
import numpy as np
from utils_out.pyct_attack_exp import get_save_dir_from_save_exp

##### Generate Inputs #####

def pyct_shap_1_to_8(model_name,ton_n_shap_list ,first_n_img,model_type='origin',delta_factor=0.75):
    from utils_out.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
    inputs = []
    for solve_order_stack in [False]:
    # for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in ton_n_shap_list:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"cnn_{delta_factor}_shap_{ton_n_shap}",
                    "save_smt": True
                }
                if model_type=="origin":
                    save_exp['exp_name']=f"cnn/shap_{ton_n_shap}"

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                }

                inputs.append(one_input)
                
    return inputs