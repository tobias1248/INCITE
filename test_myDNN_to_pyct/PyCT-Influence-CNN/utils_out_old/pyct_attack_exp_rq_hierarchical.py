import os
import numpy as np
import json
from utils_out.pyct_attack_exp import get_save_dir_from_save_exp

def mnist_shap_1_4_8_16_32(model_name, first_n_img):
    from utils_out.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
    inputs = []
    attacked_input = {
        'stack': [],
        'queue': [],
    }

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                input_name = f"mnist_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"shap_{ton_n_shap}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                stats_fp = os.path.join(save_dir, 'stats.json')
                if os.path.exists(stats_fp):
                    # 已經有紀錄的讀取來看有沒有攻擊成功
                    with open(stats_fp, 'r') as f:
                        stats = json.load(f)
                        meta = stats['meta']
                        atk_label = meta['attack_label']
                        if atk_label is not None:
                            attacked_input[s_or_q].append(input_name)

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
    
    # 篩選掉已經攻擊成功的case
    notyet_attack_input = []
    for one_input in inputs:
        input_name = one_input['save_exp']['input_name']
        
        if one_input['solve_order_stack']:
            s_or_q = "stack"
        else:
            s_or_q = "queue"
            
        if input_name not in attacked_input[s_or_q]:
            notyet_attack_input.append(one_input)
            
    return notyet_attack_input


def rnn_mnist_shap_1_4_8_16_32(model_name, first_n_img):
    from utils_out.dataset import RNN_MnistDataset
    mnist_dataset = RNN_MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
    inputs = []
    attacked_input = {
        'stack': [],
        'queue': [],
    }

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                input_name = f"mnist_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"shap_{ton_n_shap}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                stats_fp = os.path.join(save_dir, 'stats.json')
                if os.path.exists(stats_fp):
                    # 已經有紀錄的讀取來看有沒有攻擊成功
                    with open(stats_fp, 'r') as f:
                        stats = json.load(f)
                        meta = stats['meta']
                        atk_label = meta['attack_label']
                        if atk_label is not None:
                            attacked_input[s_or_q].append(input_name)

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
    
    # 篩選掉已經攻擊成功的case
    notyet_attack_input = []
    for one_input in inputs:
        input_name = one_input['save_exp']['input_name']
        
        if one_input['solve_order_stack']:
            s_or_q = "stack"
        else:
            s_or_q = "queue"
            
        if input_name not in attacked_input[s_or_q]:
            notyet_attack_input.append(one_input)
            
    return notyet_attack_input


def rnn_mnist_shap_1_4_8_16_32_filter_input(model_name):
    from utils_out.dataset import RNN_MnistDataset
    mnist_dataset = RNN_MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
    inputs = []
    attacked_input = {
        'stack': [],
        'queue': [],
    }
    
    filter_queue = ['mnist_test_0', 'mnist_test_14', 'mnist_test_17', 'mnist_test_25', 'mnist_test_26',
                    'mnist_test_28', 'mnist_test_30', 'mnist_test_32', 'mnist_test_35', 'mnist_test_36', 'mnist_test_39',
                    'mnist_test_40', 'mnist_test_58', 'mnist_test_59', 'mnist_test_6', 'mnist_test_60', 'mnist_test_61',
                    'mnist_test_69', 'mnist_test_70', 'mnist_test_71', 'mnist_test_72', 'mnist_test_75', 'mnist_test_76',
                    'mnist_test_77', 'mnist_test_79', 'mnist_test_8', 'mnist_test_82', 'mnist_test_83', 'mnist_test_85',
                    'mnist_test_86', 'mnist_test_87', 'mnist_test_91', 'mnist_test_94']
    
    filter_stack = ['mnist_test_0', 'mnist_test_1', 'mnist_test_10', 'mnist_test_13',
                    'mnist_test_14', 'mnist_test_16', 'mnist_test_17', 'mnist_test_19', 'mnist_test_2', 'mnist_test_20', 'mnist_test_25',
                    'mnist_test_26', 'mnist_test_27', 'mnist_test_28', 'mnist_test_29', 'mnist_test_30', 'mnist_test_32', 'mnist_test_34',
                    'mnist_test_35', 'mnist_test_36', 'mnist_test_40', 'mnist_test_41', 'mnist_test_45', 'mnist_test_57', 'mnist_test_58',
                    'mnist_test_59', 'mnist_test_6', 'mnist_test_60', 'mnist_test_61', 'mnist_test_64', 'mnist_test_69', 'mnist_test_70',
                    'mnist_test_71', 'mnist_test_72', 'mnist_test_75', 'mnist_test_76', 'mnist_test_77', 'mnist_test_78', 'mnist_test_79',
                    'mnist_test_8', 'mnist_test_80', 'mnist_test_82', 'mnist_test_83', 'mnist_test_85', 'mnist_test_86', 'mnist_test_87',
                    'mnist_test_9', 'mnist_test_90', 'mnist_test_93']


    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1,4,8,16,32]:
            
            filter_input = []
            if solve_order_stack:
                filter_input = filter_stack
            else:
                filter_input = filter_queue
            
            for input_case in filter_input:
                idx = int(input_case.split('_')[-1])
                input_name = f"mnist_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"shap_{ton_n_shap}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                stats_fp = os.path.join(save_dir, 'stats.json')
                if os.path.exists(stats_fp):
                    # 已經有紀錄的讀取來看有沒有攻擊成功
                    with open(stats_fp, 'r') as f:
                        stats = json.load(f)
                        meta = stats['meta']
                        atk_label = meta['attack_label']
                        if atk_label is not None:
                            attacked_input[s_or_q].append(input_name)

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
    
    # 篩選掉已經攻擊成功的case
    notyet_attack_input = []
    for one_input in inputs:
        input_name = one_input['save_exp']['input_name']
        
        if one_input['solve_order_stack']:
            s_or_q = "stack"
        else:
            s_or_q = "queue"
            
        if input_name not in attacked_input[s_or_q]:
            notyet_attack_input.append(one_input)
            
    return notyet_attack_input
