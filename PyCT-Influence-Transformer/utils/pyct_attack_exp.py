import os
import json
import numpy as np
import queue
import traceback

# all of Generate Inputs functions in this file are only for model "mnist_sep_act_m6_9628"

def get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False):
    if only_first_forward:
        save_dir = os.path.join("exp", model_name + "_only_first_forward", s_or_q, save_exp['exp_name'], save_exp['input_name'])
    else:        
        save_dir = os.path.join("exp", model_name, s_or_q, save_exp['exp_name'], save_exp['input_name'])
    return save_dir


def run_multi_attack_subprocess_wall_timeout(args, timeout, norm,model_type="cnn"):
    import run_dnnct
    
    for one_input in args:
        print(one_input['save_exp'])
        
        result = run_dnnct.run(
            **one_input, norm=norm,
            max_iter=0, total_timeout=timeout, single_timeout=timeout, timeout=timeout
        )
        
        recorder = result[1]

def run_multi_attack_subprocess_wall_timeout_shap(args):
    from shapInfl import ShapValuesComparator

    for one_input in args:
        ShapValuesComparator(
                **one_input
            )
def run_multi_attack_subprocess_wall_timeout_task_queue(task_queue, hierarchical_input, timeout, norm):
    import run_dnnct
    
    while True:
        try:
            atk_feature_num, one_input = task_queue.get(timeout=1)  # 從佇列中取出任務，等待 1 秒
            exp_name = one_input['save_exp']['exp_name']
            input_name = one_input['save_exp']['input_name']
            model_name = one_input['model_name']
            q_or_s = None
            
            if one_input['solve_order_stack'] is True:
                q_or_s = 'stack'
            elif one_input['solve_order_stack'] is False:
                q_or_s = 'queue'
                
            print(one_input['save_exp'])
            
            tmp = f"{model_name} {q_or_s} {exp_name} {input_name}\n"
            with open(f'./log/{model_name}.log', 'a') as f:
                f.write(tmp)
            
            
            result = run_dnnct.run(
                **one_input, norm=norm,
                max_iter=0, total_timeout=timeout, single_timeout=timeout, timeout=timeout
            )
            
            recorder = result[1]
                                    
            if recorder.attack_label is not None:
                # 攻擊成功
                print("#" * 80)
                print(f"攻擊成功，參數如下：")
                print(one_input['save_exp'])
                print(q_or_s)
                print("#" * 80)
            else:
                # 攻擊失敗 加入下一個input到 task_queue                
                next_input_dict = hierarchical_input[q_or_s][input_name]['next_input_dict']
                if atk_feature_num in next_input_dict:                    
                    # 如果有下一個可以跑的atk_feature_num
                    task_queue.put(next_input_dict[atk_feature_num])
                    
        except queue.Empty:
            break  # 如果佇列空了，結束子進程
        
        except Exception as e:
            print("#"*50, "Exception", "#"*50)
            traceback.print_stack()
            print("#" * 100)
            

def run_multi_attack_subprocess_cpu_timeout(args, cpu_timeout):
    import run_dnnct
    import multiprocessing
    
    for one_input in args:
        print(one_input['save_exp'])
        
        def run():
            import resource
            # 設置 CPU 時間限制為 timeout 秒
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_timeout, cpu_timeout))
            
            wall_time_timeout = cpu_timeout*10
            result = run_dnnct.run(
                **one_input, norm=True,
                max_iter=0, total_timeout=wall_time_timeout, single_timeout=wall_time_timeout, timeout=wall_time_timeout
            )
            recorder = result[0]
            
        process = multiprocessing.Process(target=run)
        process.start()
        process.join()
        
        
            
        
##### Generate Inputs #####
# exp test shap 1 - idx 18, 261, 352, 420, 443, 559 will attack succesfully
# but pyct can only attack 18
def pyct_shap_1_test_20_3tak(model_name):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP and hard image index
    test_shap_pixel_sorted = np.load('./shap_value/mnist_sep_act_m6_9628/mnist_sort_shap_pixel.npy')
    
    ton_n_shap = 1
    s_or_q = "queue"
    if s_or_q == "queue":
        solve_order_stack = False
    elif s_or_q == "stack":
        solve_order_stack = True

    inputs = []
    for idx in list(range(20)) + [261, 352, 420, 443, 559]:
        save_exp = {
            "input_name": f"mnist_test_{idx}",
            "exp_name": f"test/shap_{ton_n_shap}"
        }

        save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q)
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

# exp test shap 1, only attack hard images
def pyct_shap_1_test(model_name, first_n_img=500):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP and hard image index
    test_shap_pixel_sorted = np.load('./shap_value/mnist_sep_act_m6_9628/mnist_sort_shap_pixel.npy')
    is_hard_img = np.load('./exp_result/is_hard_img.npy')
    

    inputs = []

    for solve_order_stack in [False]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1]:
            
            for idx, is_hard in zip(range(first_n_img), is_hard_img):
                if not is_hard:
                    # 該張圖不是困難的就跳過
                    continue

                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"test/shap_{ton_n_shap}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q)
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

# exp 1
def pyct_shap_3_5_10_hard(model_name, first_n_img):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP and hard image index
    test_shap_pixel_sorted = np.load('./shap_value/mnist_sep_act_m6_9628/mnist_sort_shap_pixel.npy')
    is_hard_img = np.load('./exp_result/is_hard_img.npy')
    

    inputs = []

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [3, 5, 10]:
            
            for idx, is_hard in zip(range(first_n_img), is_hard_img):
                if not is_hard:
                    # 該張圖不是困難的就跳過
                    continue

                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"shap_{ton_n_shap}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q)
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


# exp 2 only use queue for SHAP 1,4,8,16,32
def pyct_shap_1_4_8_16_32(model_name, first_n_img):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load('./shap_value/mnist_sep_act_m6_9628/mnist_sort_shap_pixel.npy')
    
    inputs = []

    for solve_order_stack in [False]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"shap_{ton_n_shap}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q)
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


# exp 3 only use queue for random 1,4,8,16,32
def pyct_random_1_4_8_16_32(model_name, first_n_img):
    from utils.dataset import MnistDataset
    from utils.gen_random_pixel_location import mnist_test_data_10000
    
    mnist_dataset = MnistDataset()        
    random_pixels = mnist_test_data_10000()
    
    inputs = []

    for solve_order_stack in [False]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"random_{ton_n}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = random_pixels[idx, :ton_n].tolist()
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


### copy exp2, set upper and lower bound for smt variable ###
def pyct_shap_1_4_8_16_32_limitp(model_name, first_n_img, limit_p):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load('./shap_value/mnist_sep_act_m6_9628/mnist_sort_shap_pixel.npy')
    
    inputs = []

    for solve_order_stack in [False]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"limit_{limit_p}/shap_{ton_n_shap}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q)
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
                    'limit_change_percentage': limit_p,
                }

                inputs.append(one_input)
                
    return inputs

### copy exp 3, set upper and lower bound for smt variable ###
def pyct_random_1_4_8_16_32_limitp(model_name, first_n_img, limit_p):
    from utils.dataset import MnistDataset
    from utils.gen_random_pixel_location import mnist_test_data_10000
    
    mnist_dataset = MnistDataset()        
    random_pixels = mnist_test_data_10000()
    
    inputs = []

    for solve_order_stack in [False]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"limit_{limit_p}/random_{ton_n}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = random_pixels[idx, :ton_n].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                    'limit_change_percentage': limit_p,
                }

                inputs.append(one_input)
                
    return inputs


# copy limitp
def pyct_shap_1_4_8_16_32_limit(model_name, first_n_img, limit):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load('./shap_value/mnist_sep_act_m6_9628/mnist_sort_shap_pixel.npy')
    
    inputs = []

    for solve_order_stack in [False]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"limit_{limit}/shap_{ton_n_shap}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q)
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
                    'limit_change_range': limit,
                }

                inputs.append(one_input)
                
    return inputs


# copy limitp
def pyct_random_1_4_8_16_32_limit(model_name, first_n_img, limit):
    from utils.dataset import MnistDataset
    from utils.gen_random_pixel_location import mnist_test_data_10000
    
    mnist_dataset = MnistDataset()        
    random_pixels = mnist_test_data_10000()
    
    inputs = []

    for solve_order_stack in [False]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"limit_{limit}/random_{ton_n}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q)
                if os.path.exists(save_dir):
                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = random_pixels[idx, :ton_n].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                    'limit_change_range': limit,
                }

                inputs.append(one_input)
                
    return inputs


# exp 4 only use queue for SHAP 1,4,8,16,32 + only_first_forward
def pyct_shap_1_4_8_16_32_only_first_forward(model_name, first_n_img):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
    inputs = []

    for solve_order_stack in [False]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                save_exp = {
                    "input_name": f"mnist_test_{idx}",
                    "exp_name": f"shap_{ton_n_shap}"
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q)
                if os.path.exists(os.path.join(save_dir, 'stats.json')):
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
                    'only_first_forward': True,
                }

                inputs.append(one_input)
                
    return inputs
