import time
from multiprocessing import Process


# model_name="transformer_fashion_mnist"
model_name="transformer_fashion_mnist_two_mha"


NUM_PROCESS = 1
TIMEOUT = 3600

NORM_01 = False

if __name__ == "__main__":
    from utils.pyct_attack_exp import run_multi_attack_subprocess_wall_timeout
    from utils.pyct_attack_exp_research_question import fashion_mnist_transformer_shap
    
    model_type="transformer"
    inputs = fashion_mnist_transformer_shap(model_name, first_n_img=range(30))
    print("#"*40, f"number of inputs: {len(inputs)}", "#"*45)
    time.sleep(3)

    ########## 分派input給各個subprocesses ##########    
    all_subprocess_tasks = [[] for _ in range(NUM_PROCESS)]
    cursor = 0
    for task in inputs:    
        all_subprocess_tasks[cursor].append(task)    
       
        cursor+=1
        if cursor == NUM_PROCESS:
            cursor = 0


    running_processes = []
    for sub_tasks in all_subprocess_tasks:
        if len(sub_tasks) > 0:
            p = Process(target=run_multi_attack_subprocess_wall_timeout, args=(sub_tasks, TIMEOUT, NORM_01,model_type))
            p.start()
            running_processes.append(p)
            time.sleep(1) # subprocess start 的間隔時間
       
    for p in running_processes:
        p.join()

    print('done')
