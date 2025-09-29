import time
from multiprocessing import Process

def main():
    # ==============================
    # ❗ 不該寫死的參數（建議移到 config 或命令列參數）
    # ==============================
    model_name = "transformer_fashion_mnist"  # ← 應從外部傳入
    NUM_PROCESS = 1                                   # ← 可用 argparse
    TIMEOUT = 3600                                    # ← 可用 argparse
    NORM_01 = False                                   # ← 可用 argparse
    MODEL_TYPE = "transformer"                        # ← 應與 model_name 一致或自動推斷
    FIRST_N_IMG_RANGE = range(30)                     # ← 應為整數（如 10），非 range

    # ==============================
    # 主程式邏輯（不變）
    # ==============================
    from utils.pyct_attack_exp import run_multi_attack_subprocess_wall_timeout
    from utils.pyct_attack_exp_research_question import fashion_mnist_transformer_shap

    inputs = fashion_mnist_transformer_shap(model_name, first_n_img=FIRST_N_IMG_RANGE)
    print("#" * 40, f"number of inputs: {len(inputs)}", "#" * 45)
    time.sleep(3)

    ########## 分派 input 給各個 subprocesses ##########
    all_subprocess_tasks = [[] for _ in range(NUM_PROCESS)]
    cursor = 0
    for task in inputs:
        all_subprocess_tasks[cursor].append(task)
        cursor += 1
        if cursor == NUM_PROCESS:
            cursor = 0

    running_processes = []
    for sub_tasks in all_subprocess_tasks:
        if len(sub_tasks) > 0:
            p = Process(
                target=run_multi_attack_subprocess_wall_timeout,
                args=(sub_tasks, TIMEOUT, NORM_01, MODEL_TYPE)
            )
            p.start()
            running_processes.append(p)
            time.sleep(1)  # subprocess start 的間隔時間

    for p in running_processes:
        p.join()

    print('done')


if __name__ == "__main__":
    main()