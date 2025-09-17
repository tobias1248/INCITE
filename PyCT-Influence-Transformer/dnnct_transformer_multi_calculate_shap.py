import time
from multiprocessing import Process


model_name="transformer_fashion_mnist"
model_name="transformer_fashion_mnist_two_mha"


NUM_PROCESS = 1
TIMEOUT = 3600
NORM_01 = False
if __name__ == "__main__":
    from utils.pyct_attack_exp_research_question import (        
       fashion_mnist_transformer_shap_calculate_all
    )
    from libct.shapInfl import ShapValuesComparator
    inputs, model = fashion_mnist_transformer_shap_calculate_all(model_name, first_n_img=100)
    print("#"*40, f"number of inputs: {len(inputs)}", "#"*45)
    time.sleep(3)

    for input in inputs:
        ShapValuesComparator(model, input['background_dataset_for_shap'], input['input_for_shap'], input['idx'],shap_value_pre_calculated=False)
    print('done')

