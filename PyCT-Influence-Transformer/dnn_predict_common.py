import itertools
import logging

import keras
import numpy as np
from dnnct.myDNN import NNModel
from dnnct.tnnDNN import NNModel as tnnNNModel

from libct.position import register_layer_number_mapping, to_Keras_layer_number

log = logging.getLogger("ct.model")


myModel = None
loaded_model_path = None


def init_model(model_path):
    global myModel, loaded_model_path
    if myModel is not None and loaded_model_path == model_path:
        return
    if loaded_model_path is not None and loaded_model_path != model_path:
        keras.backend.clear_session()
    model = keras.models.load_model(model_path)
    model.summary()
    layers = [l for l in model.layers if type(l).__name__ not in ['InputLayer','Embedding','Dropout']]
    myModel = NNModel()

    # 1: is because 1st dim of input shape of Keras model is batch size (None)
    myModel.input_shape = model.input_shape[1:]
    myLayerCount = 0
    for i, layer in enumerate(layers):

        numberOfMyLayers = myModel.addLayer(layer)
        log.info(
            "Layer %s mapped to %s internal layer(s)",
            i,
            numberOfMyLayers,
        )
        for _ in range(numberOfMyLayers):
            register_layer_number_mapping(i, myLayerCount)
            myLayerCount += 1

    log.info("Number of layers in my model: %s", len(myModel.layers))
    log.info("Number of layers in original Keras model: %s", len(layers))
    log.info("Correspondence between layers in Keras model and my model:")
    for myLayerNumber in range(myLayerCount):
        log.info(
            "My layer %s -> Keras layer %s",
            myLayerNumber,
            to_Keras_layer_number(myLayerNumber),
        )
    for myLayer in myModel.layers:
        log.debug("My model layer type: %s", type(myLayer).__name__)

    loaded_model_path = model_path

def predict(**data):
	input_shape = myModel.input_shape
	# print("[DEBUG]input_shape:", input_shape)
	iter_args = (range(dim) for dim in input_shape)
	X = np.zeros(input_shape).tolist()
	data_name_prefix = "v_"
	# print("data",data.keys())
	for i in itertools.product(*iter_args):
		if len(i) == 2:
			X[i[0]][i[1]] = data[f"{data_name_prefix}{i[0]}_{i[1]}"]
		elif len(i) == 3:
			X[i[0]][i[1]][i[2]] = data[f"{data_name_prefix}{i[0]}_{i[1]}_{i[2]}"]
		elif len(i) == 4:
			X[i[0]][i[1]][i[2]][i[3]] = data[f"{data_name_prefix}{i[0]}_{i[1]}_{i[2]}_{i[3]}"]
	

	out_val = myModel.forward(X)
	# print("[DEBUG]out_val:", out_val)
 
	# 用一顆神經元做二分類
	if len(out_val) == 1:
		if isinstance(out_val[0], list):
			if out_val[0][0]>0.5:
				ret_class = 1
			else:
				ret_class = 0
		else:
			if out_val[0] > 0.5:
				ret_class = 1
			else:
				ret_class = 0
	else:
		max_val, ret_class = out_val[0], 0
		for i,cl_val in enumerate(out_val):
			if cl_val > max_val:
				max_val, ret_class = cl_val, i

	# print("[DEBUG]predicted class:", ret_class)
	return ret_class
