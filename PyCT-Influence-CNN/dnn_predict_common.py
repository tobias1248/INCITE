import keras
from dnnct.myDNN import NNModel 
import numpy as np
import itertools

from libct.position import register_layer_number_mapping, to_Keras_layer_number



myModel = None
def init_model(model_path):
	global myModel
	model = keras.models.load_model(model_path)
	model.summary()
	layers = [l for l in model.layers if type(l).__name__ not in ['Dropout']]
	myModel = NNModel()

	# 1: is because 1st dim of input shape of Keras model is batch size (None)
	myModel.input_shape = model.input_shape[1:]
	myLayerCount = 0
	for i, layer in enumerate(layers):
		numberOfMyLayers = myModel.addLayer(layer)
		# maintain the mapping of layers between Keras model and my model
		for j in range(numberOfMyLayers):
			register_layer_number_mapping(i, myLayerCount)
			myLayerCount += 1
	model.summary()
	print('Number of layers in my model:', len(myModel.layers))
	print('Number of layers in original Keras model:', len(layers))
	print('Correspondence between layers in Keras model and my model:')
	for myLayerNumber in range(myLayerCount):
		print('My: ', myLayerNumber, 'Keras: ', to_Keras_layer_number(myLayerNumber))
	for myLayer in myModel.layers:
		print(type(myLayer))	

myInput = None
def init_input(input):
	global myInput
	myInput = input

def predict_2(**weights):
	# TODO: make a new model with the weights
	# TODO: run the globally defined input through the model
	pass

def predict(**data):
	# cast the flattened input dict 'data' back to the original shape and save to X
	input_shape = myModel.input_shape

	iter_args = (range(dim) for dim in input_shape)
	X = np.zeros(input_shape).tolist()
	data_name_prefix = "v_"
	for i in itertools.product(*iter_args):
		if len(i) == 2:
			X[i[0]][i[1]] = data[f"{data_name_prefix}{i[0]}_{i[1]}"]
		elif len(i) == 3:
			X[i[0]][i[1]][i[2]] = data[f"{data_name_prefix}{i[0]}_{i[1]}_{i[2]}"]
		elif len(i) == 4:
			X[i[0]][i[1]][i[2]][i[3]] = data[f"{data_name_prefix}{i[0]}_{i[1]}_{i[2]}_{i[3]}"]
	# calculate the output of the model
	
	out_val = myModel.forward(X)
 
	# 用一顆神經元做二分類
	if len(out_val) == 1:
		if out_val[0] > 0.5:
			ret_class = 1
		else:
			ret_class = 0
	else:
		max_val, ret_class = out_val[0], 0
		for i,cl_val in enumerate(out_val):
			if cl_val > max_val:
				max_val, ret_class = cl_val, i

	print("[DEBUG]predicted class:", ret_class)
	return ret_class