
import keras
from dnnct.myDNN import ACTIVATIONS, NNModel, LAYERS 
import numpy as np
import sys
from keras.models import Model
from keras.layers import (
    Dense,
    Conv1D, Conv2D,
    LocallyConnected1D, LocallyConnected2D,
    Flatten,
    ELU,
    Activation,
    MaxPooling2D,
    LSTM,
    Embedding,
    BatchNormalization,
)


model = keras.models.load_model("REPLACEME")
layers = [l for l in model.layers if type(l).__name__ not in ['Dropout']]
myModel = NNModel()
for layer in layers:
	myModel.addLayer(layer)
