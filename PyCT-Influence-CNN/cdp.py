from __future__ import annotations
import sys

import functools
import shap
from keras import Model
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import InputLayer
from typing import Tuple
import numpy as np
import os
import numpy as np
import numpy
from keras.models import load_model
from numpy.typing import ArrayLike

# Wayu
def read_all_adverserial_inputs() -> dict[str, ArrayLike]:
  # 把所有攻擊成功的 input 讀進來，存成 index -> (position -> value) 的格式
  # 指定存放.npy文件的文件夹路径
  folder_path = './cnn_priority_queue_success'

  # 获取文件夹及其子文件夹中所有.npy文件的文件路径
  npy_files = []
  for root, dirs, files in os.walk(folder_path):
      for file in files:
          if file.endswith('.npy'):
              npy_files.append(os.path.join(root, file))


  # 创建一个字典来存储加载的.npy文件
  npy_data = {}
  for i, file_path in enumerate(npy_files):
      file_name = file_path.split("/")[-2]
      npy_array = np.load(file_path)
      print(f'File: {file_name}, Shape: {npy_array.shape}')
      last_image = npy_array[-1]  # 提取最后一张图片
      print(f'Last Image Shape: {last_image.shape}')
      idx = file_name.split("_")[-1]
      npy_data[idx] = last_image

  # 检查加载的.npy文件
  for key, value in npy_data.items():
      print(f'File: {key}')
  return npy_data

# Wayu
def get_model_for_shap():
  model_path = './model/mnist_sep_act_m6_9628.h5'
  model = load_model(model_path)
  # print("input_shape", model.input_shape)
  model.summary()
  return model

# Wayu
def get_background_dataset_for_shap():
  # 把算 shap 用的 background_dataset 讀好/做好
   from tensorflow.keras.datasets.mnist import load_data            
  # Load the data and split it between train and test sets
   (x_train, y_train), (x_test, y_test) = load_data()
   x_test = x_test.astype("float32") / 255
   del x_train
   del y_train
   del y_test
   x_test = np.expand_dims(x_test, -1)
   background_dataset_for_shap = x_test[:100]
   return background_dataset_for_shap

inputs: dict[str, ArrayLike] = read_all_adverserial_inputs()
model = get_model_for_shap()
background_dataset = get_background_dataset_for_shap()

class ShapValuesComparator:
    def __init__(self, model: Model, background_dataset, input) -> None:
        self.model = model
        self.background_dataset = background_dataset
        self.input = input
        self.shap_values: dict[str, float] = dict()
        self.calculate_shap_values_for_all_neurons()

    def calculate_shap_values_for_all_neurons(self) -> None:
        trimmed_model = self.model
        transformed_background = self.background_dataset
        transformed_input = self.input
        number_of_layers = len(self.model.layers)
        for layer_number in range(0, number_of_layers):
            # print("------layer_number------", layer_number)
            # print("------transformed_background------", transformed_background.shape)
            # print("------transformed_input------", transformed_input.shape)
            self.calculate_shap_values(
                trimmed_model, transformed_background, transformed_input, layer_number)
            transformed_input = ShapValuesComparator.apply_one_layer( trimmed_model, transformed_input)
            transformed_background = ShapValuesComparator.apply_one_layer_to_dataset( trimmed_model, transformed_background)
            if layer_number == number_of_layers - 1: break
            trimmed_model = ShapValuesComparator.without_first_layer(trimmed_model)
   
    @staticmethod
    def without_first_layer(original_model: Sequential | Model) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            new_model = Sequential()
            for layer in original_model.layers[1:]:
                new_model.add(layer)
            new_model.build(original_model.layers[1].input_shape)
        elif isinstance(original_model, Model):
            # Get the output of the second layer
            new_input = original_model.layers[1].output
            # Get the output of the last layer
            new_output = original_model.layers[-1].output
            new_model = Model(inputs=new_input, outputs=new_output)
        return new_model

    @staticmethod
    def apply_one_layer_to_dataset( original_model: Sequential | Model, dataset: np.ndarray) -> np.ndarray:
        model_with_only_first_layer = ShapValuesComparator.get_model_with_only_first_layer(
            original_model)
        model_with_only_first_layer.build(original_model.layers[0].input_shape)
        new_dataset = model_with_only_first_layer.predict(dataset)
        # new_dataset = np.array([model_with_only_first_layer.predict(data) for data in dataset ])
        return new_dataset
    
    @staticmethod
    def apply_one_layer( original_model: Sequential | Model, original_input: np.ndarray) -> np.ndarray:
        return ShapValuesComparator.get_model_with_only_first_layer(original_model).predict(original_input)

    @staticmethod
    def get_model_with_only_first_layer(original_model: Sequential | Model) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            model_with_only_first_layer = Sequential(original_model.layers[:1])
        elif isinstance(original_model, Model):
            model_with_only_first_layer = Model(
                inputs=original_model.input, outputs=original_model.layers[0].output)
        return model_with_only_first_layer
    
    def calculate_shap_values(self, model, background_dataset, input, layer_number: int) -> None:
        modelInputAndOutput = (model.layers[0].input, model.layers[-1].output)
        print("------input.shape------", input.shape)
        print("------background_dataset.shape------", background_dataset.shape)
        explainer = shap.GradientExplainer(model, background_dataset) 
        
        shap_values = explainer.shap_values(input) 
        for indices, shap_value in np.ndenumerate(shap_values):
            # print("------indices------", indices)
            indices = indices[2:] # remove the first two dimensions
            # print("------indices[2:]------", indices)
            self.shap_values[ShapValuesComparator.get_position_key(
                layer_number - 1, indices)] = shap_value
        
    def get_shap_value(self, layer_number: int, indices: tuple[int, ...]) -> float:
        # print("------self.shap_values positionkeys------", ShapValuesComparator.get_position_key(layer_number, indices))
        # print("------self.shap_values------")
        # for key, value in list(self.shap_values.items())[:10]:
        #     print(key, value)
        # print("---layer_number---", layer_number)
        # print("---indices---", indices)
        
        number_of_layers = len(self.model.layers)
        if layer_number == number_of_layers - 1:
            return float('inf')
        return self.shap_values[ShapValuesComparator.get_position_key(layer_number, indices)]

    @staticmethod
    def get_position_key(layer_number: int, indices: tuple[int, ...]) -> str:
        key = str(layer_number)
        # if layer_number == 1:
        #     print("------key------", key)
        for i in indices:
            key += "_" + str(i)
        # if layer_number == 1:
        #     print("------key after + indices------", key)
        return key
    def print_shap_values(self) -> None:
      for key in self.shap_values:
        print(key, self.shap_values[key])

import functools

class Cdp_calculator:
  def __init__(self, model, background_dataset):
    self.model = model
    self.background_dataset = background_dataset
    self.number_of_layers = len(model.layers)
    self.layer_shapes: dict[int, tuple[int, ...]] = self.get_layer_output_shapes()

  def get_position_key(layer: int, indices: tuple[int, ...]) -> str:
    return f'{layer}_' + '_'.join([str(i) for i in indices])

  def get_layer_output_shapes(self) -> dict[int, tuple[int, ...]]:
    layer_shapes: dict[int, tuple[int,]] = dict()
    layer_shapes[-1] = self.model.input_shape[1:]
    for i in range(self.number_of_layers):
      layer_shapes[i] = self.model.layers[i].output_shape[1:]
    return layer_shapes

  def calculate_shap_for_all_neurons(self, input: dict[str, float]):
    return ShapValuesComparator(self.model, self.background_dataset, np.array([input])).shap_values

  def get_cdp(self, input: dict[str, float], critical_threshold: float=2) -> dict[str, bool]:
    relevance: dict[str, float] = self.calculate_shap_for_all_neurons(input)
    cdp: dict[str, bool] = self.calculate_cdp(relevance, critical_threshold)
    return cdp

  def calculate_cdp(self, relevances: dict[str, float], critical_threshold: float) -> dict[str, bool]:
    cdp: dict[str, bool] = dict()
    for l in range(-1, self.number_of_layers-1):
      shape: tuple[int, ...] = self.layer_shapes[l]
      print("layer ", l, "shape", shape)
      indices_keys: list[tuple[int, ...]] = Cdp_calculator.get_indices_keys_by_shape(shape)

      sortedRelevances: list[float] = sorted(
          [(indices_key, relevances[f'{l}_{indices_key}']) for indices_key in indices_keys],
          key=lambda positionedRelevance: positionedRelevance[1],
          reverse=True
          )
      total_importance = 0
      for (indices_key, relevance) in sortedRelevances:
        if total_importance < critical_threshold and relevance > 0:
          total_importance += relevance
          cdp[f"{l}_{indices_key}"] = True
        else:
          break
      if total_importance < critical_threshold:
        raise Exception(f"Threshold too high. should be at most {total_importance}")
    return cdp


  # ex1. (2,2) => [(0,0), (0,1), (1,0), (1,1)]
  # ex2. (1,2,3) => [(0,0,0), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,1,2)]
  def get_indices_keys_by_shape(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    ranges = [range(size) for size in shape]
    return functools.reduce(lambda xs, ys: [ f'{x}_{y}' for x in xs for y in ys], ranges)

  def union_cdps(self, cdps: dict[int, dict[str, bool]]) -> dict[str, float]:
    # get the frequencies
    number_of_cdps = len(cdps.keys())
    cdp_union: dict[str, float] = dict()
    for l in range(-1, self.number_of_layers-1):
      shape = self.layer_shapes[l]
      indices_keys = self.get_indices_keys_by_shape(shape)
      for indices_key in indices_keys:
        position_key = f'{l}_{indices_key}'
        count = 0
        for cdp in cdp.values():
          if cdp[position_key]:
            count += 1
        weight = count / number_of_cdps
        cdp_union[position_key] = weight
    return cdp_union

  def to_abstract_cdp(cdp_union: dict[str, float], threshold: float) -> dict[str, bool]:
    abstract_cdp: dict[str, bool] = dict()
    for position_key in cdp_union.keys():
      abstract_cdp[position_key] = cdp_union[position_key] > threshold
    return abstract_cdp

  def split_cdp_union_to_abstract_cdps(cdp_union: dict[str, float], number_of_abstract_cdps: int = 9) -> list[dict[str, bool]]:
    return [Cdp_calculator.to_abstract_cdp(cdp_union, i/(number_of_abstract_cdps+1)) for i in range(1, number_of_abstract_cdps)]

# cdp_calculator = Cdp_calculator(model, background_dataset)
# cdps: dict[int, dict[str, bool]] = dict()
# for idx in inputs.keys():
#   print("================ file ", idx, " ================")
#   input: ArrayLike = inputs[idx]
#   cdps[idx] = cdp_calculator.get_cdp(input)
# print(cdps)

# print("attack input", inputs['0'].shape, type(inputs['0']) )
from utils_out.dataset import MnistDataset
print("attack input", inputs['0'])
mnist_dataset = MnistDataset()
_, _, input_for_shap, background_dataset_for_shap = mnist_dataset.get_mnist_test_data(idx=6)
# print("input", input_for_shap.shape, type(input_for_shap))
# print("attack background", background_dataset_for_shap.shape, type(background_dataset_for_shap))
# print("background", background_dataset_for_shap.shape, type(background_dataset_for_shap))
# print("input", input_for_shap)
# cdp_calculator = Cdp_calculator(model, background_dataset_for_shap)
# cdp = cdp_calculator.get_cdp(input_for_shap)
# print(cdp.keys())

cdp_calculator = Cdp_calculator(model, background_dataset_for_shap)
cdp = cdp_calculator.get_cdp(inputs['0'],0.43278539553284645)
print(cdp.keys())