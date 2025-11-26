

import numpy as np
import math
from itertools import product
import collections
from functools import reduce
import logging

from typing import Tuple
from libct.position import register_current_indices, register_current_layer_number, to_Keras_layer_number

from keras.layers import (
    Dense,
    Conv1D, Conv2D,
    LocallyConnected1D, LocallyConnected2D,
    Flatten,
    ELU,
    Activation,
    MaxPool2D,
    LSTM,
    Embedding,
    BatchNormalization,
    SimpleRNN,
    Add,
    ZeroPadding2D,
    GlobalAveragePooling2D
)

LAYERS = (
    Dense,
    Conv1D, Conv2D,
    LocallyConnected1D, LocallyConnected2D,
    Flatten,
    ELU,
    Activation,
    MaxPool2D,
    LSTM,
    Embedding,
    BatchNormalization,
    Add,
    ZeroPadding2D,
    GlobalAveragePooling2D
)

ACTIVATIONS = (
    'linear',
    'relu',
    'elu',
    'softplus',
    'softsign',
    'sigmoid',
    'tanh',
    'hard_sigmoid',
    'softmax',
)


debug = False


def act_tanh(x):
    if x == 0:
        return 0.0
    elif x < 0:
        return -act_tanh(-x)
    else:
        exp_x = math.exp(x)
        exp_minus_x = math.exp(-x)
        return (exp_x - exp_minus_x) / (exp_x + exp_minus_x)


def act_sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# https://stackoverflow.com/questions/17531796/find-the-dimensions-of-a-multidimensional-python-array
# return the dimension of a python list


def dim(a):  # 遞迴求list的shape
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])


# 將padding參數標準化成(top, bottom, left, right)的形式
def _normalize_padding_2d(padding):
    # Keras ZeroPadding2D padding can be int, tuple of 2 ints, or tuple of 2 tuples
    if isinstance(padding, int):  # padding is an int
        return padding, padding, padding, padding
    if isinstance(padding, (list, tuple)):
        if len(padding) == 2 and all(isinstance(x, int) for x in padding):  # padding is tuple of 2 ints
            top = bottom = padding[0]
            left = right = padding[1]
            return top, bottom, left, right
        # padding is tuple of 2 tuples ((top, bottom), (left, right))
        if len(padding) == 2 and all(isinstance(x, (list, tuple)) for x in padding):
            (top, bottom), (left, right) = padding
            return int(top), int(bottom), int(left), int(right)
    raise ValueError(f"Unsupported padding format: {padding}")

# 實現多個tensor相加


def _recursive_elementwise_sum(values):
    if not values:
        raise ValueError(
            "AddLayer.forward() requires at least one input tensor")
    first = values[0]
    if isinstance(first, list):
        length = len(first)
        for tensor in values[1:]:
            if not isinstance(tensor, list) or len(tensor) != length:
                raise ValueError(
                    "AddLayer.forward() input tensors must share the same shape")
        return [_recursive_elementwise_sum([tensor[i] for tensor in values]) for i in range(length)]
    total = first
    for tensor in values[1:]:
        total += tensor
    return total


# acivation function
def actFunc(val, type):
    if type == 'linear':
        return val
    elif type == 'relu':

        if val < 0.0:
            return 0.0
        else:
            return val
    elif type == 'softmax':
        pass
    elif type == 'sigmoid':
        return act_sigmoid(val)
    elif type == 'tanh':
        return act_tanh(val)
    elif type == 'elu':
        pass
    elif type == 'softplus':
        pass
    elif type == 'softsign':
        pass
    else:
        raise NotImplementedError()
    return 0


class ActivationLayer:
    def __init__(self, type):
        if type not in ACTIVATIONS:
            raise NotImplementedError()
        self.type = type
        self._output = None

    def forward(self, tensor_in):
        out_shape = dim(tensor_in)
        tensor_out = tensor_in
        if len(out_shape) == 1:
            if self.type == "softmax":
                denom = 0
                for idx in range(0, out_shape[0]):
                    denom = denom + math.exp(tensor_in[idx])
                for idx in range(0, out_shape[0]):
                    tensor_out[idx] = math.exp(tensor_in[idx]) / denom
            else:
                for idx in range(0, out_shape[0]):
                    tensor_out[idx] = actFunc(tensor_in[idx], self.type)
        elif len(out_shape) == 2:
            for i, j in product(range(0, out_shape[0]),
                                range(0, out_shape[1])):
                tensor_out[i][j] = actFunc(tensor_in[i][j], self.type)
        elif len(out_shape) == 3:
            for i, j, k in product(range(0, out_shape[0]),
                                   range(0, out_shape[1]),
                                   range(0, out_shape[2])):
                tensor_out[i][j][k] = actFunc(tensor_in[i][j][k], self.type)
        else:
            raise NotImplementedError()

        if debug:
            print("[DEBUG]Finish Activation Layer forwarding!!")

        # print("Output #Activations=%i" % len(tensor_out))
        # DEBUG
        self._output = tensor_out
        # print(tensor_in)
        # print(tensor_out)
        return tensor_out

    def getOutput(self):
        return self._output


class AddLayer:
    def __init__(self, input_from):
        if not isinstance(input_from, (list, tuple)) or len(input_from) < 2:
            raise ValueError("AddLayer requires at least two input sources")
        self.input_from = list(input_from)
        self.multi_input = True
        self._output = None

    def forward(self, tensors):
        if len(tensors) != len(self.input_from):
            raise ValueError(
                "AddLayer.forward() received unexpected number of inputs")
        base_dim = dim(tensors[0])
        for tensor in tensors[1:]:
            if dim(tensor) != base_dim:  # 陸續檢查每個tensor的shape是否能夠對齊
                raise ValueError(
                    "AddLayer.forward() input tensors must share the same shape")
        result = _recursive_elementwise_sum(tensors)
        self._output = result
        return result

    def getOutput(self):
        return self._output


class DenseLayer:
    def __init__(self, weights, bias, shape, activation="None"):
        self.weights = weights.astype(float)
        self.bias = bias
        self.shape = shape
        self.activation = activation
        self._output = None

    def addActivation(self, activation):
        self.activation = activation

    def forward(self, tensor_in):
        in_shape = dim(tensor_in)
        assert len(
            in_shape) == 1, "DenseLayer.forward() with non flattened input!"
        assert in_shape[0] == self.shape[1], "DenseLayer.forward(), dim. mismatching between input and weights!"
        tensor_out = self.bias.tolist()

        for out_id in range(0, self.shape[0]):
            register_current_indices((out_id,))
            # Dot operation
            for in_id in range(0, self.shape[1]):
                tensor_out[out_id] = tensor_in[in_id] * \
                    float(self.weights[out_id][in_id]) + tensor_out[out_id]
            if self.activation != "None":
                tensor_out[out_id] = actFunc(tensor_out[id], self.activation)

        if debug:
            print("[DEBUG]Finish Dense Layer forwarding!!")

        # print("Output #Activations=%i" % len(tensor_out))
        self._output = tensor_out
        return tensor_out

    def getOutput(self):
        return self._output


class Conv2DLayer:
    def __init__(self, weights, bias, shape, activation="None", stride=(1, 1), padding='valid', name=None):
        self.weights = weights.astype(float)
        self.shape = shape
        self.bias = bias
        self.padding = padding
        if isinstance(stride, (list, tuple)):
            self.stride = (int(stride[0]), int(stride[1]))
        else:
            self.stride = (int(stride), int(stride))
        self.activation = activation
        self.name = name or "Conv2D"
        self._output = None

    def addActivation(self, activation):
        self.activation = activation

    def forward(self, tensor_in):
        in_shape = dim(tensor_in)
        if in_shape[2] != self.shape[3]:
            raise ValueError(
                f"Conv2DLayer({self.name}) channel mismatch: input shape {in_shape} has {in_shape[2]} channels, "
                f"but weights expect {self.shape[3]}"
            )
        kernel_h, kernel_w, _ = self.shape[1], self.shape[2], self.shape[3]
        stride_h, stride_w = self.stride

        if self.padding.lower() == 'same':
            out_h = math.ceil(in_shape[0] / stride_h)
            out_w = math.ceil(in_shape[1] / stride_w)
            pad_along_height = max((out_h - 1) * stride_h + kernel_h - in_shape[0], 0)
            pad_along_width = max((out_w - 1) * stride_w + kernel_w - in_shape[1], 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            tensor_in = self._pad_tensor(
                tensor_in, pad_top, pad_bottom, pad_left, pad_right)
            padded_shape = dim(tensor_in)
        else:
            out_h = (in_shape[0] - kernel_h) // stride_h + 1
            out_w = (in_shape[1] - kernel_w) // stride_w + 1
            padded_shape = in_shape

        out_shape = [out_h, out_w, self.shape[0]]
        tensor_out = [[[0.0 for _ in range(out_shape[2])]
                       for _ in range(out_shape[1])]
                      for _ in range(out_shape[0])]

        for channel in range(0, out_shape[2]):
            filter_weights = self.weights[channel]
            num_row, num_col, num_depth = kernel_h, kernel_w, self.shape[3]
            for row in range(out_shape[0]):
                for col in range(out_shape[1]):
                    register_current_indices((row, col, channel))
                    tensor_out[row][col][channel] = float(self.bias[channel])
                    base_i = row * stride_h
                    base_j = col * stride_w
                    # inner product of the filter and the input image segments
                    for i, j, k in product(range(num_row),
                                           range(num_col),
                                           range(num_depth)):
                        input_i = base_i + i
                        input_j = base_j + j
                        tensor_out[row][col][channel] += tensor_in[input_i][input_j][k] * float(
                            filter_weights[i][j][k])
                    if self.activation != "None":
                        tensor_out[row][col][channel] = actFunc(
                            tensor_out[row][col][channel], self.activation)
                    # print(type(tensor_out[row][col][channel]))
            # print("Finished %i feature Map" % channel)

        if debug:
            print("[DEBUG]Finish Conv2D Layer forwarding!!")

        # print("Feature Map Shape: %ix%ix%i" % tuple(out_shape))
        self._output = tensor_out
        return tensor_out

    def getOutput(self):
        return self._output

    def _pad_tensor(self, tensor_in, top, bottom, left, right):
        h, w, c = dim(tensor_in)
        new_h = h + top + bottom
        new_w = w + left + right
        tensor_out = [[[0.0 for _ in range(c)] for _ in range(new_w)]
                      for _ in range(new_h)]
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    tensor_out[i + top][j + left][k] = tensor_in[i][j][k]
        return tensor_out


class MaxPool2DLayer:
    def __init__(self, shape, stride=1, padding='valid'):
        self.pool_size = shape
        self.stride = stride
        self.padding = padding
        self._output = None

    def forward(self, tensor_in):
        in_shape = dim(tensor_in)
        assert (len(in_shape) == 3)

        # For now, we assume stride=1 and padding='valid'
        # TODO  stride!=1 and padding!='valid'
        r, c = self.pool_size[0], self.pool_size[1]
        out_shape = [in_shape[0] // r,
                     in_shape[1] // c,
                     in_shape[2]]
        # tensor_out = np.zeros(out_shape).tolist()
        tensor_out = []
        for _ in range(out_shape[0]):
            tensor_out.append([[0.0]*out_shape[2]
                              for i in range(out_shape[1])])

        for row in range(0, out_shape[0]):
            for col in range(0, out_shape[1]):
                for depth in range(0, out_shape[2]):
                    register_current_indices((row, col, depth))
                    max_val = -10000
                    if tensor_in[row*r][col*c][depth] > max_val:
                        max_val = tensor_in[row*r][col*c][depth]
                    if tensor_in[row*r+1][col*c][depth] > max_val:
                        max_val = tensor_in[row*r+1][col*c][depth]
                    if tensor_in[row*r][col*c+1][depth] > max_val:
                        max_val = tensor_in[row*r][col*c+1][depth]
                    if tensor_in[row*r+1][col*c+1][depth] > max_val:
                        max_val = tensor_in[row*r+1][col*c+1][depth]
                    tensor_out[row][col][depth] = max_val
                    # print(type(tensor_out[row][col][depth]))
        # fix the shape of tensor_out

        if debug:
            print("[DEBUG]Finish MaxPool2D Layer forwarding!!")

        # print("Feature Map Shape: %ix%ix%i" % tuple(out_shape))
        self._output = tensor_out
        return tensor_out

    def getOutput(self):
        return self._output


class FlattenLayer:
    def __init__(self):
        self._output = None

    def forward(self, tensor_in):
        try:
            tensor_out = self._flatten(tensor_in)
        except Exception as e:
            print(e)
        self._output = tensor_out
        return tensor_out

    def _flatten(self, x):
        if isinstance(x, collections.abc.Iterable):
            return [a for i in x for a in self._flatten(i)]
        else:
            return [x]

    def getOutput(self):
        return self._output


# Zero padding for channels_last tensors
class ZeroPadding2DLayer:
    def __init__(self, padding):
        self.padding = _normalize_padding_2d(padding)
        self._output = None

    def forward(self, tensor_in):
        top, bottom, left, right = self.padding
        in_shape = dim(tensor_in)
        assert len(in_shape) == 3, "ZeroPadding2D expects 3D input [H][W][C]"
        h, w, c = in_shape
        out_h = h + top + bottom  # top-padding + bottom-padding + height
        out_w = w + left + right
        tensor_out = [[[0.0 for _ in range(c)] for _ in range(out_w)]
                      for _ in range(out_h)]
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    tensor_out[i + top][j + left][k] = tensor_in[i][j][k]
        self._output = tensor_out
        return tensor_out

    def getOutput(self):
        return self._output

# 只做推論重現前向傳播結果 把keras層保存的參數傳進來（假設學習率1e-3）


class BatchNormalization2DLayer:
    def __init__(self, gamma, beta, moving_mean, moving_var, epsilon=1e-3):
        self.gamma = gamma
        self.beta = beta
        self.moving_mean = moving_mean
        self.moving_var = moving_var
        self.epsilon = epsilon
        self._output = None

    def forward(self, tensor_in):
        in_shape = dim(tensor_in)
        assert len(
            in_shape) == 3, "BatchNormalization2D expects 3D input [H][W][C]"
        h, w, c = in_shape  # height, width, channels
        tensor_out = [[[0.0 for _ in range(c)] for _ in range(w)]
                      for _ in range(h)]
        for i in range(h):
            for j in range(w):
                for ch in range(c):
                    x = tensor_in[i][j][ch]
                    norm = (
                        x - self.moving_mean[ch]) / math.sqrt(self.moving_var[ch] + self.epsilon)
                    tensor_out[i][j][ch] = self.gamma[ch] * \
                        norm + self.beta[ch]
        self._output = tensor_out
        return tensor_out

    def getOutput(self):
        return self._output


# 把每個 feature map 壓成單一數值，維持通道維度，後面接Dense 做分類
class GlobalAveragePooling2DLayer:
    def __init__(self):
        self._output = None

    def forward(self, tensor_in):
        in_shape = dim(tensor_in)
        assert len(
            in_shape) == 3, "GlobalAveragePooling2D expects 3D input [H][W][C]"
        h, w, c = in_shape
        out = [0.0 for _ in range(c)]
        total = h * w
        for ch in range(c):
            acc = 0.0
            for i in range(h):
                for j in range(w):
                    acc += tensor_in[i][j][ch]
            out[ch] = acc / total
        self._output = out
        return out

    def getOutput(self):
        return self._output


# Define SimpleRNN class
class SimpleRNNLayer:
    def __init__(self, input_dim, weights, activation='tanh'):
        self.input_dim = input_dim
        assert activation in ('linear', "tanh")
        self.activation = activation
        self.units = weights[0].shape[1]

        self.w_xh, self.w_hh, self.b_h = (w.tolist() for w in weights)

        # Initialize weights
#         self.w_hh = [[0 for i in range(units)] for j in range(units)]
#         self.w_xh = [[0 for i in range(units)] for j in range(input_shape)]

        # Initialize biases
#         self.b_h = [0 for i in range(units)]

        # Initialize hidden state
        self.h = [0 for i in range(self.units)]
        self._output = None

    def call(self, x):
        # Update hidden state
        curr_h = self.h.copy()
        for i in range(self.units):
            h_i = 0
            for j in range(self.units):
                h_i += curr_h[j] * self.w_hh[j][i]

            for j in range(self.input_dim):
                h_i += x[j] * self.w_xh[j][i]

            h_i += self.b_h[i]

            if self.activation == 'tanh':
                self.h[i] = act_tanh(h_i)
            else:
                self.h[i] = h_i

        # Return hidden state
        return self.h

    def init_state(self):
        self.h = [0 for i in range(self.units)]

    def forward(self, X):
        self.init_state()
        for i in range(len(X)):
            output_h = self.call(X[i])
        self._output = output_h

        if debug:
            print("[DEBUG]Finish SimpleRNN Layer forwarding!!")

        return output_h

    def getOutput(self):
        return self._output


class LSTMLayer:
    def __init__(self, input_size, weights):
        self.input_size = input_size
        W, U, b = (w for w in weights)
        self.hidden_size = int(W.shape[1] / 4)

        # load weights
        self.W_i = W[:, :self.hidden_size].tolist()
        self.W_f = W[:, self.hidden_size: self.hidden_size * 2].tolist()
        self.W_c = W[:, self.hidden_size * 2: self.hidden_size * 3].tolist()
        self.W_o = W[:, self.hidden_size * 3:].tolist()

        self.U_i = U[:, :self.hidden_size].tolist()
        self.U_f = U[:, self.hidden_size: self.hidden_size * 2].tolist()
        self.U_c = U[:, self.hidden_size * 2: self.hidden_size * 3].tolist()
        self.U_o = U[:, self.hidden_size * 3:].tolist()

        # load biases
        self.b_i = b[:self.hidden_size].tolist()
        self.b_f = b[self.hidden_size: self.hidden_size * 2].tolist()
        self.b_c = b[self.hidden_size * 2: self.hidden_size * 3].tolist()
        self.b_o = b[self.hidden_size * 3:].tolist()

    def forward(self, X):
        # init states
        h0 = np.zeros(self.hidden_size).astype('float32').tolist()
        c0 = np.zeros(self.hidden_size).astype('float32').tolist()

        for i in range(len(X)):
            h0, c0 = self.step(X[i], h0, c0)

        return h0

    def step(self, x, h, c):
        i = [0.0] * self.hidden_size
        f = [0.0] * self.hidden_size
        o = [0.0] * self.hidden_size
        g = [0.0] * self.hidden_size

        for j in range(self.hidden_size):
            for k in range(self.input_size):
                i[j] += x[k] * self.W_i[k][j]
                f[j] += x[k] * self.W_f[k][j]
                o[j] += x[k] * self.W_o[k][j]
                g[j] += x[k] * self.W_c[k][j]

            for l in range(self.hidden_size):
                i[j] += h[l] * self.U_i[l][j]
                f[j] += h[l] * self.U_f[l][j]
                o[j] += h[l] * self.U_o[l][j]
                g[j] += h[l] * self.U_c[l][j]

            i[j] += self.b_i[j]
            f[j] += self.b_f[j]
            o[j] += self.b_o[j]
            g[j] += self.b_c[j]

            i[j] = act_sigmoid(i[j])
            f[j] = act_sigmoid(f[j])
            o[j] = act_sigmoid(o[j])
            g[j] = act_tanh(g[j])

        new_c = [0.0] * self.hidden_size
        new_h = [0.0] * self.hidden_size

        for j in range(self.hidden_size):
            new_c[j] = f[j] * c[j] + i[j] * g[j]
            new_h[j] = o[j] * act_tanh(new_c[j])

        return new_h, new_c


class NNModel:
    def __init__(self):
        self.layers = []
        self.originalLayerNumbers = dict()
        self.input_shape = None
        self.my_layer_keys = []
        self.keras_to_cache_key = {}
        self.input_layer_names = []
        self.multiple_inputs = False
        self.layer_type_counter = {}

    def register_input_names(self, names):
        self.input_layer_names = list(names or [])
        if not self.input_layer_names:
            return
        if len(self.input_layer_names) > 1:
            self.multiple_inputs = True
            raise NotImplementedError(
                "Multiple input tensors are not supported yet.")
        self.keras_to_cache_key[self.input_layer_names[0]] = "layer_input"

    def _resolve_cache_key(self, keras_name):
        if keras_name not in self.keras_to_cache_key:
            raise KeyError(f"Unknown inbound source: {keras_name}")
        return self.keras_to_cache_key[keras_name]

    def _register_cache_key(self, keras_name, cache_key):
        if keras_name:
            self.keras_to_cache_key[keras_name] = cache_key

    def _append_layer(self, layer_obj):
        type_name = type(layer_obj).__name__
        count = self.layer_type_counter.get(type_name, 0)
        cache_key = f"{type_name}_{count}"
        self.layer_type_counter[type_name] = count + 1
        self.layers.append(layer_obj)
        self.my_layer_keys.append(cache_key)
        return cache_key

    def forward(self, tensor_in):
        logging.info("DNN start forwarding")
        cache = {"layer_input": tensor_in}
        x = tensor_in
        for i, layer in enumerate(self.layers):
            register_current_layer_number(to_Keras_layer_number(i))
            layer_key = self.my_layer_keys[i]
            if hasattr(layer, "input_from") and layer.input_from:
                inputs = [cache[name] for name in layer.input_from]
                if getattr(layer, "multi_input", False):
                    x = layer.forward(inputs)
                else:
                    x = layer.forward(inputs[0])
            else:
                x = layer.forward(x)
            cache[layer_key] = x

        logging.info("DNN finish forwarding")
        return x

    def getLayOutput(self, idx):
        if idx >= len(self.layers):
            return None
        else:
            return self.layers[idx].getOutput()

    # turn one Keras layer into my layers and add them to my model
    # @returns the number of my layers added
    def addLayer(self, layer, inbound_names=None) -> int:
        inbound_names = inbound_names or []
        keras_name = getattr(layer, "name", f"layer_{len(self.layers)}")
        resolved_inbounds = [self._resolve_cache_key(
            name) for name in inbound_names] if inbound_names else []
        created = 0

        if isinstance(layer, Conv2D):
            raw_weights = layer.get_weights()
            weights = raw_weights[0].transpose(3, 0, 1, 2)
            if len(raw_weights) > 1:
                biases = raw_weights[1]
            else:
                biases = np.zeros(weights.shape[0], dtype=weights.dtype)
            activation = layer.get_config()['activation']
            strides = layer.get_config().get('strides', (1, 1))
            padding = layer.get_config().get('padding', 'valid')
            conv_layer = Conv2DLayer(
                weights, biases, weights.shape, stride=strides,
                padding=padding, name=keras_name)
            if resolved_inbounds:
                conv_layer.input_from = resolved_inbounds
            self._append_layer(conv_layer)
            created += 1
            activation_key = self._append_layer(ActivationLayer(activation))
            created += 1
            self._register_cache_key(keras_name, activation_key)
        elif isinstance(layer, Dense):
            raw_weights = layer.get_weights()
            weights = raw_weights[0].transpose()
            if len(raw_weights) > 1:
                biases = raw_weights[1]
            else:
                biases = np.zeros(weights.shape[0], dtype=weights.dtype)
            activation = layer.get_config()['activation']

            dense_layer = DenseLayer(weights, biases, weights.shape)
            if resolved_inbounds:
                dense_layer.input_from = resolved_inbounds
            self._append_layer(dense_layer)
            created += 1
            activation_key = self._append_layer(ActivationLayer(activation))
            created += 1
            self._register_cache_key(keras_name, activation_key)
        elif isinstance(layer, MaxPool2D):
            pool_size = layer.get_config()['pool_size']
            maxpool_layer = MaxPool2DLayer(pool_size)
            if resolved_inbounds:
                maxpool_layer.input_from = resolved_inbounds
            key = self._append_layer(maxpool_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, ZeroPadding2D):
            padding = layer.get_config()['padding']
            zp_layer = ZeroPadding2DLayer(padding)
            if resolved_inbounds:
                zp_layer.input_from = resolved_inbounds
            key = self._append_layer(zp_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, BatchNormalization):
            gamma, beta, moving_mean, moving_var = (
                arr.tolist() for arr in layer.get_weights())
            epsilon = layer.epsilon
            bn_layer = BatchNormalization2DLayer(
                gamma, beta, moving_mean, moving_var, epsilon)
            if resolved_inbounds:
                bn_layer.input_from = resolved_inbounds
            key = self._append_layer(bn_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, Flatten):
            flatten_layer = FlattenLayer()
            if resolved_inbounds:
                flatten_layer.input_from = resolved_inbounds
            key = self._append_layer(flatten_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, Activation):
            activation = layer.get_config()['activation']
            act_layer = ActivationLayer(activation)
            if resolved_inbounds:
                act_layer.input_from = resolved_inbounds
            key = self._append_layer(act_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, SimpleRNN):
            input_dim = layer.input_shape[-1]
            activation = layer.get_config()['activation']
            rnn_layer = SimpleRNNLayer(
                input_dim, weights=layer.get_weights(), activation=activation)
            if resolved_inbounds:
                rnn_layer.input_from = resolved_inbounds
            key = self._append_layer(rnn_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, LSTM):
            input_dim = layer.input_shape[-1]
            lstm_layer = LSTMLayer(input_dim, weights=layer.get_weights())
            if resolved_inbounds:
                lstm_layer.input_from = resolved_inbounds
            key = self._append_layer(lstm_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, Add):
            key = self._append_layer(AddLayer(resolved_inbounds))
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, GlobalAveragePooling2D):
            gap_layer = GlobalAveragePooling2DLayer()
            if resolved_inbounds:
                gap_layer.input_from = resolved_inbounds
            key = self._append_layer(gap_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        else:
            raise NotImplementedError()

        return created
