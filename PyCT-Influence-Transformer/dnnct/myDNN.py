
import sys
import numpy as np
import math
from itertools import product
import collections.abc
from functools import reduce
import logging
from typing import Tuple, Optional, List, Any
from libct.position import register_current_indices, register_current_layer_number, to_Keras_layer_number
from libct.utils import unwrap

from keras.layers import (
    Dense,
    Conv1D, Conv2D,
    LocallyConnected1D, LocallyConnected2D,
    Flatten,
    ELU,
    Activation,
    ReLU,
    MaxPool2D,
    MaxPooling2D,
    RandomCrop,
    RandomFlip,
    Dropout,
    ZeroPadding2D,
    LSTM,
    Embedding,
    BatchNormalization,
    LayerNormalization,
    SimpleRNN,
    MultiHeadAttention,
    Add,
    GlobalAveragePooling2D,
    GlobalAveragePooling1D,
    Reshape,
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
    'gelu',
)

log = logging.getLogger("ct.model")
debug = False


def _piecewise_linear_exp(x):
    # Segment is chosen by concrete value to avoid introducing symbolic branches.
    concrete = float(unwrap(x))
    exp_neg10 = 4.5399929762484854e-05
    exp_neg6 = 0.0024787521766663585
    exp_neg3 = 0.049787068367863944
    exp_neg1 = 0.36787944117144233
    exp_0 = 1.0
    exp_2 = 7.38905609893065
    exp_5 = 148.4131591025766

    def interp(x0, y0, x1, y1):
        slope = (y1 - y0) / (x1 - x0)
        intercept = y0 - slope * x0
        return slope * x + intercept

    if concrete <= -10.0:
        return exp_neg10
    if concrete <= -6.0:
        return interp(-10.0, exp_neg10, -6.0, exp_neg6)
    if concrete <= -3.0:
        return interp(-6.0, exp_neg6, -3.0, exp_neg3)
    if concrete <= -1.0:
        return interp(-3.0, exp_neg3, -1.0, exp_neg1)
    if concrete <= 0.0:
        return interp(-1.0, exp_neg1, 0.0, exp_0)
    if concrete <= 2.0:
        return interp(0.0, exp_0, 2.0, exp_2)
    if concrete <= 5.0:
        return interp(2.0, exp_2, 5.0, exp_5)
    return exp_5


def concolic_exp(x):
    if x < 0:
        return 1.0 / concolic_exp(-x)
    if x > 1:
        try:
            return concolic_exp(x / 2) ** 2
        except OverflowError:
            print(x, 'OverflowError')   
    a0 = 1.0
    a1 = x        #6737
    a2 = x**2 / 2 #9891
    # a3 = x**3 / 6 #9985
    # a4 = x**4 / 24
    # a5 = x**5 / 120
    # a6 = x**6 / 720
    # a7 = x**7 / 5040
    # a8 = x**8 / 40320
    # a9 = x**9 / 362880

    return a0 + a1 + a2
    # return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9

def my_exp_complex(x):
    if x < 0:
        return 1.0 / my_exp(-x)
    elif x > 1:
        return my_exp(x/2) * my_exp(x/2)
    elif (math.exp(x) >= x + 1) and (math.exp(x) <= 2*x + 1):
        try:
            return math.exp(x)
        except mathexpError:
            print(x, 'math.expError')
def my_exp(x):
    # Core default for transformer path: symbolic-safe exp approximation.
    return _piecewise_linear_exp(x)


# exp_func = math.exp
exp_func = my_exp

def act_tanh(x):
    if x == 0:
        return 0.0
    elif x >= 3:
        return 1.0
    elif x <= -3:
        return -1.0
    elif x < 0:
        return -act_tanh(-x)
    else:
        # exp_x = my_exp(x)
        # exp_minus_x = my_exp(-x)
        exp_x = exp_func(x)
        exp_minus_x = exp_func(-x)
        return (exp_x - exp_minus_x) / (exp_x + exp_minus_x)

def act_sigmoid(x):
    log.debug("act_sigmoid input=%s", x)
    if x == 0:
        log.debug("act_sigmoid midpoint -> 0.5")
        return 0.5
    if x >= 5:
        log.debug("act_sigmoid saturating to 1.0")
        return 1.0
    if x <= -5:
        log.debug("act_sigmoid saturating to 0.0")
        return 0.0
    return 1.0 / (1.0 + exp_func(-x))

def act_gelu(x):
    # Concrete-guided piecewise-linear GELU approximation for symbolic stability.
    concrete = float(unwrap(x))
    if concrete <= -3.0:
        factor = 0.0
    elif concrete >= 3.0:
        factor = 1.0
    else:
        factor = (concrete + 3.0) / 6.0
    return factor * x

def act_softmax(x):
    # Keep softmax concrete to avoid inflating symbolic expressions.
    concrete = [float(unwrap(val)) for val in x]
    max_val = max(concrete)
    exp_values = [math.exp(val - max_val) for val in concrete]
    exp_sum = sum(exp_values)
    softmax_values = [val / exp_sum for val in exp_values]
    return softmax_values

# https://stackoverflow.com/questions/17531796/find-the-dimensions-of-a-multidimensional-python-array
# return the dimension of a python list
def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])


def _recursive_elementwise_sum(values):
    if not values:
        raise ValueError("AddLayer.forward() requires at least one input tensor")
    first = values[0]
    if isinstance(first, list):
        length = len(first)
        for tensor in values[1:]:
            if not isinstance(tensor, list) or len(tensor) != length:
                raise ValueError("AddLayer.forward() input tensors must share the same shape")
        return [_recursive_elementwise_sum([tensor[i] for tensor in values]) for i in range(length)]
    total = first
    for tensor in values[1:]:
        total += tensor
    return total


# acivation function
def actFunc(val, type):
    if type=='linear':
        return val
    elif type=='relu':
        if val < 0.0:
            return 0.0
        else:
            return val
    elif type=='softmax':
        return act_softmax(val)
    elif type=='sigmoid':
        log.debug("Applying sigmoid activation to value=%s", val)
        return act_sigmoid(val)
    elif type=='tanh':
        return act_tanh(val)
    elif type=='gelu':
        return act_gelu(val)
    elif type=='elu':
        pass
    elif type=='softplus':
        pass
    elif type=='softsign':
        pass
    else:
        raise NotImplementedError(f"Unsupported activation function: {type}")
    return 0


class ActivationLayer:
    def __init__(self, type):
        if type not in ACTIVATIONS:
            raise NotImplementedError(f"Unsupported activation: {type}")
        self.type = type
        self._output = None
    def forward(self, tensor_in):
        out_shape = dim(tensor_in)
        tensor_out = tensor_in
        log.debug("ActivationLayer type=%s input_shape=%s", self.type, out_shape)
        if len(out_shape)==1:
            # print('start 1: ', self.type, tensor_in)
            if self.type=="softmax":
                # print('start 1-0 softmax')
                tensor_out = act_softmax(tensor_in)
                # raise NotImplementedError()
                # denom = 0
                # for idx in range(0, out_shape[0]):
                #     denom = denom + math.exp(tensor_in[idx])
                # for idx in range(0, out_shape[0]):
                #     tensor_out[idx] = math.exp(tensor_in[idx]) / denom
            else:
                # print('start 1-0')
                for idx in range(0, out_shape[0]):
                    # print('start 1-', idx, tensor_in[idx])
                    tensor_out[idx] = actFunc(tensor_in[idx], self.type)
                    # print('end 1-', tensor_out[idx])
            # print('end 1')
        elif len(out_shape)==2:
            # print('start 2')
            for i, j in product( range(0, out_shape[0]),
                                range(0, out_shape[1])):
                tensor_out[i][j] = actFunc(tensor_in[i][j], self.type)
            # print('end 2')
        elif len(out_shape)==3:
            # print('start 3')
            for i, j, k in product( range(0, out_shape[0]),
                                    range(0, out_shape[1]),
                                    range(0, out_shape[2])):
                tensor_out[i][j][k] = actFunc(tensor_in[i][j][k], self.type)
            # print('end 3')
        else:
            raise NotImplementedError()
        if debug:
            log.debug("[ActivationLayer] Finished forwarding %s", self.type)

        #print("Output #Activations=%i" % len(tensor_out))
        ## DEBUG
        self._output = tensor_out
        #print(tensor_in)
        #print(tensor_out)
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
            raise ValueError("AddLayer.forward() received unexpected number of inputs")
        base_dim = dim(tensors[0])
        for tensor in tensors[1:]:
            if dim(tensor) != base_dim:
                raise ValueError("AddLayer.forward() input tensors must share the same shape")
        self._output = _recursive_elementwise_sum(tensors)
        return self._output

    def getOutput(self):
        return self._output


class NoOpLayer:
    def __init__(self, name: str = "NoOp"):
        self.name = name
        self._output = None

    def forward(self, tensor_in):
        # deterministic no-op for inference-only layers (e.g., RandomFlip)
        self._output = tensor_in
        return tensor_in

    def getOutput(self):
        return self._output

class ZeroPadding2DLayer:
    def __init__(self, padding):
        # padding: (top, bottom, left, right)
        self.padding = padding
        self._output = None

    def forward(self, tensor_in):
        if isinstance(tensor_in, np.ndarray):
            tensor_in = tensor_in.tolist()
        h = len(tensor_in)
        w = len(tensor_in[0]) if h > 0 else 0
        c = len(tensor_in[0][0]) if w > 0 else 0
        top, bottom, left, right = self.padding
        new_h = h + top + bottom
        new_w = w + left + right
        tensor_out = [[[0.0 for _ in range(c)] for _ in range(new_w)] for _ in range(new_h)]
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    tensor_out[i + top][j + left][k] = tensor_in[i][j][k]
        self._output = tensor_out
        return tensor_out

    def getOutput(self):
        return self._output

class CenterCrop2DLayer:
    def __init__(self, target_h: int, target_w: int):
        self.target_h = target_h
        self.target_w = target_w
        self._output = None

    def forward(self, tensor_in):
        if isinstance(tensor_in, np.ndarray):
            tensor_in = tensor_in.tolist()
        h = len(tensor_in)
        w = len(tensor_in[0]) if h > 0 else 0
        if self.target_h is None or self.target_w is None:
            raise ValueError("CenterCrop2DLayer requires target_h and target_w.")
        if h == self.target_h and w == self.target_w:
            self._output = tensor_in
            return tensor_in
        start_h = max((h - self.target_h) // 2, 0)
        start_w = max((w - self.target_w) // 2, 0)
        tensor_out = [
            [list(tensor_in[i][j]) for j in range(start_w, start_w + self.target_w)]
            for i in range(start_h, start_h + self.target_h)
        ]
        self._output = tensor_out
        return tensor_out

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
        # assert len(in_shape)==1, "DenseLayer.forward() with non flattened input!"
        tensor_out = self.bias.tolist()

        if len(in_shape) == 1:
            assert in_shape[0]==self.shape[1], "DenseLayer.forward(), dim. mismatching between input and weights!"
            for out_id in range(0, self.shape[0]):
                register_current_indices((out_id,))
                ## Dot operation
                for in_id in range(0, self.shape[1]):
                    tensor_out[out_id] = tensor_in[in_id]*float(self.weights[out_id][in_id]) + tensor_out[out_id]
                if self.activation!="None":
                    tensor_out[out_id] = actFunc(tensor_out[out_id], self.activation)
        elif len(in_shape) == 2:
            assert in_shape[1]==self.shape[1], "DenseLayer.forward(), dim. mismatching between input and weights!"
            tensor_out = [tensor_out.copy() for _ in range(len(tensor_in))]
            for i in range(len(tensor_in)):
                for out_id in range(0, self.shape[0]):
                    register_current_indices((i,out_id))
                ## Dot operation
                    for in_id in range(0, self.shape[1]):
                        tensor_out[i][out_id] += tensor_in[i][in_id]*float(self.weights[out_id][in_id])
                    if self.activation!="None":
                        tensor_out[i][out_id] = actFunc(tensor_out[i][out_id], self.activation)

        else:
            raise ValueError("Dense dosn't support the input ")

        if debug:
            print("[DEBUG]Finish Dense Layer forwarding!!")

        #print("Output #Activations=%i" % len(tensor_out))
        self._output = tensor_out
        return tensor_out
    def getOutput(self):
        return self._output


class Conv2DLayer:
    def __init__(self, weights, bias, shape, activation="None", stride=(1, 1), padding='valid'):
        self.weights = weights.astype(float)
        self.shape = shape
        self.bias = bias
        self.padding = padding
        self.stride = stride
        self.activation = activation
        self._output = None
    def addActivation(self, activation):
        self.activation = activation
    def _pad_input(self, tensor_in, pad_h, pad_w):
        if pad_h <= 0 and pad_w <= 0:
            return tensor_in
        if isinstance(tensor_in, np.ndarray):
            tensor_in = tensor_in.tolist()
        h = len(tensor_in)
        w = len(tensor_in[0]) if h > 0 else 0
        c = len(tensor_in[0][0]) if w > 0 else 0
        new_h = h + pad_h * 2
        new_w = w + pad_w * 2
        tensor_out = [[[0.0 for _ in range(c)] for _ in range(new_w)] for _ in range(new_h)]
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    tensor_out[i + pad_h][j + pad_w][k] = tensor_in[i][j][k]
        return tensor_out
    def forward(self, tensor_in):
        stride_h, stride_w = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        filter_h, filter_w, _ = self.shape[1], self.shape[2], self.shape[3]
        if self.padding == 'same':
            pad_h = (filter_h - 1) // 2
            pad_w = (filter_w - 1) // 2
            tensor_in = self._pad_input(tensor_in, pad_h, pad_w)
        in_shape = dim(tensor_in)
        assert in_shape[2] == self.shape[3], "Conv2DLayer, channel length mismatching!"
        out_shape = [
            (in_shape[0] - filter_h) // stride_h + 1,
            (in_shape[1] - filter_w) // stride_w + 1,
            self.shape[0],
        ]
        #tensor_out = np.zeros( out_shape ).tolist()
        tensor_out = []
        for _ in range(out_shape[0]):
            tensor_out.append( [[0.0]*out_shape[2] for i in range(out_shape[1])] ) 

        for channel in range(0, out_shape[2]):
            filter_weights = self.weights[channel]
            num_row, num_col, num_depth = filter_h, filter_w, self.shape[3]
            for row in range(0, out_shape[0]):
                for col in range(0, out_shape[1]):
                    register_current_indices((row, col, channel))
                    tensor_out[row][col][channel] = float(self.bias[channel])
                    ## inner product of the filter and the input image segments
                    row_base = row * stride_h
                    col_base = col * stride_w
                    for i, j, k in product( range(row_base, row_base+num_row),
                                            range(col_base, col_base+num_col), 
                                            range(0, num_depth)):
                        tensor_out[row][col][channel] = tensor_in[i][j][k] * float(filter_weights[i-row_base][j-col_base][k]) + tensor_out[row][col][channel] 
                    if self.activation!="None":
                        tensor_out[row][col][channel] = actFunc(tensor_out[row][col][channel], self.activation)
                    #print(type(tensor_out[row][col][channel]))
            #print("Finished %i feature Map" % channel)
        
        if debug:
            print("[DEBUG]Finish Conv2D Layer forwarding!!")

        #print("Feature Map Shape: %ix%ix%i" % tuple(out_shape))
        self._output = tensor_out
        return tensor_out

    def getOutput(self):
        return self._output


class MaxPool2DLayer:
    def __init__(self, shape, stride=1, padding='valid'):
        self.pool_size = shape
        self.stride = stride
        self.padding = padding
        self._output = None
    def forward(self, tensor_in):
        in_shape = dim(tensor_in)
        assert(len(in_shape)==3)

        ## For now, we assume stride=1 and padding='valid'
        ## TODO  stride!=1 and padding!='valid'
        r, c = self.pool_size[0], self.pool_size[1]
        out_shape = [ in_shape[0] // r,
                        in_shape[1] // c,
                        in_shape[2]]
        # tensor_out = np.zeros(out_shape).tolist()
        tensor_out = []
        for _ in range(out_shape[0]):
            tensor_out.append( [[0.0]*out_shape[2] for i in range(out_shape[1])] )

        for row in range(0, out_shape[0]):
            for col in range(0, out_shape[1]):
                for depth in range(0, out_shape[2]):
                    register_current_indices((row, col, depth))
                    max_val = -10000
                    if tensor_in[row*r  ][col*c  ][depth] > max_val:
                        max_val = tensor_in[row*r  ][col*c  ][depth]
                    if tensor_in[row*r+1][col*c  ][depth] > max_val:
                        max_val = tensor_in[row*r+1][col*c  ][depth]
                    if tensor_in[row*r  ][col*c+1][depth] > max_val:
                        max_val = tensor_in[row*r  ][col*c+1][depth]
                    if tensor_in[row*r+1][col*c+1][depth] > max_val:
                        max_val = tensor_in[row*r+1][col*c+1][depth]
                    tensor_out[row][col][depth] = max_val
                    #print(type(tensor_out[row][col][depth]))
        ## fix the shape of tensor_out

        if debug:
            print("[DEBUG]Finish MaxPool2D Layer forwarding!!")

        #print("Feature Map Shape: %ix%ix%i" % tuple(out_shape))
        self._output = tensor_out
        return tensor_out

    def getOutput(self):
        return self._output


class FlattenLayer:
    def __init__(self):
        self._output = None
    def forward(self, tensor_in):
        tensor_out = self._flatten(tensor_in)
        self._output = tensor_out
        return tensor_out
    def _flatten(self, x):
        if isinstance(x, collections.abc.Iterable):
            return [a for i in x for a in self._flatten(i)]
        else:
            return [x]
    def getOutput(self):
        return self._output


class BatchNormLayer:
    """BatchNorm supporting arbitrary prefix dims; normalizes last axis."""

    def __init__(self, gamma, beta, moving_mean, moving_var, epsilon=1e-3):
        # Convert to plain Python lists of float
        self.gamma = [float(v) for v in gamma]
        self.beta = [float(v) for v in beta]
        self.moving_mean = [float(v) for v in moving_mean]
        self.moving_var = [float(v) for v in moving_var]
        self.epsilon = float(epsilon)
        self._output = None

    def forward(self, tensor_in):
        shape = dim(tensor_in)
        if not shape:
            raise ValueError("BatchNorm expects tensor input.")
        channels = shape[-1]
        if not (
            len(self.gamma)
            == len(self.beta)
            == len(self.moving_mean)
            == len(self.moving_var)
            == channels
        ):
            raise ValueError(
                f"BatchNorm channel mismatch: weights={len(self.gamma)} input_channels={channels}"
            )

        def _norm_leaf(vec):
            if len(vec) != channels:
                raise ValueError("BatchNorm expects last dimension size == channel count.")
            out = []
            for c, x in enumerate(vec):
                denom = math.sqrt(self.moving_var[c] + self.epsilon)
                # Avoid direct subtraction to preserve ConcolicFloat (no __sub__ support)
                norm = (x + (-self.moving_mean[c])) / denom
                out.append(self.gamma[c] * norm + self.beta[c])
            return out

        def _recurse(t):
            if not isinstance(t, collections.abc.Iterable) or isinstance(t, (str, bytes)):
                raise ValueError("BatchNorm expects iterable tensor.")
            # leaf: last dimension
            if not t or not isinstance(t[0], collections.abc.Iterable):
                return _norm_leaf(list(t))
            return [_recurse(sub) for sub in t]

        self._output = _recurse(tensor_in)
        return self._output

    def getOutput(self):
        return self._output


class LayerNormLayer:
    """LayerNorm with concrete-stat approximation for symbolic stability."""

    def __init__(self, gamma, beta, epsilon=1e-6):
        self.gamma = [float(v) for v in gamma]
        self.beta = [float(v) for v in beta]
        self.epsilon = float(epsilon)
        self._output = None

    def forward(self, tensor_in):
        channels = len(self.gamma)
        if channels != len(self.beta):
            raise ValueError("LayerNorm gamma/beta size mismatch.")

        def _norm_leaf(vec):
            if len(vec) != channels:
                raise ValueError(
                    f"LayerNorm channel mismatch: expected {channels}, got {len(vec)}"
                )
            concrete_vals = [float(unwrap(v)) for v in vec]
            mean = sum(concrete_vals) / channels
            var = sum((v - mean) * (v - mean) for v in concrete_vals) / channels
            inv_std = 1.0 / math.sqrt(var + self.epsilon)

            out = []
            for i, x in enumerate(vec):
                centered = x + (-mean)
                out.append(self.gamma[i] * (centered * inv_std) + self.beta[i])
            return out

        def _recurse(t):
            if not isinstance(t, collections.abc.Iterable) or isinstance(t, (str, bytes)):
                raise ValueError("LayerNorm expects iterable tensor.")
            if not t or not isinstance(t[0], collections.abc.Iterable):
                return _norm_leaf(list(t))
            return [_recurse(sub) for sub in t]

        self._output = _recurse(tensor_in)
        return self._output

    def getOutput(self):
        return self._output


class AddPositionEmbeddingLayer:
    def __init__(self, pos_embedding):
        pos = np.asarray(pos_embedding).tolist()
        if len(pos) == 1 and isinstance(pos[0], list):
            pos = pos[0]
        self.pos_embedding = pos
        self._output = None

    def _add_to_sequence(self, seq):
        if len(seq) != len(self.pos_embedding):
            raise ValueError(
                f"PositionEmbedding length mismatch: input={len(seq)} pos={len(self.pos_embedding)}"
            )
        out = []
        for token_idx, token in enumerate(seq):
            pos_token = self.pos_embedding[token_idx]
            if len(token) != len(pos_token):
                raise ValueError("PositionEmbedding dim mismatch.")
            out.append([token[d] + pos_token[d] for d in range(len(token))])
        return out

    def forward(self, tensor_in):
        shape = dim(tensor_in)
        if len(shape) == 2:
            self._output = self._add_to_sequence(tensor_in)
        elif len(shape) == 3:
            self._output = [self._add_to_sequence(sample) for sample in tensor_in]
        else:
            raise ValueError("AddPositionEmbedding expects rank-2 or rank-3 tensor.")
        return self._output

    def getOutput(self):
        return self._output


def _softmax_from_concrete(scores):
    if not scores:
        return []
    max_score = max(scores)
    exp_values = [float(unwrap(exp_func(v - max_score))) for v in scores]
    denom = sum(exp_values)
    if denom <= 0.0:
        return [1.0 / len(scores) for _ in scores]
    return [v / denom for v in exp_values]


class SequencePoolingLayer:
    """Token attention pooling; attention weights are concrete for solver safety."""

    def __init__(self, kernel, bias):
        kernel_arr = np.asarray(kernel, dtype=float)
        if kernel_arr.ndim != 2 or kernel_arr.shape[1] != 1:
            raise ValueError(f"SequencePooling kernel shape must be [D,1], got {kernel_arr.shape}.")
        self.kernel = [float(v[0]) for v in kernel_arr.tolist()]
        self.bias = float(np.asarray(bias, dtype=float).reshape(-1)[0])
        self._output = None

    def _pool_one(self, seq):
        token_count = len(seq)
        if token_count == 0:
            raise ValueError("SequencePooling expects non-empty sequence.")
        dim_size = len(self.kernel)
        for token in seq:
            if len(token) != dim_size:
                raise ValueError("SequencePooling token dim mismatch.")

        scores = []
        for token in seq:
            score = self.bias
            for d, x in enumerate(token):
                score += float(unwrap(x)) * self.kernel[d]
            scores.append(score)

        weights = _softmax_from_concrete(scores)
        pooled = []
        for d in range(dim_size):
            value = 0.0
            for t in range(token_count):
                value += weights[t] * seq[t][d]
            pooled.append(value)
        return pooled

    def forward(self, tensor_in):
        shape = dim(tensor_in)
        if len(shape) == 2:
            self._output = self._pool_one(tensor_in)
        elif len(shape) == 3:
            self._output = [self._pool_one(sample) for sample in tensor_in]
        else:
            raise ValueError("SequencePooling expects rank-2 or rank-3 tensor.")
        return self._output

    def getOutput(self):
        return self._output


# Define SimpleRNN class
class SimpleRNNLayer:
    def __init__(self, input_dim, weights, activation='tanh'):        
        self.input_dim = input_dim
        assert activation in (None, "tanh", "linear")
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

class MultiHeadAttentionLayer:
    def __init__(self, num_heads, key_dim_per_heads, wq, bq, wk, bk, wv, bv, output_weights, output_bias):
        self.num_heads = num_heads#20
        self.key_dim_per_heads = key_dim_per_heads#32
        self.WQ = wq.numpy().tolist()
        self.BQ = bq.numpy().tolist()
        self.WK = wk.numpy().tolist()
        self.BK = bk.numpy().tolist()
        self.WV = wv.numpy().tolist()
        self.BV = bv.numpy().tolist()
        self.WO = output_weights.numpy().tolist()
        self.BO = output_bias.numpy().tolist()

    def forward(self, input, mask=None):
        if len(dim(input)) == 2:
            return self.forwardSingle(input, mask)
        else:
            return self.forwardBatch(input, mask)
    
    def forwardBatch(self, inputs, mask=None):
        return [self.forwardSingle(input, mask) for input in inputs]
    def forwardSingle(self, input, mask=None):
        self.seq_len, self.model_dim = np.array(input).shape
        Q = self.transform_and_split(input, self.WQ, self.BQ)
        K = self.transform_and_split(input, self.WK, self.BK)
        V = self.transform_and_split(input, self.WV, self.BV)
        # print("KQV done")
        attentions = [self.dot_product_attention(Q[i], K[i], V[i]) for i in range(self.num_heads)]
        # print("attentions done")
        outputs = self.concatenate_and_transform(attentions, self.WO, self.BO)
        # print("outputs done")
        return outputs
    
    def transform_and_split(self,sequence_of_vectors, weights, bias):
    

        outputs = [
                        [
                            [self.mySum((weights[k][i][j] * vector[k]) for k in range(self.model_dim)) + bias[i][j] for j in range(self.key_dim_per_heads)]#32
                            for vector in sequence_of_vectors   #vector:1*32
                        ]
                        for i in range(self.num_heads)
                    ]
                
        return outputs
    def concatenate_and_transform(self,attentions, output_weights, output_bias):
        assert np.array(output_bias).shape == (self.model_dim,)
        outputs = [
            [
                self.mySum([
                      attentions[j][word][k] * output_weights[j][k][i]
                    for j in range(self.num_heads)
                    for k in range(self.key_dim_per_heads)
                ]) + output_bias[i]
                for i in range(self.model_dim)
            ]
            for word in range(self.seq_len)
        ]
        assert np.array(outputs).shape == (self.seq_len, self.model_dim)
        return outputs
    def mySum(self,x):
            s = 0.0
            # print('x:',type(x))
            # print('x[0]:',type(list(x)[0]))
            for i in x:
                s = s + i
            return s
    def dot_product_attention(self, Q, K, V):
        # print('$$$$$ dot product attention')
        K_T = [*zip(*K)]#32,500
    
        attention_scores = self.matrix_multiply(Q, K_T)#500,500
        # print("777")
        attention_scores = [[score / (self.key_dim_per_heads ** 0.5) for score in attention_score ]for attention_score in attention_scores]#500,500
        # print("779")
        attention_scores = self.softmax(attention_scores)#500,500
        # print("781")
        context_vector = self.matrix_multiply(attention_scores, V)#500,32
        # print("783")
        return context_vector

    def _register_attention_position(self, query_idx, key_idx):
        del key_idx
        register_current_indices([(query_idx, j) for j in range(self.model_dim)])

    def myMax(self,x,i):

        max = x[i][0]
        for j in range(len(x[i])):
            self._register_attention_position(i, j)
            if x[i][j] > max:
                max = x[i][j]
        return max
    
    def softmax(self,x):
        x_max = [self.myMax(x,i) for i in range(len(x))]
        concrete_x = [[float(unwrap(val)) for val in row] for row in x]
        concrete_max = [float(unwrap(val)) for val in x_max]
        e_x = [[math.exp(concrete_x[i][j] - concrete_max[i]) for j in range(len(concrete_x[i]))] for i in range(len(concrete_x))]
        e_x_sum = [self.mySum(e_x[i]) for i in range(len(e_x))]
        result = [[e_x[i][j] / e_x_sum[i] for j in range(len(e_x[i]))] for i in range(len(e_x)) ]
        return result
    
    def matrix_multiply(self, matrix1, matrix2):
        # print("start")
        if len(matrix1[0]) != len(matrix2):
            print("error")
            raise ValueError("矩陣的維度不符合乘法要求。")

        result = [[0] * len(matrix2[0]) for _ in range(len(matrix1))]
        # print("result before ", len(matrix1), len(matrix2[0]), len(matrix2))
        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)):
                    # print("i,j,k:",i,'/',len(matrix1),j,'/',len(matrix2[0]),k,'/',len(matrix2))
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
                    # print("805")
        # print("result after:",result[i][j])

        return result
    
    def dot_product(self, vector1, vector2):
        # assert vector1.shape == (self.model_dim,)
        # assert vector2.shape == (self.model_dim,)
        return self.mySum(vector1[i] * vector2[i] for i in range(self.model_dim))

class ReshapeLayer:
    def __init__(self, target_shape):
        self.target_shape = target_shape
        self._output = None

    def forward(self, tensor_in):
        tensor_out = self._reshape(tensor_in, self.target_shape)
        self._output = tensor_out
        return tensor_out

    def _reshape(self, x, shape):
        # Flatten the input
        # print("x:",np.array(x).shape)
        # print("shape:",shape)
        flat_list = self._flatten(x)
        # print("flat_list:",flat_list)
        # Create an iterator for the flattened list
        iterator = iter(flat_list)
        # Recursively build the reshaped list
        return self._build_shape(iterator, shape)

    def _flatten(self, x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes)):
            flat_list = []
            for item in x:
                flat_list.extend(self._flatten(item))
            return flat_list
        else:
            return [x]

    def _build_shape(self, iterator, shape):
        if not shape:
            return next(iterator)
        return [self._build_shape(iterator, shape[1:]) for _ in range(shape[0])]

    def getOutput(self):
        return self._output



class GlobalAveragePooling2DLayer:
    def __init__(self):
        self._output = None

    def forward(self, tensor_in):
        in_shape = dim(tensor_in)
        assert len(in_shape) == 3, "GlobalAveragePooling2D expects 3D input [H][W][C]"
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


class GlobalAveragePooling1DLayer:
    def __init__(self):
        self._output = None

    def forward(self, tensor_in):
        # print("globol tensor_in:",len(tensor_in),len(tensor_in[0])) 500,32
        tensor_out = self._global_avg_pooling(tensor_in)

        self._output = tensor_out
        # print("globol tensor_out:",len(tensor_out),len(tensor_out[0]))#1,32
        return tensor_out

    def _global_avg_pooling(self, x):
        if isinstance(x, collections.abc.Iterable):
            if isinstance(x[0], collections.abc.Iterable):
                channel_pooled_values = []
                for channel in zip(*x):
                    avg_value = sum(channel) / len(channel)
                    channel_pooled_values.append(avg_value)
                return channel_pooled_values
            else:
                raise ValueError("Input tensor must be 3D")
        else:
            raise ValueError("Input tensor must be iterable")

    def getOutput(self):
        return self._output



class NNModel:
    def __init__(self):
        self.layers: List[Any] = []
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.my_layer_keys: List[str] = []
        self.keras_to_cache_key: dict[str, str] = {}
        self.input_layer_names: List[str] = []
        self.multiple_inputs = False
        self.layer_type_counter: dict[str, int] = {}

    def register_input_names(self, names: List[str]):
        self.input_layer_names = list(names or [])
        if not self.input_layer_names:
            return
        if len(self.input_layer_names) > 1:
            self.multiple_inputs = True
            raise NotImplementedError("Multiple input tensors are not supported yet.")
        self.keras_to_cache_key[self.input_layer_names[0]] = "layer_input"

    def _resolve_cache_key(self, keras_name: str) -> str:
        if keras_name not in self.keras_to_cache_key:
            known = ", ".join(sorted(self.keras_to_cache_key.keys()))
            raise KeyError(f"Unknown inbound source '{keras_name}'. Known sources: {known}")
        return self.keras_to_cache_key[keras_name]

    def _register_cache_key(self, keras_name: str, cache_key: str) -> None:
        if keras_name:
            self.keras_to_cache_key[keras_name] = cache_key

    def _append_layer(self, layer_obj: Any) -> str:
        type_name = type(layer_obj).__name__
        count = self.layer_type_counter.get(type_name, 0)
        cache_key = f"{type_name}_{count}"
        self.layer_type_counter[type_name] = count + 1
        self.layers.append(layer_obj)
        self.my_layer_keys.append(cache_key)
        return cache_key

    def forward(self, tensor_in):
        log.info("DNN start forwarding")
        cache = {"layer_input": tensor_in}
        x = tensor_in
        for i, layer in enumerate(self.layers):
            register_current_layer_number(to_Keras_layer_number(i))
            log.debug("Forwarding layer %s (%s)", i, layer.__class__.__name__)
            layer_key = self.my_layer_keys[i] if i < len(self.my_layer_keys) else f"layer_{i}"
            if hasattr(layer, "input_from") and layer.input_from:
                inputs = []
                for name in layer.input_from:
                    if name not in cache:
                        raise KeyError(
                            f"Missing cached input '{name}' while forwarding layer "
                            f"index={i} type={layer.__class__.__name__}"
                        )
                    inputs.append(cache[name])
                if getattr(layer, "multi_input", False):
                    x = layer.forward(inputs)
                else:
                    x = layer.forward(inputs[0])
            else:
                x = layer.forward(x)
            cache[layer_key] = x
        log.info("DNN finish forwarding")
        return x

    def getLayOutput(self, idx):
        if idx >= len(self.layers):
            return None
        else:
            return self.layers[idx].getOutput()

    def addLayer(self, layer, inbound_names: Optional[List[str]] = None):
        inbound_names = inbound_names or []
        keras_name = getattr(layer, "name", f"layer_{len(self.layers)}")
        layer_type_name = layer.__class__.__name__
        resolved_inbounds = []
        for inbound_name in inbound_names:
            try:
                resolved_inbounds.append(self._resolve_cache_key(inbound_name))
            except KeyError as exc:
                raise KeyError(
                    f"Missing inbound '{inbound_name}' for layer '{keras_name}' "
                    f"({layer_type_name})"
                ) from exc
        created = 0

        if isinstance(layer, Conv2D):
            #print("Conv2D")
            # shape: (outputs, rows, cols, channel)
            layer_weights = layer.get_weights()
            weights = layer_weights[0].transpose(3, 0, 1, 2)
            if len(layer_weights) > 1:
                biases = layer_weights[1]
            else:
                # Conv2D can be configured with use_bias=False
                biases = np.zeros(weights.shape[0], dtype=float)
            config = layer.get_config()
            activation = config['activation']
            stride = config.get('strides', (1, 1))
            padding = config.get('padding', 'valid')

            conv_layer = Conv2DLayer(weights, biases, weights.shape, stride=stride, padding=padding)
            if resolved_inbounds:
                conv_layer.input_from = resolved_inbounds
            conv_key = self._append_layer(conv_layer)
            created += 1
            activation_layer = ActivationLayer(activation)
            activation_layer.input_from = [conv_key]
            activation_key = self._append_layer(activation_layer)
            created += 1
            self._register_cache_key(keras_name, activation_key)
        elif isinstance(layer, Dense):
            #print("Dense")
            # shape: (outputs, inputs)
            layer_weights = layer.get_weights()
            weights = layer_weights[0].transpose()
            if len(layer_weights) > 1:
                biases = layer_weights[1]
            else:
                # Dense can be configured with use_bias=False
                biases = np.zeros(weights.shape[0], dtype=float)
            activation = layer.get_config()['activation']

            dense_layer = DenseLayer(weights, biases, weights.shape)
            if resolved_inbounds:
                dense_layer.input_from = resolved_inbounds
            dense_key = self._append_layer(dense_layer)
            created += 1
            log.debug("Add Activation Layer: %s", activation)
            activation_layer = ActivationLayer(activation)
            activation_layer.input_from = [dense_key]
            activation_key = self._append_layer(activation_layer)
            created += 1
            self._register_cache_key(keras_name, activation_key)
        elif isinstance(layer, MaxPool2D) or isinstance(layer, MaxPooling2D):
            #print("MaxPool2D")
            pool_size = layer.get_config()['pool_size']
            # print(pool_size)
            maxpool_layer = MaxPool2DLayer(pool_size)
            if resolved_inbounds:
                maxpool_layer.input_from = resolved_inbounds
            key = self._append_layer(maxpool_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, Flatten):
            #print("Flatten")
            flatten_layer = FlattenLayer()
            if resolved_inbounds:
                flatten_layer.input_from = resolved_inbounds
            key = self._append_layer(flatten_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, Activation):
            activation = layer.get_config()['activation']
            activation_layer = ActivationLayer(activation)
            if resolved_inbounds:
                activation_layer.input_from = resolved_inbounds
            key = self._append_layer(activation_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, ReLU):
            relu_layer = ActivationLayer("relu")
            if resolved_inbounds:
                relu_layer.input_from = resolved_inbounds
            key = self._append_layer(relu_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, Dropout):
            # inference-time deterministic: no-op
            noop_layer = NoOpLayer("Dropout")
            if resolved_inbounds:
                noop_layer.input_from = resolved_inbounds
            key = self._append_layer(noop_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, RandomFlip):
            # inference-time deterministic: no-op
            noop_layer = NoOpLayer("RandomFlip")
            if resolved_inbounds:
                noop_layer.input_from = resolved_inbounds
            key = self._append_layer(noop_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, ZeroPadding2D):
            padding = layer.get_config().get("padding", 0)
            if isinstance(padding, int):
                top = bottom = left = right = padding
            elif len(padding) == 2 and isinstance(padding[0], int):
                top = bottom = padding[0]
                left = right = padding[1]
            else:
                (top, bottom), (left, right) = padding
            padding_layer = ZeroPadding2DLayer((top, bottom, left, right))
            if resolved_inbounds:
                padding_layer.input_from = resolved_inbounds
            key = self._append_layer(padding_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, RandomCrop):
            cfg = layer.get_config()
            crop_layer = CenterCrop2DLayer(cfg.get("height"), cfg.get("width"))
            if resolved_inbounds:
                crop_layer.input_from = resolved_inbounds
            key = self._append_layer(crop_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, BatchNormalization):
            gamma, beta, moving_mean, moving_var = layer.get_weights()
            epsilon = layer.get_config().get("epsilon", 1e-3)
            bn_layer = BatchNormLayer(gamma, beta, moving_mean, moving_var, epsilon)
            if resolved_inbounds:
                bn_layer.input_from = resolved_inbounds
            key = self._append_layer(bn_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, LayerNormalization):
            gamma, beta = layer.get_weights()
            epsilon = layer.get_config().get("epsilon", 1e-6)
            ln_layer = LayerNormLayer(gamma, beta, epsilon)
            if resolved_inbounds:
                ln_layer.input_from = resolved_inbounds
            key = self._append_layer(ln_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, SimpleRNN):
            input_dim = layer.input_shape[-1]
            activation = layer.get_config()['activation']
            rnn_layer = SimpleRNNLayer(input_dim, weights=layer.get_weights(), activation=activation)
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
        elif isinstance(layer, MultiHeadAttention):
            num_heads = layer.get_config()['num_heads']
            # num_heads#20
            #32*20
            key_dim_per_heads = layer.get_config()['key_dim']
            wq=layer._query_dense.kernel
            bq=layer._query_dense.bias
            wk=layer._key_dense.kernel
            bk=layer._key_dense.bias
            wv=layer._value_dense.kernel
            bv=layer._value_dense.bias
            output_weights=layer._output_dense.kernel
            output_bias=layer._output_dense.bias
            mha_layer = MultiHeadAttentionLayer(num_heads,key_dim_per_heads,wq,bq,wk,bk,wv,bv,output_weights,output_bias)
            if resolved_inbounds:
                mha_layer.input_from = resolved_inbounds
            key = self._append_layer(mha_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, Add):
            add_layer = AddLayer(resolved_inbounds)
            key = self._append_layer(add_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, GlobalAveragePooling2D):
            gap2d_layer = GlobalAveragePooling2DLayer()
            if resolved_inbounds:
                gap2d_layer.input_from = resolved_inbounds
            key = self._append_layer(gap2d_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, GlobalAveragePooling1D):
            log.debug("GlobalAveragePooling1D layer added")
            gap1d_layer = GlobalAveragePooling1DLayer()
            if resolved_inbounds:
                gap1d_layer.input_from = resolved_inbounds
            key = self._append_layer(gap1d_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif isinstance(layer, Reshape):
            reshape_layer = ReshapeLayer(layer.target_shape)
            if resolved_inbounds:
                reshape_layer.input_from = resolved_inbounds
            key = self._append_layer(reshape_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif layer_type_name == "AddPositionEmbedding":
            layer_weights = layer.get_weights()
            if not layer_weights:
                raise ValueError("AddPositionEmbedding layer has no position embedding weights.")
            pos_layer = AddPositionEmbeddingLayer(layer_weights[0])
            if resolved_inbounds:
                pos_layer.input_from = resolved_inbounds
            key = self._append_layer(pos_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif layer_type_name == "DropPath":
            drop_path_layer = NoOpLayer("DropPath")
            if resolved_inbounds:
                drop_path_layer.input_from = resolved_inbounds
            key = self._append_layer(drop_path_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        elif layer_type_name == "SequencePooling":
            layer_weights = layer.get_weights()
            if len(layer_weights) < 2:
                raise ValueError(
                    "SequencePooling layer requires [kernel, bias] weights, but none were found."
                )
            seq_pool_layer = SequencePoolingLayer(layer_weights[0], layer_weights[1])
            if resolved_inbounds:
                seq_pool_layer.input_from = resolved_inbounds
            key = self._append_layer(seq_pool_layer)
            created += 1
            self._register_cache_key(keras_name, key)
        else:
            raise NotImplementedError(f"Unsupported layer: {layer_type_name} (name={keras_name})")

        return created
