import os
import re
import keras
from dnnct.myDNN import NNModel
import numpy as np
import itertools

from libct.position import register_layer_number_mapping, to_Keras_layer_number


def _get_inbound_layers(layer):  # 取得某層的上一層神經層的資訊
    inbound = []
    for node in getattr(layer, "_inbound_nodes", []):  # keras把上一層的所有資訊存在_inbound_nodes
        inbound_layers = getattr(
            node, "inbound_layers", None)  # 取得上一層layer name
        if inbound_layers is None:
            continue
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        # 保證神經層依照資料流動順序維護神經層關係
        inbound.extend(l for l in inbound_layers if l is not None)
    return inbound


def _collect_layers_and_inbound(model):  # 收集模型的所有層及其上一層關係
    layers = [l for l in model.layers if type(l).__name__ not in [
        'Dropout', 'InputLayer']]
    inbound_map = {}
    for layer in layers:
        inbound_map[layer.name] = [
            parent.name for parent in _get_inbound_layers(layer)]
    return layers, inbound_map


def _format_shape(shape):
    if shape is None:
        return "None"
    return str(shape)


# 純粹對照 keras 模型定義層
def _describe_model_layers(model, layers):
    print("=== Keras model.layers order ===")
    for idx, layer in enumerate(model.layers):
        inbound = [parent.name for parent in _get_inbound_layers(layer)]
        print(
            f"[{idx:03d}] {layer.name:<30} {layer.__class__.__name__:<28} "
            f"output={_format_shape(getattr(layer, 'output_shape', None))} "
            f"inbound={inbound}"
        )


RESNET_BUILDER_BY_DEPTH = {
    18: "ResNet18",
    34: "ResNet34",
    50: "ResNet50",
    101: "ResNet101",
    152: "ResNet152",
}

KNOWN_RESNET_SPECS = {
    "resnet18_mnist": {"depth": 18, "input_shape": (28, 28, 1), "num_classes": 10},
    "resnet50_imagenet": {"depth": 50, "input_shape": (224, 224, 3), "num_classes": 1000},
}


def _get_resnet_custom_objects():
    custom_objects = {}
    try:
        from keras_resnet.models import ResNet2D18
        from keras_resnet.layers import BatchNormalization as ResNetBatchNorm

        class ResNetBatchNormCompat(ResNetBatchNorm):
            def __init__(self, *args, freeze=False, **kwargs):
                super().__init__(*args, freeze=freeze, **kwargs)

            @classmethod
            def from_config(cls, config):
                config = dict(config)
                config.setdefault("freeze", False)
                return super(ResNetBatchNormCompat, cls).from_config(config)

        custom_objects["ResNet2D18"] = ResNet2D18
        custom_objects["BatchNormalization"] = ResNetBatchNormCompat
    except ModuleNotFoundError:
        pass
    return custom_objects


def _needs_resnet_fallback(exc):
    msg = str(exc)
    return ("Unknown layer: 'ResNet2D" in msg
            or "missing 1 required positional argument: 'inputs'" in msg)


def _infer_resnet_specs(basename, input_shape_override=None, num_classes_override=None):
    specs = KNOWN_RESNET_SPECS.get(basename, {}).copy()
    if input_shape_override is not None:
        specs["input_shape"] = input_shape_override
    if num_classes_override is not None:
        specs["num_classes"] = num_classes_override
    if "depth" not in specs:
        match = re.search(r"resnet(\d+)", basename)
        if match:
            specs["depth"] = int(match.group(1))
    if "input_shape" not in specs or "num_classes" not in specs or "depth" not in specs:
        return None
    return specs


def _build_resnet_model(depth, input_shape, num_classes):
    try:
        from keras_resnet import models as kr_models
    except ModuleNotFoundError as exc:
        raise ImportError(
            "載入 ResNet 權重需要 `keras-resnet` 套件，請先安裝 `pip install keras-resnet`。"
        ) from exc

    builder_name = RESNET_BUILDER_BY_DEPTH.get(depth)
    if builder_name is None:
        raise ValueError(f"不支援的 ResNet 深度: {depth}")
    builder = getattr(kr_models, builder_name, None)
    if builder is None:
        raise ValueError(f"在 keras_resnet 中找不到建構函式 {builder_name}")

    inputs = keras.layers.Input(shape=input_shape)
    return builder(inputs=inputs, classes=num_classes, include_top=True, freeze_bn=False)


def load_keras_model(model_path, input_shape_override=None, num_classes_override=None):
    """載入 Keras 模型，必要時改以建立 ResNet 架構後匯入權重。"""
    custom_objects = _get_resnet_custom_objects()
    try:
        return keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False)
    except (ValueError, TypeError) as exc:
        if not _needs_resnet_fallback(exc):
            raise
        basename = os.path.splitext(os.path.basename(model_path))[0]
        specs = _infer_resnet_specs(
            basename, input_shape_override=input_shape_override,
            num_classes_override=num_classes_override
        )
        if specs is None:
            raise ValueError(
                f"無法解析 {basename} 的輸入形狀或分類數，請提供 input_shape / num_classes。"
            ) from exc
        model = _build_resnet_model(specs["depth"], specs["input_shape"], specs["num_classes"])
        model.load_weights(model_path)
        return model


myModel = None


def init_model(model_path):
    global myModel
    model = load_keras_model(model_path)
    model.summary()
    layers, inbound_map = _collect_layers_and_inbound(model)
    _describe_model_layers(model, layers)
    myModel = NNModel()
    input_layer_names = [
        layer.name for layer in model.layers if type(layer).__name__ == "InputLayer"]
    myModel.register_input_names(input_layer_names)

    # 1: is because 1st dim of input shape of Keras model is batch size (None)
    myModel.input_shape = model.input_shape[1:]
    myLayerCount = 0
    for i, layer in enumerate(layers):
        inbound_names = inbound_map.get(layer.name, [])
        numberOfMyLayers = myModel.addLayer(
            layer, inbound_names=inbound_names)
        # maintain the mapping of layers between Keras model and my model
        for j in range(numberOfMyLayers):
            register_layer_number_mapping(i, myLayerCount)
            myLayerCount += 1
    model.summary()
    print('Number of layers in my model:', len(myModel.layers))
    print('Number of layers in original Keras model:', len(layers))
    print('Correspondence between layers in Keras model and my model:')
    for myLayerNumber in range(myLayerCount):
        print('My: ', myLayerNumber, 'Keras: ',
              to_Keras_layer_number(myLayerNumber))
    for myLayer in myModel.layers:
        print(type(myLayer))


def _forward_from_flat_data(data):
    """將扁平輸入還原成 tensor 並回傳 NNModel logits。"""
    input_shape = myModel.input_shape
    iter_args = (range(dim) for dim in input_shape)
    X = np.zeros(input_shape).tolist()
    data_name_prefix = "v_"
    for i in itertools.product(*iter_args):
        if len(i) == 2:
            X[i[0]][i[1]] = data[f"{data_name_prefix}{i[0]}_{i[1]}"]
        elif len(i) == 3:
            X[i[0]][i[1]][i[2]
                          ] = data[f"{data_name_prefix}{i[0]}_{i[1]}_{i[2]}"]
        elif len(i) == 4:
            X[i[0]][i[1]][i[2]][i[3]
                                ] = data[f"{data_name_prefix}{i[0]}_{i[1]}_{i[2]}_{i[3]}"]
    return myModel.forward(X)


def predict_logits(**data):
    """回傳 NNModel 的 logits，方便和 Keras 模型做數值比對。"""
    return _forward_from_flat_data(data)


def predict(**data):
    # calculate the output of the model
    out_val = _forward_from_flat_data(data)

    # 用一顆神經元做二分類
    if len(out_val) == 1:
        if out_val[0] > 0.5:
            ret_class = 1
        else:
            ret_class = 0
    else:
        max_val, ret_class = out_val[0], 0
        for i, cl_val in enumerate(out_val):
            if cl_val > max_val:
                max_val, ret_class = cl_val, i

    print("[DEBUG]predicted class:", ret_class)
    return ret_class
