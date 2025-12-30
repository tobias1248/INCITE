import os
import re
import json
import keras
import h5py
from dnnct.myDNN import NNModel
import numpy as np
import itertools

from libct.position import register_layer_number_mapping, to_Keras_layer_number


def _get_inbound_layers(layer):
    inbound = []
    for node in getattr(layer, "_inbound_nodes", []):
        inbound_layers = getattr(node, "inbound_layers", None)
        if inbound_layers is None:
            continue
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        inbound.extend(l for l in inbound_layers if l is not None)
    return inbound


def _collect_layers_and_inbound(model):
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


def _describe_model_layers(model, layers):
    print("=== Keras model.layers order ===")
    for idx, layer in enumerate(model.layers):
        inbound = [parent.name for parent in _get_inbound_layers(layer)]
        print(
            f"[{idx:03d}] {layer.name:<30} {layer.__class__.__name__:<28} "
            f"output={_format_shape(getattr(layer, 'output_shape', None))} "
            f"inbound={inbound}"
        )


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


# 此時h5模型還沒被讀 不能直接get_config
def _read_model_config(model_path):
    try:
        with h5py.File(model_path, 'r') as f:
            if 'model_config' not in f.attrs:
                return None
            config = f.attrs['model_config']
            if isinstance(config, bytes):
                config = config.decode('utf-8')
            return json.loads(config)
    except Exception:
        return None


def _get_input_shape_from_h5(model_path):
    """從 H5 model_config 直接讀取第一個 batch_input_shape（備援用）。"""
    try:
        with h5py.File(model_path, 'r') as f:
            raw = f.attrs.get('model_config')
            if raw is None:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8')
            config = json.loads(raw)
    except Exception:
        return None

    layers = config.get('config', {}).get('layers', [])
    for layer in layers:
        layer_conf = layer.get('config', {})
        batch_shape = layer_conf.get('batch_input_shape') or layer_conf.get('batch_shape')
        if batch_shape:
            if len(batch_shape) > 1:
                return tuple(None if dim is None else int(dim) for dim in batch_shape[1:])
            return tuple(None if dim is None else int(dim) for dim in batch_shape)
    return None


def _extract_specs_from_weights(model_path):
    specs = {}
    try:
        with h5py.File(model_path, 'r') as f:
            weights_group = f.get('model_weights')
            if weights_group is None:
                return specs

            input_channels = None
            dense_candidates = []

            def visitor(name, obj):
                nonlocal input_channels
                if not isinstance(obj, h5py.Dataset):
                    return
                if not name.endswith('kernel:0'):
                    return
                shape = obj.shape
                lname = name.lower()
                if len(shape) == 4 and input_channels is None:
                    input_channels = shape[2]
                elif len(shape) == 2:
                    hint = 0 if any(tag in lname for tag in (
                        'pred', 'logit', 'fc', 'dense', 'classifier')) else 1
                    dense_candidates.append((hint, int(shape[1])))

            weights_group.visititems(visitor)

            if input_channels is not None:
                specs['input_shape'] = (None, None, int(input_channels))

            # 分類模型中，最後一個 Dense 層的權重矩陣大小正好是 (特徵數, 類別數)，所以只要抓到那個權重，就能把 final_out 視為 num_classes
            if dense_candidates:
                dense_candidates.sort(key=lambda tup: (tup[0], tup[1]))
                specs['num_classes'] = dense_candidates[0][1]

    except Exception:
        return specs

    return specs


def _extract_specs_from_h5(model_path):
    specs = {}
    config = _read_model_config(model_path)
    if config:
        class_name = config.get('class_name', '')
        matches = re.findall(r'(\d+)', class_name)
        if matches:
            specs['depth'] = int(matches[-1])

        conf = config.get('config', {})
        layers = conf.get('layers', [])
        for layer in layers:
            class_name = layer.get('class_name')
            layer_conf = layer.get('config', {})
            if specs.get('input_shape') is None and class_name == 'InputLayer':
                batch_shape = layer_conf.get('batch_input_shape')
                if batch_shape:
                    specs['input_shape'] = tuple(
                        None if dim is None else int(dim) for dim in batch_shape[1:])
            if class_name == 'Dense':
                units = layer_conf.get('units')
                if units is not None:
                    specs['num_classes'] = int(units)

    weight_specs = _extract_specs_from_weights(model_path)
    for key, value in weight_specs.items():
        specs.setdefault(key, value)

    if 'input_shape' not in specs:
        fallback_shape = _get_input_shape_from_h5(model_path)
        if fallback_shape is not None:
            specs['input_shape'] = fallback_shape

    return specs


def _infer_resnet_specs(model_path, input_shape_override=None, num_classes_override=None):
    specs = _extract_specs_from_h5(model_path)
    if input_shape_override is not None:
        specs["input_shape"] = input_shape_override
    if num_classes_override is not None:
        specs["num_classes"] = num_classes_override

    input_shape = specs.get("input_shape")
    if input_shape is not None:
        if any(dim is None for dim in input_shape):
            if input_shape_override is not None:
                specs["input_shape"] = input_shape_override
            else:
                # 若形狀不完整，寧可要求使用者提供，避免不可靠的推斷
                specs.pop("input_shape", None)

    if "depth" not in specs:
        basename = os.path.splitext(os.path.basename(model_path))[0]
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

    builder = None
    candidate_names = [f"ResNet2D{depth}", f"ResNet{depth}"]
    for name in candidate_names:
        builder = getattr(kr_models, name, None)
        if builder is not None:
            break

    if builder is None:
        raise ValueError(
            f"在 keras_resnet.models 中找不到對應深度 {depth} 的 ResNet builder")

    inputs = keras.layers.Input(shape=input_shape)
    return builder(inputs=inputs, classes=num_classes, include_top=True, freeze_bn=False)


def load_keras_model(model_path, input_shape_override=None, num_classes_override=None):
    custom_objects = _get_resnet_custom_objects()
    try:
        return keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False)
    except (ValueError, TypeError) as exc:
        if not _needs_resnet_fallback(exc):
            raise
        basename = os.path.splitext(os.path.basename(model_path))[0]
        specs = _infer_resnet_specs(
            model_path, input_shape_override=input_shape_override,
            num_classes_override=num_classes_override
        )
        if specs is None:
            raise ValueError(
                f"無法解析 {basename} 的輸入形狀或分類數，請提供 input_shape / num_classes。"
            ) from exc
        model = _build_resnet_model(
            specs["depth"], specs["input_shape"], specs["num_classes"])
        model.load_weights(model_path)
        return model


myModel = None


def init_model(model_path, verbose=True, input_shape_override=None, num_classes_override=None):
    global myModel
    model = load_keras_model(model_path,
                             input_shape_override=input_shape_override,
                             num_classes_override=num_classes_override)
    if verbose:
        model.summary()
    layers, inbound_map = _collect_layers_and_inbound(model)
    _describe_model_layers(model, layers)
    myModel = NNModel()
    input_layer_names = [
        layer.name for layer in model.layers if type(layer).__name__ == "InputLayer"]
    myModel.register_input_names(input_layer_names)

    myModel.input_shape = model.input_shape[1:]
    myLayerCount = 0
    for i, layer in enumerate(layers):
        inbound_names = inbound_map.get(layer.name, [])
        numberOfMyLayers = myModel.addLayer(
            layer, inbound_names=inbound_names)
        for j in range(numberOfMyLayers):
            register_layer_number_mapping(i, myLayerCount)
            myLayerCount += 1
    if verbose:
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
    return _forward_from_flat_data(data)


def predict(**data):
    out_val = _forward_from_flat_data(data)

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
