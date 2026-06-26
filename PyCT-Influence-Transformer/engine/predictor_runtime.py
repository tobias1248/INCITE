import itertools
import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, cast

import keras
import numpy as np
from dnnct.myDNN import NNModel

from libct.position import register_layer_number_mapping, to_Keras_layer_number
from modeling.custom_layers import (
    AddClsToken,
    AddPositionEmbedding,
    DropPath,
    ExtractClsToken,
    SequencePooling,
)
from modeling.keras_loader import load_model_with_compat

log = logging.getLogger("ct.model")

ModelCacheKey = Tuple[str, bool, float]
ModelRole = Literal["default", "search", "validation"]
myModel: Optional[NNModel] = None
loaded_model_path: Optional[str] = None
loaded_model_key: Optional[ModelCacheKey] = None
_MODEL_CACHE: Dict[ModelCacheKey, NNModel] = {}
searchModel: Optional[NNModel] = None
search_model_key: Optional[ModelCacheKey] = None
validationModel: Optional[NNModel] = None
validation_model_key: Optional[ModelCacheKey] = None


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


def _collect_input_names(model):
    names = []
    seen = set()

    for tensor in getattr(model, 'inputs', []) or []:
        tensor_name = getattr(tensor, 'name', None)
        if isinstance(tensor_name, str) and tensor_name:
            normalized = tensor_name.split(':', 1)[0]
            if normalized not in seen:
                seen.add(normalized)
                names.append(normalized)

    for layer in model.layers:
        if type(layer).__name__ != 'InputLayer':
            continue
        layer_name = getattr(layer, 'name', None)
        if isinstance(layer_name, str) and layer_name and layer_name not in seen:
            seen.add(layer_name)
            names.append(layer_name)

    input_names_attr = getattr(model, 'input_names', None)
    if isinstance(input_names_attr, (list, tuple)):
        for name in input_names_attr:
            if isinstance(name, str) and name and name not in seen:
                seen.add(name)
                names.append(name)

    return names


def _collect_layers_and_inbound(model):
    excluded = {"Dropout", "InputLayer", "Embedding"}
    layers = [layer for layer in model.layers if type(layer).__name__ not in excluded]
    kept_names = {layer.name for layer in layers}
    inbound_map = {}

    def _resolve_parent_names(parent):
        if parent is None:
            return []
        parent_type = type(parent).__name__
        if parent.name in kept_names or parent_type == "InputLayer":
            return [parent.name]

        names = []
        for grand_parent in _get_inbound_layers(parent):
            names.extend(_resolve_parent_names(grand_parent))
        return names

    for layer in layers:
        resolved = []
        seen = set()
        for parent in _get_inbound_layers(layer):
            for name in _resolve_parent_names(parent):
                if name in seen:
                    continue
                seen.add(name)
                resolved.append(name)
        inbound_map[layer.name] = resolved
    return layers, inbound_map


def _assign_model_for_role(
    role: ModelRole,
    model: NNModel,
    model_key: ModelCacheKey,
    model_path: str,
) -> None:
    global myModel, loaded_model_key, loaded_model_path
    global searchModel, search_model_key, validationModel, validation_model_key

    if role == "default":
        myModel = model
        loaded_model_key = model_key
        loaded_model_path = model_path
        return
    if role == "search":
        searchModel = model
        search_model_key = model_key
        return
    if role == "validation":
        validationModel = model
        validation_model_key = model_key
        return
    raise ValueError(f"Unsupported model role: {role}")


def init_model(
    model_path,
    *,
    ternary_simplification: bool = False,
    ternary_threshold_scale: float = 0.75,
    role: ModelRole = "default",
):
    global myModel, loaded_model_key, loaded_model_path
    resolved_model_path = str(model_path)
    model_key: ModelCacheKey = (
        resolved_model_path,
        bool(ternary_simplification),
        float(ternary_threshold_scale),
    )
    cached = _MODEL_CACHE.get(model_key)
    if cached is not None:
        _assign_model_for_role(role, cached, model_key, resolved_model_path)
        return
    if loaded_model_key is not None and loaded_model_key != model_key:
        keras.backend.clear_session()
    model = load_model_with_compat(resolved_model_path)
    model_stem = Path(resolved_model_path).stem
    model._name = model_stem
    model.summary()
    log.info("Loaded model '%s' with input_shape=%s", model_stem, model.input_shape)
    layers, inbound_map = _collect_layers_and_inbound(model)
    my_model = NNModel(
        ternary_simplification=ternary_simplification,
        ternary_threshold_scale=ternary_threshold_scale,
    )
    input_layer_names = _collect_input_names(model)
    my_model.register_input_names(input_layer_names)
    my_model.input_shape = model.input_shape[1:]
    my_layer_count = 0
    for i, layer in enumerate(layers):
        inbound_names = inbound_map.get(layer.name, [])
        number_of_my_layers = my_model.addLayer(layer, inbound_names=inbound_names)
        log.info("Layer %s mapped to %s internal layer(s)", i, number_of_my_layers)
        for _ in range(number_of_my_layers):
            register_layer_number_mapping(i, my_layer_count)
            my_layer_count += 1

    log.info("Number of layers in my model: %s", len(my_model.layers))
    log.info("Number of layers in original Keras model: %s", len(layers))
    for my_layer_number in range(my_layer_count):
        log.info("My layer %s -> Keras layer %s", my_layer_number, to_Keras_layer_number(my_layer_number))
    for my_layer in my_model.layers:
        log.debug("My model layer type: %s", type(my_layer).__name__)

    _MODEL_CACHE[model_key] = my_model
    _assign_model_for_role(role, my_model, model_key, resolved_model_path)


def _predict_with_model(model: Optional[NNModel], *, require_finite: bool = False, **data):
    if model is None or model.input_shape is None:
        raise RuntimeError("Model not initialized. Call init_model() before predict().")

    model = cast(NNModel, model)
    input_shape = cast(Tuple[int, ...], model.input_shape)
    iter_args = (range(dim) for dim in input_shape)
    tensor_input = np.zeros(input_shape).tolist()
    data_name_prefix = "v_"
    for index in itertools.product(*iter_args):
        if len(index) == 2:
            tensor_input[index[0]][index[1]] = data[f"{data_name_prefix}{index[0]}_{index[1]}"]
        elif len(index) == 3:
            tensor_input[index[0]][index[1]][index[2]] = data[
                f"{data_name_prefix}{index[0]}_{index[1]}_{index[2]}"
            ]
        elif len(index) == 4:
            tensor_input[index[0]][index[1]][index[2]][index[3]] = data[
                f"{data_name_prefix}{index[0]}_{index[1]}_{index[2]}_{index[3]}"
            ]

    if require_finite and not np.isfinite(np.asarray(tensor_input, dtype=np.float64)).all():
        raise ValueError("Validation input contains NaN or Inf")

    out_val = model.forward(tensor_input)
    log.debug("Completed forward pass for input_shape=%s", input_shape)

    if require_finite and not np.isfinite(np.asarray(out_val, dtype=np.float64)).all():
        raise ValueError("Validation model output contains NaN or Inf")

    if len(out_val) == 1:
        if isinstance(out_val[0], list):
            ret_class = 1 if out_val[0][0] > 0.5 else 0
        else:
            ret_class = 1 if out_val[0] > 0.5 else 0
    else:
        max_val, ret_class = out_val[0], 0
        for i, cl_val in enumerate(out_val):
            if cl_val > max_val:
                max_val, ret_class = cl_val, i

    log.info("Forward prediction complete (class=%s)", ret_class)
    return ret_class


def predict(**data):
    return _predict_with_model(myModel, require_finite=True, **data)


def predict_search(**data):
    return _predict_with_model(searchModel, **data)


def predict_validation(**data):
    return _predict_with_model(validationModel, require_finite=True, **data)


__all__ = [
    "AddClsToken",
    "AddPositionEmbedding",
    "DropPath",
    "ExtractClsToken",
    "SequencePooling",
    "init_model",
    "predict",
    "predict_search",
    "predict_validation",
]
