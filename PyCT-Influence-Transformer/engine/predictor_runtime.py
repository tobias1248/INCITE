import itertools
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, cast

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
ModelRole = Literal["default", "search"]
myModel: Optional[NNModel] = None
loaded_model_path: Optional[str] = None
loaded_model_key: Optional[ModelCacheKey] = None
_MODEL_CACHE: Dict[ModelCacheKey, NNModel] = {}
_KERAS_MODEL_CACHE: Dict[str, Any] = {}
searchModel: Optional[NNModel] = None
search_model_key: Optional[ModelCacheKey] = None
referenceModel: Optional[Any] = None
reference_model_path: Optional[str] = None


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
    global searchModel, search_model_key

    if role == "default":
        myModel = model
        loaded_model_key = model_key
        loaded_model_path = model_path
        return
    if role == "search":
        searchModel = model
        search_model_key = model_key
        return
    raise ValueError(f"Unsupported model role: {role}")


def _load_reference_model(model_path: str) -> Any:
    global referenceModel, reference_model_path

    cached = _KERAS_MODEL_CACHE.get(model_path)
    if cached is None:
        cached = load_model_with_compat(model_path)
        _KERAS_MODEL_CACHE[model_path] = cached
        log.info(
            "Loaded Keras reference model '%s' with input_shape=%s",
            Path(model_path).stem,
            cached.input_shape,
        )
    referenceModel = cached
    reference_model_path = model_path
    return cached


def init_reference_model(model_path) -> None:
    _load_reference_model(str(Path(model_path).resolve()))


def init_model(
    model_path,
    *,
    ternary_simplification: bool = False,
    ternary_threshold_scale: float = 0.75,
    role: ModelRole = "default",
):
    resolved_model_path = str(Path(model_path).resolve())
    model_key: ModelCacheKey = (
        resolved_model_path,
        bool(ternary_simplification),
        float(ternary_threshold_scale),
    )
    keras_model = _load_reference_model(resolved_model_path)
    cached = _MODEL_CACHE.get(model_key)
    if cached is not None:
        _assign_model_for_role(role, cached, model_key, resolved_model_path)
        return
    model_stem = Path(resolved_model_path).stem
    keras_model._name = model_stem
    keras_model.summary()
    layers, inbound_map = _collect_layers_and_inbound(keras_model)
    my_model = NNModel(
        ternary_simplification=ternary_simplification,
        ternary_threshold_scale=ternary_threshold_scale,
    )
    input_layer_names = _collect_input_names(keras_model)
    my_model.register_input_names(input_layer_names)
    my_model.input_shape = _get_reference_input_shape(keras_model)
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


def _get_reference_input_shape(model: Any) -> Tuple[int, ...]:
    input_shape = getattr(model, "input_shape", None)
    if (
        not isinstance(input_shape, tuple)
        or len(input_shape) < 2
        or any(dim is None for dim in input_shape[1:])
    ):
        raise ValueError(
            "Keras reference prediction requires one fully-defined tensor input; "
            f"got input_shape={input_shape!r}."
        )
    return tuple(int(dim) for dim in input_shape[1:])


def _build_tensor_input(input_shape: Tuple[int, ...], data: Dict[str, Any]) -> list:
    tensor_input = np.zeros(input_shape, dtype=object)
    for index in itertools.product(*(range(dim) for dim in input_shape)):
        key = "v_" + "_".join(str(axis) for axis in index)
        tensor_input[index] = data[key]
    return tensor_input.tolist()


def _prediction_to_label(predictions: np.ndarray) -> int:
    if not np.isfinite(predictions.astype(np.float64, copy=False)).all():
        raise ValueError("Keras reference model output contains NaN or Inf")
    if predictions.ndim == 1 and predictions.shape == (1,):
        return int(predictions[0] > 0.5)
    if predictions.ndim == 2 and predictions.shape[0] == 1:
        if predictions.shape[1] == 1:
            return int(predictions[0, 0] > 0.5)
        if predictions.shape[1] > 1:
            return int(np.argmax(predictions[0]))
    raise ValueError(
        "Keras reference model must return one binary or multiclass prediction; "
        f"got output shape {predictions.shape}."
    )


def predict_reference_array(array: np.ndarray) -> Tuple[np.ndarray, int]:
    if referenceModel is None:
        raise RuntimeError("Keras reference model not initialized. Call init_model() first.")

    expected_shape = _get_reference_input_shape(referenceModel)
    reference_input = np.asarray(array, dtype=np.float32)
    if reference_input.shape != expected_shape:
        raise ValueError(
            f"Keras reference input shape {reference_input.shape} does not match {expected_shape}."
        )
    if not np.isfinite(reference_input).all():
        raise ValueError("Keras reference input contains NaN or Inf")

    predictions = np.asarray(
        referenceModel.predict(reference_input[np.newaxis, ...], verbose=0),
    )
    label = _prediction_to_label(predictions)
    if predictions.ndim == 2:
        output = predictions[0]
    else:
        output = predictions.copy()
    log.info("Keras reference prediction complete (class=%s)", label)
    return np.asarray(output), label


def predict_reference(**data):
    if referenceModel is None:
        raise RuntimeError("Keras reference model not initialized. Call init_model() first.")
    input_shape = _get_reference_input_shape(referenceModel)
    tensor_input = _build_tensor_input(input_shape, data)
    _, label = predict_reference_array(np.asarray(tensor_input, dtype=np.float32))
    return label


def _predict_with_model(model: Optional[NNModel], *, require_finite: bool = False, **data):
    if model is None or model.input_shape is None:
        raise RuntimeError("Model not initialized. Call init_model() before predict().")

    model = cast(NNModel, model)
    input_shape = cast(Tuple[int, ...], model.input_shape)
    tensor_input = _build_tensor_input(input_shape, data)

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


__all__ = [
    "AddClsToken",
    "AddPositionEmbedding",
    "DropPath",
    "ExtractClsToken",
    "SequencePooling",
    "init_model",
    "init_reference_model",
    "predict",
    "predict_reference",
    "predict_reference_array",
    "predict_search",
]
