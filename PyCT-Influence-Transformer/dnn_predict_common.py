import itertools
import json
import logging
import re
from pathlib import Path
from typing import Optional, cast, Tuple

import keras
import numpy as np
from dnnct.myDNN import NNModel

from libct.position import register_layer_number_mapping, to_Keras_layer_number

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

log = logging.getLogger("ct.model")


myModel: Optional[NNModel] = None
loaded_model_path: Optional[str] = None


class AddPositionEmbedding(keras.layers.Layer):
    """Inference-compatible positional embedding layer."""

    def build(self, input_shape):
        if len(input_shape) < 3:
            raise ValueError("AddPositionEmbedding expects rank-3 inputs [B, L, D].")
        seq_len = int(input_shape[1])
        dim = int(input_shape[2])
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, seq_len, dim),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.pos_embedding


class AddClsToken(keras.layers.Layer):
    """Prepend a learnable CLS token to a token sequence."""

    def build(self, input_shape):
        if len(input_shape) < 3:
            raise ValueError("AddClsToken expects rank-3 inputs [B, L, D].")
        dim = int(input_shape[2])
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, dim),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        if hasattr(keras, "ops"):
            batch = keras.ops.shape(inputs)[0]
            cls = keras.ops.broadcast_to(
                self.cls_token,
                (batch, 1, int(self.cls_token.shape[-1])),
            )
            return keras.ops.concatenate([cls, inputs], axis=1)
        import tensorflow as tf

        batch = tf.shape(inputs)[0]
        cls = tf.repeat(self.cls_token, repeats=batch, axis=0)
        return tf.concat([cls, inputs], axis=1)


class ExtractClsToken(keras.layers.Layer):
    """Return the CLS token (index 0) from a token sequence."""

    def call(self, inputs):
        return inputs[:, 0, :]


class DropPath(keras.layers.Layer):
    """DropPath as identity during inference."""

    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)

    def call(self, inputs, training=None):
        del training
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


class SequencePooling(keras.layers.Layer):
    """Sequence pooling with learnable token attention."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Keep sublayer naming stable with saved checkpoints.
        self.score = keras.layers.Dense(1, name="dense_16")

    def call(self, inputs):
        scores = self.score(inputs)
        if hasattr(keras, "ops"):
            weights = keras.ops.softmax(scores, axis=1)
            return keras.ops.sum(weights * inputs, axis=1)
        import tensorflow as tf

        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(weights * inputs, axis=1)


def _inject_keras_resnet_libct() -> None:
    # keras_resnet BatchNormalization may reference module-level `libct` during graph execution.
    try:
        import libct as libct_module
        import keras_resnet.layers as kr_layers

        setattr(kr_layers, "libct", libct_module)
        if hasattr(kr_layers, "BatchNormalization"):
            setattr(kr_layers.BatchNormalization, "libct", libct_module)
    except Exception:
        pass


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
    excluded = {"Dropout", "InputLayer", "Embedding"}
    layers = [l for l in model.layers if type(l).__name__ not in excluded]
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


def _get_resnet_custom_objects():
    custom_objects = {}
    try:
        _inject_keras_resnet_libct()
        from keras_resnet import models as kr_models
        from keras_resnet.layers import BatchNormalization as ResNetBatchNorm

        ResNet2D18 = getattr(kr_models, "ResNet2D18", None)
        if ResNet2D18 is None:
            return custom_objects

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
    except Exception as exc:
        log.debug("Unable to build keras_resnet custom_objects: %r", exc)
    return custom_objects


def _get_transformer_custom_objects():
    layer_map = {
        "AddClsToken": AddClsToken,
        "AddPositionEmbedding": AddPositionEmbedding,
        "DropPath": DropPath,
        "ExtractClsToken": ExtractClsToken,
        "SequencePooling": SequencePooling,
        "Custom>AddClsToken": AddClsToken,
        "Custom>AddPositionEmbedding": AddPositionEmbedding,
        "Custom>DropPath": DropPath,
        "Custom>ExtractClsToken": ExtractClsToken,
        "Custom>SequencePooling": SequencePooling,
    }
    return layer_map


def _needs_resnet_fallback(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "Unknown layer: 'ResNet2D" in msg
        or "missing 1 required positional argument: 'inputs'" in msg
        or "Unable to revive model from config" in msg and "ResNet2D" in msg
    )


def _read_model_config(model_path: str) -> Optional[dict]:
    if h5py is None:
        return None
    try:
        with h5py.File(model_path, "r") as handle:
            config = handle.attrs.get("model_config")
            if config is None:
                return None
            if isinstance(config, bytes):
                config = config.decode("utf-8")
            return json.loads(config)
    except Exception:
        return None


def _extract_specs_from_weights(model_path: str) -> dict:
    specs: dict = {}
    if h5py is None:
        return specs
    try:
        with h5py.File(model_path, "r") as handle:
            weights_group = handle.get("model_weights")
            if weights_group is None:
                if "layer_names" in handle.attrs:
                    weights_group = handle
                else:
                    return specs

            input_channels = None
            dense_candidates = []

            def visitor(name, obj):
                nonlocal input_channels
                if not isinstance(obj, h5py.Dataset):
                    return
                if not (
                    name.endswith("kernel:0")
                    or name.endswith("W:0")
                    or name.endswith("weights:0")
                ):
                    return
                shape = obj.shape
                if len(shape) == 4 and input_channels is None:
                    input_channels = int(shape[2])
                elif len(shape) == 2:
                    dense_candidates.append(int(shape[1]))

            weights_group.visititems(visitor)

            if input_channels is not None:
                specs["input_channel_only"] = input_channels
            if dense_candidates:
                specs["num_classes"] = min(dense_candidates)
    except Exception:
        return specs
    return specs


def _guess_default_input_shape(model_path: str, channel_hint: Optional[int]) -> Optional[Tuple[int, int, int]]:
    name = Path(model_path).stem.lower()
    if "mnist" in name:
        return (28, 28, channel_hint if channel_hint is not None else 1)
    if "cifar" in name:
        return (32, 32, channel_hint if channel_hint is not None else 3)
    if "imagenet" in name:
        return (224, 224, channel_hint if channel_hint is not None else 3)
    return None


def _guess_default_num_classes(model_path: str) -> Optional[int]:
    name = Path(model_path).stem.lower()
    if "mnist" in name or "fashion" in name or "cifar10" in name:
        return 10
    return None


def _infer_resnet_specs(model_path: str) -> Optional[dict]:
    specs: dict = {}
    config = _read_model_config(model_path)
    if config:
        class_name = str(config.get("class_name", ""))
        matches = re.findall(r"(\d+)", class_name)
        if matches:
            specs["depth"] = int(matches[-1])

    specs.update(_extract_specs_from_weights(model_path))

    if "depth" not in specs:
        match = re.search(r"resnet(\d+)", Path(model_path).stem.lower())
        if match:
            specs["depth"] = int(match.group(1))

    if "input_shape" not in specs:
        guessed_shape = _guess_default_input_shape(
            model_path,
            channel_hint=specs.get("input_channel_only")
            if isinstance(specs.get("input_channel_only"), int)
            else None,
        )
        if guessed_shape is not None:
            specs["input_shape"] = guessed_shape

    if "num_classes" not in specs:
        guessed_classes = _guess_default_num_classes(model_path)
        if guessed_classes is not None:
            specs["num_classes"] = guessed_classes

    if "depth" not in specs or "input_shape" not in specs or "num_classes" not in specs:
        return None
    return specs


def _build_resnet_model(depth: int, input_shape: Tuple[int, ...], num_classes: int):
    from keras_resnet import models as kr_models

    builder = getattr(kr_models, f"ResNet2D{depth}", None)
    if builder is None:
        builder = getattr(kr_models, f"ResNet{depth}", None)
    if builder is None:
        raise ValueError(f"Cannot find a keras_resnet builder for ResNet depth={depth}.")

    inputs = keras.layers.Input(shape=tuple(int(v) for v in input_shape))
    build_kwargs_candidates = (
        {"inputs": inputs, "classes": int(num_classes), "include_top": True, "freeze_bn": False},
        {"inputs": inputs, "classes": int(num_classes), "freeze_bn": False},
        {"inputs": inputs, "classes": int(num_classes)},
    )
    last_error: Optional[Exception] = None
    for kwargs in build_kwargs_candidates:
        try:
            return builder(**kwargs)
        except TypeError as exc:
            last_error = exc
    raise TypeError(
        f"Failed to instantiate ResNet builder '{builder.__name__}' with inferred specs."
    ) from last_error


def _load_keras_model(model_path):
    custom_objects = _get_resnet_custom_objects()
    custom_objects.update(_get_transformer_custom_objects())
    try:
        if custom_objects:
            return keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return keras.models.load_model(model_path, compile=False)
    except (ValueError, TypeError) as exc:
        if not _needs_resnet_fallback(exc):
            raise

        specs = _infer_resnet_specs(model_path)
        if specs is None:
            raise ValueError(
                f"Failed to rebuild ResNet model from '{model_path}': insufficient inferred specs."
            ) from exc

        rebuilt = _build_resnet_model(
            int(specs["depth"]),
            tuple(int(v) for v in specs["input_shape"]),
            int(specs["num_classes"]),
        )
        rebuilt.load_weights(model_path)
        return rebuilt


def init_model(model_path):
    global myModel, loaded_model_path
    if myModel is not None and loaded_model_path == model_path:
        return
    if loaded_model_path is not None and loaded_model_path != model_path:
        keras.backend.clear_session()
    model = _load_keras_model(model_path)
    model_stem = Path(model_path).stem
    model._name = model_stem  # improve model.summary() readability
    model.summary()
    log.info("Loaded model '%s' with input_shape=%s", model_stem, model.input_shape)
    layers, inbound_map = _collect_layers_and_inbound(model)
    myModel = NNModel()
    input_layer_names = [layer.name for layer in model.layers if type(layer).__name__ == "InputLayer"]
    myModel.register_input_names(input_layer_names)

    # 1: is because 1st dim of input shape of Keras model is batch size (None)
    myModel.input_shape = model.input_shape[1:]
    myLayerCount = 0
    for i, layer in enumerate(layers):
        inbound_names = inbound_map.get(layer.name, [])
        numberOfMyLayers = myModel.addLayer(layer, inbound_names=inbound_names)
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
	if myModel is None or myModel.input_shape is None:
		raise RuntimeError("Model not initialized. Call init_model() before predict().")

	model = cast(NNModel, myModel)
	input_shape = cast(Tuple[int, ...], model.input_shape)
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

	out_val = model.forward(X)
	log.debug("Completed forward pass for input_shape=%s", input_shape)

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

	log.info("Forward prediction complete (class=%s)", ret_class)
	return ret_class
