from __future__ import annotations

import functools
import json
import logging
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Tuple

import numpy as np
import shap
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Sequential, load_model

from libct.constraint import Constraint

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

if TYPE_CHECKING:
    from typing import Iterable

    class PositionedConstraint(Tuple[Constraint, Tuple[int, Tuple[int, ...]]]):
        ...


__all__ = [
    "ShapValuesCalculator",
    "ShapValuesComparator",
    "pop_last_constraint",
    "pop_first_constraint",
    "pop_the_most_important_constraint",
]

log = logging.getLogger("ct.shap")
_KERAS_RESNET_IMPORT_ERROR: Exception | None = None


class AddPositionEmbedding(tf.keras.layers.Layer):
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


class AddClsToken(tf.keras.layers.Layer):
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
        batch = tf.shape(inputs)[0]
        cls = tf.repeat(self.cls_token, repeats=batch, axis=0)
        return tf.concat([cls, inputs], axis=1)


class ExtractClsToken(tf.keras.layers.Layer):
    """Return the CLS token (index 0) from a token sequence."""

    def call(self, inputs):
        return inputs[:, 0, :]


class DropPath(tf.keras.layers.Layer):
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


class SequencePooling(tf.keras.layers.Layer):
    """Sequence pooling with learnable token attention."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Keep sublayer naming stable with saved checkpoints.
        self.score = tf.keras.layers.Dense(1, name="dense_16")

    def call(self, inputs):
        scores = self.score(inputs)
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
        # Best-effort compatibility hook; model loading can still proceed without it.
        pass


def _get_resnet_custom_objects() -> Dict[str, object]:
    global _KERAS_RESNET_IMPORT_ERROR
    custom_objects: Dict[str, object] = {}
    _KERAS_RESNET_IMPORT_ERROR = None
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
        _KERAS_RESNET_IMPORT_ERROR = exc
        log.debug("Unable to build keras_resnet custom_objects: %r", exc)
        return custom_objects
    return custom_objects


def _get_transformer_custom_objects() -> Dict[str, object]:
    return {
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


def _needs_resnet_fallback(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "Unknown layer: 'ResNet2D" in msg
        or "missing 1 required positional argument: 'inputs'" in msg
        or "Unable to revive model from config" in msg and "ResNet2D" in msg
    )


def _read_model_config(model_path: str) -> dict | None:
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


def _extract_specs_from_weights(model_path: str) -> Dict[str, int]:
    specs: Dict[str, int] = {}
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

            input_channels: Optional[int] = None
            dense_candidates: list[int] = []

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
                    # Common TF format: (H, W, In, Out)
                    input_channels = int(shape[2])
                elif len(shape) == 2:
                    # Common Dense kernel format: (In, Out)
                    dense_candidates.append(int(shape[1]))

            weights_group.visititems(visitor)

            if input_channels is not None:
                specs["input_channel_only"] = input_channels
            if dense_candidates:
                # Prefer small class-count heads (e.g., 10 for MNIST) over hidden dims.
                specs["num_classes"] = min(dense_candidates)
    except Exception:
        return specs
    return specs


def _guess_default_input_shape(model_path: str, channel_hint: Optional[int]) -> Optional[Tuple[int, int, int]]:
    name = Path(model_path).stem.lower()
    if "mnist" in name:
        channels = channel_hint if channel_hint is not None else 1
        return (28, 28, channels)
    if "cifar" in name:
        channels = channel_hint if channel_hint is not None else 3
        return (32, 32, channels)
    if "imagenet" in name:
        channels = channel_hint if channel_hint is not None else 3
        return (224, 224, channels)
    return None


def _guess_default_num_classes(model_path: str) -> Optional[int]:
    name = Path(model_path).stem.lower()
    if "mnist" in name or "fashion" in name or "cifar10" in name:
        return 10
    return None


def _infer_resnet_specs(
    model_path: str,
    *,
    input_shape_override: Optional[Tuple[int, ...]] = None,
    num_classes_override: Optional[int] = None,
) -> Dict[str, int | Tuple[int, ...]] | None:
    specs: Dict[str, int | Tuple[int, ...]] = {}
    config = _read_model_config(model_path)
    if config:
        class_name = str(config.get("class_name", ""))
        matches = re.findall(r"(\d+)", class_name)
        if matches:
            specs["depth"] = int(matches[-1])

    weight_specs = _extract_specs_from_weights(model_path)
    specs.update(weight_specs)

    if input_shape_override is not None:
        specs["input_shape"] = tuple(int(v) for v in input_shape_override)
    if num_classes_override is not None:
        specs["num_classes"] = int(num_classes_override)

    if "depth" not in specs:
        match = re.search(r"resnet(\d+)", Path(model_path).stem.lower())
        if match:
            specs["depth"] = int(match.group(1))

    if "input_shape" not in specs:
        guessed_shape = _guess_default_input_shape(
            model_path,
            channel_hint=specs.get("input_channel_only") if isinstance(specs.get("input_channel_only"), int) else None,
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


def _build_resnet_model(depth: int, input_shape: Tuple[int, ...], num_classes: int) -> Model:
    try:
        from keras_resnet import models as kr_models
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Loading ResNet weights requires `keras-resnet`. "
            "Install with: python -m pip install keras-resnet"
        ) from exc

    builder = getattr(kr_models, f"ResNet2D{depth}", None)
    if builder is None:
        builder = getattr(kr_models, f"ResNet{depth}", None)
    if builder is None:
        raise ValueError(f"Cannot find a keras_resnet builder for ResNet depth={depth}.")

    inputs = Input(shape=tuple(int(v) for v in input_shape))
    build_kwargs_candidates = (
        {"inputs": inputs, "classes": int(num_classes), "include_top": True, "freeze_bn": False},
        {"inputs": inputs, "classes": int(num_classes), "freeze_bn": False},
        {"inputs": inputs, "classes": int(num_classes)},
    )
    last_error: Exception | None = None
    for kwargs in build_kwargs_candidates:
        try:
            return builder(**kwargs)
        except TypeError as exc:
            last_error = exc
    raise TypeError(
        f"Failed to instantiate ResNet builder '{builder.__name__}' with inferred specs."
    ) from last_error


def _load_model_with_compat(
    model_path: str,
    *,
    input_shape_override: Optional[Tuple[int, ...]] = None,
    num_classes_override: Optional[int] = None,
) -> Sequential | Model:
    custom_objects = _get_resnet_custom_objects()
    custom_objects.update(_get_transformer_custom_objects())
    try:
        if custom_objects:
            return load_model(model_path, custom_objects=custom_objects, compile=False)
        return load_model(model_path, compile=False)
    except (ValueError, TypeError) as exc:
        msg = str(exc)
        if not _needs_resnet_fallback(exc):
            raise

        specs = _infer_resnet_specs(
            model_path,
            input_shape_override=input_shape_override,
            num_classes_override=num_classes_override,
        )
        if specs is None:
            import_hint = "Please ensure `keras-resnet` is installed and importable in this environment."
            if _KERAS_RESNET_IMPORT_ERROR is not None:
                import_hint += f" Import error: {_KERAS_RESNET_IMPORT_ERROR!r}"
            raise ValueError(
                f"{msg} Failed to rebuild ResNet model for '{model_path}'. "
                f"{import_hint}"
            ) from exc

        depth = int(specs["depth"])
        input_shape = tuple(int(v) for v in specs["input_shape"])
        num_classes = int(specs["num_classes"])
        rebuilt = _build_resnet_model(depth, input_shape, num_classes)
        try:
            rebuilt.load_weights(model_path)
        except Exception as load_exc:
            raise ValueError(
                f"ResNet fallback rebuild succeeded, but loading weights failed for '{model_path}': {load_exc}"
            ) from load_exc
        return rebuilt


class ShapValuesCalculator:
    """Load, compute, and cache SHAP values for a given model/input pair."""

    def __init__(
        self,
        *,
        model_path: str,
        background_dataset: np.ndarray,
        input_data: np.ndarray,
        idx: int,
        explainer_type: Literal["gradient", "kernel"] = "gradient",
        output_root: str = "shap_value_all_layer",
    ) -> None:
        self.model_path = model_path
        self.model_name = Path(model_path).stem
        self.background_dataset = background_dataset
        self.input = input_data
        self.idx = idx
        self.explainer_type = explainer_type
        self.output_dir = Path(output_root) / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Avoid recompiling saved models (optimizer state may be incompatible across TF/Keras versions).
        inferred_input_shape: Optional[Tuple[int, ...]] = None
        input_arr = np.asarray(input_data)
        if input_arr.ndim >= 2:
            inferred_input_shape = tuple(int(v) for v in input_arr.shape[1:])

        self._model = _load_model_with_compat(
            model_path,
            input_shape_override=inferred_input_shape,
        )
        self._shap_values: Dict[str, float] | None = None
        self._cache_meta = self._read_background_meta()
        self._tracked_layers = [
            layer
            for layer in self._model.layers
            if type(layer).__name__ not in ("InputLayer", "Dropout", "Embedding")
        ]
        self._layer_count = (
            len(self._model.layers)
            if isinstance(self._model, Sequential)
            else len(self._model.layers) - 1
        )
        self._layerwise_enabled = isinstance(self._model, Sequential)

    @property
    def model(self) -> Sequential | Model:
        return self._model

    @property
    def layer_count(self) -> int:
        return self._layer_count

    @property
    def cache_path(self) -> Path:
        return self.output_dir / f"shap_value_{self.idx}.json"

    def ensure(
        self,
        *,
        assume_cached: bool = False,
        force_refresh: bool = False,
    ) -> Dict[str, float]:
        """Return SHAP values, computing and persisting them when needed."""
        if self._shap_values is not None and not force_refresh:
            return self._shap_values

        cache_exists = self.cache_path.is_file()
        should_load_cache = (
            cache_exists
            and not force_refresh
            and (assume_cached or self._shap_values is None)
        )

        if should_load_cache:
            try:
                self._shap_values = self._load_cache()
                return self._shap_values
            except (json.JSONDecodeError, OSError):
                # fall back to recomputing if cache is corrupt
                pass

        shap_values = self._compute_shap_values()
        self._save_cache(shap_values)
        self._shap_values = shap_values
        return shap_values

    def _load_cache(self) -> Dict[str, float]:
        with self.cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "values" in data:
            self._cache_meta = data.get("__meta__") or self._cache_meta
            values = data.get("values", {})
        else:
            values = data
        if not isinstance(values, dict):
            raise TypeError(f"Expected SHAP cache {self.cache_path} to be a JSON dict.")
        return {str(k): float(v) for k, v in values.items()}

    def _save_cache(self, shap_values: Dict[str, float]) -> None:
        with self.cache_path.open("w", encoding="utf-8") as handle:
            if self._cache_meta:
                payload = {"__meta__": self._cache_meta, "values": shap_values}
                json.dump(payload, handle)
            else:
                json.dump(shap_values, handle)

    def _read_background_meta(self) -> Dict[str, int]:
        meta: Dict[str, int] = {}
        try:
            meta["background_per_class"] = int(os.environ.get("PYCT_BG_PER_CLASS", ""))
        except ValueError:
            pass
        try:
            meta["background_seed"] = int(os.environ.get("PYCT_BG_SEED", ""))
        except ValueError:
            pass
        return meta

    def _compute_shap_values(self) -> Dict[str, float]:
        shap_values: Dict[str, float] = {}
        if not self._layerwise_enabled:
            # Functional/DAG models (e.g., ResNet) are not safe for "drop first layer"
            # slicing. Compute reliable input-level SHAP only.
            self._calculate_layer_shap_values(
                shap_values,
                self._model,
                self.background_dataset,
                self.input,
                0,
            )
            # Add per-layer branch influence values for Functional models.
            self._calculate_functional_layer_branch_influence(shap_values)
            return shap_values

        trimmed_model = self._model
        transformed_background = self.background_dataset
        transformed_input = self.input

        for layer_number in range(self._layer_count):
            self._calculate_layer_shap_values(
                shap_values,
                trimmed_model,
                transformed_background,
                transformed_input,
                layer_number,
            )
            transformed_input = self.apply_one_layer(trimmed_model, transformed_input)
            transformed_background = self.apply_one_layer_to_dataset(
                trimmed_model, transformed_background
            )
            if layer_number == self._layer_count - 1:
                break
            trimmed_model = self.without_first_layer(trimmed_model)

        return shap_values

    def _calculate_functional_layer_branch_influence(
        self,
        shap_values: Dict[str, float],
    ) -> None:
        # For Functional/DAG models, estimate per-neuron branch influence as
        # gradient * (activation - background_mean_activation).
        if not self._tracked_layers:
            return

        output_specs: list[tuple[int, object]] = []
        for layer_index, layer in enumerate(self._tracked_layers):
            layer_output = getattr(layer, "output", None)
            if layer_output is None or isinstance(layer_output, (list, tuple)):
                continue
            output_specs.append((layer_index, layer_output))

        if not output_specs:
            return

        feature_outputs = [tensor for _, tensor in output_specs]
        feature_outputs.append(self._model.output)
        feature_model = Model(inputs=self._model.inputs, outputs=feature_outputs)

        input_array = np.asarray(self.input, dtype=np.float32)
        background_array = np.asarray(self.background_dataset, dtype=np.float32)
        if input_array.ndim == background_array.ndim - 1:
            input_array = np.expand_dims(input_array, axis=0)

        x = tf.convert_to_tensor(input_array, dtype=tf.float32)
        background = tf.convert_to_tensor(background_array, dtype=tf.float32)

        background_out = feature_model(background, training=False)
        if not isinstance(background_out, (list, tuple)):
            background_out = [background_out]
        background_acts = list(background_out[:-1])

        with tf.GradientTape() as tape:
            input_out = feature_model(x, training=False)
            if not isinstance(input_out, (list, tuple)):
                input_out = [input_out]
            input_acts = list(input_out[:-1])
            logits = input_out[-1]

            for act in input_acts:
                tape.watch(act)
            if logits.shape.rank is not None and logits.shape[-1] == 1:
                target = tf.reduce_mean(logits[:, 0])
            else:
                top_class = tf.cast(tf.argmax(logits[0]), tf.int32)
                target = tf.reduce_mean(tf.gather(logits, top_class, axis=1))

        grads = tape.gradient(target, list(input_acts))

        for (layer_index, _), act_input, act_background, grad in zip(
            output_specs,
            input_acts,
            background_acts,
            grads,
        ):
            if grad is None:
                continue

            baseline = tf.reduce_mean(act_background, axis=0)
            influence = grad[0] * (act_input[0] - baseline)
            influence_np = np.asarray(influence.numpy())

            for indices, value in np.ndenumerate(influence_np):
                shap_values[self.get_position_key(layer_index, indices)] = float(value)

    def _calculate_layer_shap_values(
        self,
        shap_values: Dict[str, float],
        model: Sequential | Model,
        background_dataset: np.ndarray,
        input_data: np.ndarray,
        layer_number: int,
    ) -> None:
        if self.explainer_type == "gradient":
            bg_for_shap: Iterable[np.ndarray]
            if isinstance(background_dataset, list):
                bg_for_shap = background_dataset
            else:
                bg_for_shap = [background_dataset]

            explainer = shap.GradientExplainer(model, bg_for_shap)
            gradients = explainer.shap_values(input_data)
            average_gradients = self._reduce_gradient_shap_values(gradients, input_data)

            for indices, value in np.ndenumerate(average_gradients):
                shap_values[
                    self.get_position_key(layer_number - 1, indices)
                ] = float(value)
        else:
            should_flatten = len(input_data.shape) > 2
            original_shape: Tuple[int, ...] | None = None
            kernel_model = model
            kernel_input = input_data
            kernel_background = background_dataset

            if should_flatten:
                original_shape = input_data.shape
                (
                    kernel_model,
                    kernel_input,
                    kernel_background,
                ) = self._flatten_everything(model, input_data, background_dataset)
                kernel_input = np.expand_dims(kernel_input, axis=0)

            explainer = shap.KernelExplainer(kernel_model, kernel_background)
            kernel_shap_values = explainer.shap_values(kernel_input)

            for idx, value in np.ndenumerate(kernel_shap_values):
                flat_index = idx[1]
                if should_flatten and original_shape is not None:
                    indices = self._unflatten_index(flat_index, original_shape)
                else:
                    indices = (flat_index,)
                shap_values[
                    self.get_position_key(layer_number - 1, indices)
                ] = float(value)

    @staticmethod
    def _reduce_gradient_shap_values(
        gradients: np.ndarray | list[np.ndarray],
        input_data: np.ndarray,
    ) -> np.ndarray:
        if isinstance(gradients, list):
            grad_arr = np.asarray(gradients)
            # average over output dimensions (classes/heads)
            grad_arr = np.mean(grad_arr, axis=0)
        else:
            grad_arr = np.asarray(gradients)
        # remove batch axis
        if grad_arr.ndim >= 1 and input_data.ndim >= 1 and grad_arr.shape[0] == input_data.shape[0]:
            grad_arr = np.mean(grad_arr, axis=0)
        grad_arr = np.squeeze(grad_arr)
        return grad_arr

    @staticmethod
    def without_first_layer(original_model: Sequential | Model) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            new_model = Sequential()
            for layer in original_model.layers[1:]:
                new_model.add(layer)
            new_model.build(original_model.layers[1].input_shape)
            return new_model

        new_input = original_model.layers[2].input
        new_output = original_model.layers[-1].output
        return Model(inputs=new_input, outputs=new_output)

    @staticmethod
    def get_model_with_only_first_layer(
        original_model: Sequential | Model,
    ) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            return Sequential(original_model.layers[:1])

        return Model(
            inputs=original_model.layers[1].input,
            outputs=original_model.layers[1].output,
        )

    @classmethod
    def apply_one_layer_to_dataset(
        cls, original_model: Sequential | Model, dataset: np.ndarray
    ) -> np.ndarray:
        model_with_only_first_layer = cls.get_model_with_only_first_layer(original_model)
        model_with_only_first_layer.build(original_model.layers[0].input_shape)
        return model_with_only_first_layer.predict(dataset)

    @classmethod
    def apply_one_layer(
        cls, original_model: Sequential | Model, original_input: np.ndarray
    ) -> np.ndarray:
        return cls.get_model_with_only_first_layer(original_model).predict(
            original_input
        )

    @staticmethod
    def get_position_key(layer_number: int, indices: Tuple[int, ...]) -> str:
        key = str(layer_number)
        for index in indices:
            key += f"_{index}"
        return key

    def _flatten_everything(
        self,
        model: Sequential | Model,
        input_data: np.ndarray,
        background_dataset: np.ndarray,
    ) -> Tuple[Sequential | Model, np.ndarray, np.ndarray]:
        flattened_input = self._flatten(input_data)
        flattened_background = np.array(
            [self._flatten(data_point) for data_point in background_dataset]
        )
        reshaped_model = self._prepend_unflatten_layer(model, input_data.shape[1:])
        return reshaped_model, flattened_input, flattened_background

    @staticmethod
    def _flatten(array: np.ndarray) -> np.ndarray:
        return np.reshape(array, [-1])

    @staticmethod
    def _prepend_unflatten_layer(
        model: Sequential | Model, original_input_shape: Tuple[int, ...]
    ) -> Sequential:
        new_input_shape = (int(np.prod(original_input_shape)),)
        new_model = Sequential()
        new_model.add(Input(new_input_shape))
        new_model.add(Reshape(original_input_shape))
        for layer in model.layers:
            new_model.add(layer)
        new_model.build()
        return new_model

    @staticmethod
    def _unflatten_index(
        flattened_index: int, original_input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        indices = []
        current_modular = original_input_shape[-1]
        current_divisor = 1
        reversed_shape = tuple(reversed(original_input_shape))
        for i, dim in enumerate(reversed_shape):
            indices.append(flattened_index % current_modular // current_divisor)
            current_divisor = current_modular
            if i + 1 < len(reversed_shape):
                current_modular *= reversed_shape[i + 1]
        return tuple(reversed(indices))


def _infer_layer_count_from_cached_values(shap_values: Dict[str, float]) -> int:
    max_layer = -1
    for key in shap_values.keys():
        token = key.split("_", 1)[0]
        if token.lstrip("-").isdigit():
            max_layer = max(max_layer, int(token))
    return max_layer + 1 if max_layer >= 0 else 1


def _load_cached_shap_values(cache_path: Path) -> Dict[str, float]:
    with cache_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "values" in data:
        values = data.get("values", {})
    else:
        values = data
    if not isinstance(values, dict):
        raise TypeError(f"Expected SHAP cache {cache_path} to be a JSON dict.")
    return {str(k): float(v) for k, v in values.items()}


class ShapValuesComparator:
    """Comparator for prioritising constraints based on SHAP influence."""

    def __init__(
        self,
        *,
        model_path,
        background_dataset,
        input,
        idx,
        shap_value_pre_calculated,
        explainer_type: Literal["gradient", "kernel"] = "gradient",
        output_root: str = "shap_value_all_layer",
    ) -> None:
        cache_path = Path(output_root) / Path(model_path).stem / f"shap_value_{idx}.json"
        if shap_value_pre_calculated and cache_path.is_file():
            try:
                self.shap_values = _load_cached_shap_values(cache_path)
                self.layer_count = _infer_layer_count_from_cached_values(self.shap_values)
                self.model = None
                self.calculator = None
                log.debug("Loaded SHAP cache without model load: %s", cache_path)
                return
            except Exception as exc:
                log.warning(
                    "Failed to load SHAP cache '%s' in fast path (%r); falling back to model-backed path.",
                    cache_path,
                    exc,
                )

        self.calculator = ShapValuesCalculator(
            model_path=model_path,
            background_dataset=background_dataset,
            input_data=input,
            idx=idx,
            explainer_type=explainer_type,
            output_root=output_root,
        )
        self.shap_values = self.calculator.ensure(
            assume_cached=shap_value_pre_calculated,
            force_refresh=not shap_value_pre_calculated,
        )
        self.model = self.calculator.model
        self.layer_count = self.calculator.layer_count

    def compare(
        self,
        positioned_constraint_1: PositionedConstraint,
        positioned_constraint_2: PositionedConstraint,
    ) -> float:
        constraint_1, (row_number_1, indices_1) = positioned_constraint_1
        constraint_2, (row_number_2, indices_2) = positioned_constraint_2
        return self.get_shap_influence(row_number_1, indices_1) - self.get_shap_influence(
            row_number_2, indices_2
        )

    def get_shap_influence(
        self,
        layer_number: int,
        indices: tuple[int, ...] | list[tuple[int, ...]],
    ) -> float:
        if layer_number == self.layer_count - 1:
            return float("-inf")
        if isinstance(indices, list):
            total = 0.0
            for ids in indices:
                total += self._lookup(layer_number, ids)
            return total / len(indices)
        return self._lookup(layer_number, indices)

    def _lookup(self, layer_number: int, indices: tuple[int, ...]) -> float:
        key = self.get_position_key(layer_number, indices)
        if key in self.shap_values:
            return self.shap_values[key]
        alt_key = self.get_position_key(layer_number - 1, indices)
        if alt_key in self.shap_values:
            return self.shap_values[alt_key]
        if len(indices) > 1:
            spatial = indices[:-1]
            spatial_key = self.get_position_key(layer_number, spatial)
            if spatial_key in self.shap_values:
                return self.shap_values[spatial_key]
            spatial_alt_key = self.get_position_key(layer_number - 1, spatial)
            if spatial_alt_key in self.shap_values:
                return self.shap_values[spatial_alt_key]
        return 0.0

    @staticmethod
    def get_position_key(layer_number: int, indices: Tuple[int, ...]) -> str:
        return ShapValuesCalculator.get_position_key(layer_number, indices)

    @staticmethod
    def pop_last_constraint(
        positioned_constraints: list[PositionedConstraint],
    ) -> Constraint:
        return positioned_constraints.pop()[0]

    @staticmethod
    def pop_first_constraint(
        positioned_constraints: list[PositionedConstraint],
    ) -> Constraint:
        return positioned_constraints.pop(0)[0]

    @staticmethod
    def pop_the_most_important_constraint(
        positioned_constraints: list[PositionedConstraint],
        compare: Callable[[PositionedConstraint, PositionedConstraint], float],
    ) -> Constraint:
        positioned_constraints.sort(key=functools.cmp_to_key(compare))
        return ShapValuesComparator.pop_last_constraint(positioned_constraints)


def pop_last_constraint(
    positioned_constraints: list[PositionedConstraint],
) -> Constraint:
    return ShapValuesComparator.pop_last_constraint(positioned_constraints)


def pop_first_constraint(
    positioned_constraints: list[PositionedConstraint],
) -> Constraint:
    return ShapValuesComparator.pop_first_constraint(positioned_constraints)


def pop_the_most_important_constraint(
    positioned_constraints: list[PositionedConstraint],
    compare: Callable[[PositionedConstraint, PositionedConstraint], float],
) -> Constraint:
    return ShapValuesComparator.pop_the_most_important_constraint(
        positioned_constraints, compare
    )
