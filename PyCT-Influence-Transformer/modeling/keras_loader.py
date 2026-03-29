from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import keras

from modeling.custom_layers import get_transformer_custom_objects

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

log = logging.getLogger("ct.model")
_KERAS_RESNET_IMPORT_ERROR: Exception | None = None


def _inject_keras_resnet_libct() -> None:
    try:
        import libct as libct_module
        import keras_resnet.layers as kr_layers

        setattr(kr_layers, "libct", libct_module)
        if hasattr(kr_layers, "BatchNormalization"):
            setattr(kr_layers.BatchNormalization, "libct", libct_module)
    except Exception:
        pass


def get_resnet_custom_objects() -> Dict[str, object]:
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


def needs_resnet_fallback(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "Unknown layer: 'ResNet2D" in msg
        or "missing 1 required positional argument: 'inputs'" in msg
        or "Unable to revive model from config" in msg and "ResNet2D" in msg
    )


def read_model_config(model_path: str) -> dict | None:
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


def extract_specs_from_weights(model_path: str) -> Dict[str, int]:
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


def infer_resnet_specs(
    model_path: str,
    *,
    input_shape_override: Optional[Tuple[int, ...]] = None,
    num_classes_override: Optional[int] = None,
) -> Optional[dict]:
    specs: dict = {}
    config = read_model_config(model_path)
    if config:
        class_name = str(config.get("class_name", ""))
        matches = re.findall(r"(\d+)", class_name)
        if matches:
            specs["depth"] = int(matches[-1])

    specs.update(extract_specs_from_weights(model_path))

    if "depth" not in specs:
        match = re.search(r"resnet(\d+)", Path(model_path).stem.lower())
        if match:
            specs["depth"] = int(match.group(1))

    if input_shape_override is not None:
        specs["input_shape"] = tuple(int(v) for v in input_shape_override)
    elif "input_shape" not in specs:
        guessed_shape = _guess_default_input_shape(
            model_path,
            channel_hint=specs.get("input_channel_only")
            if isinstance(specs.get("input_channel_only"), int)
            else None,
        )
        if guessed_shape is not None:
            specs["input_shape"] = guessed_shape

    if num_classes_override is not None:
        specs["num_classes"] = int(num_classes_override)
    elif "num_classes" not in specs:
        guessed_classes = _guess_default_num_classes(model_path)
        if guessed_classes is not None:
            specs["num_classes"] = guessed_classes

    if "depth" not in specs or "input_shape" not in specs or "num_classes" not in specs:
        return None
    return specs


def build_resnet_model(depth: int, input_shape: Tuple[int, ...], num_classes: int):
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


def load_model_with_compat(
    model_path: str,
    *,
    input_shape_override: Optional[Tuple[int, ...]] = None,
    num_classes_override: Optional[int] = None,
):
    custom_objects = get_resnet_custom_objects()
    custom_objects.update(get_transformer_custom_objects())
    try:
        if custom_objects:
            return keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return keras.models.load_model(model_path, compile=False)
    except (ValueError, TypeError) as exc:
        if not needs_resnet_fallback(exc):
            raise

        specs = infer_resnet_specs(
            model_path,
            input_shape_override=input_shape_override,
            num_classes_override=num_classes_override,
        )
        if specs is None:
            import_hint = "Please ensure `keras-resnet` is installed and importable in this environment."
            if _KERAS_RESNET_IMPORT_ERROR is not None:
                import_hint += f" Import error: {_KERAS_RESNET_IMPORT_ERROR!r}"
            raise ValueError(
                f"Failed to rebuild ResNet model for '{model_path}'. {import_hint}"
            ) from exc

        rebuilt = build_resnet_model(
            int(specs["depth"]),
            tuple(int(v) for v in specs["input_shape"]),
            int(specs["num_classes"]),
        )
        rebuilt.load_weights(model_path)
        return rebuilt


__all__ = [
    "build_resnet_model",
    "extract_specs_from_weights",
    "get_resnet_custom_objects",
    "infer_resnet_specs",
    "load_model_with_compat",
    "needs_resnet_fallback",
    "read_model_config",
]
