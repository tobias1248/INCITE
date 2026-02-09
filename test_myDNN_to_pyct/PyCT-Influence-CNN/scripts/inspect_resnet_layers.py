#!/usr/bin/env python3
"""
Utility to inspect layer configurations of a Keras model (e.g., resnet18_mnist.h5).
Focuses on Conv2D, BatchNormalization, ZeroPadding2D, pooling/GAP layers so that
we can compare them with the parameters used in build_myResnet.py.
"""
import os
import sys
import argparse
from typing import Iterable, Optional

import keras

# 取得目前檔案的絕對路徑，並將其上一層目錄加入 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dnn_predict_common import load_keras_model

try:
    # Keras 3
    from keras.layers import (
        Conv2D,
        BatchNormalization,
        ZeroPadding2D,
        MaxPool2D,
        GlobalAveragePooling2D,
        AveragePooling2D,
    )
except ImportError:  # pragma: no cover
    from tensorflow.keras.layers import (  # type: ignore
        Conv2D,
        BatchNormalization,
        ZeroPadding2D,
        MaxPool2D,
        GlobalAveragePooling2D,
        AveragePooling2D,
    )


def _print_dict(d: dict, indent: int = 0):
    pad = " " * indent
    for key, value in d.items():
        print(f"{pad}{key}: {value}")


def describe_conv(layer: Conv2D):
    cfg = layer.get_config()
    info = {
        "kernel_size": cfg.get("kernel_size"),
        "strides": cfg.get("strides"),
        "padding": cfg.get("padding"),
        "filters": cfg.get("filters"),
        "use_bias": cfg.get("use_bias"),
    }
    _print_dict(info, indent=2)
    for weight in layer.weights:
        print(f"    weight: {weight.name} shape={weight.shape}")


def describe_bn(layer: BatchNormalization):
    cfg = layer.get_config()
    info = {
        "axis": cfg.get("axis"),
        "momentum": cfg.get("momentum"),
        "epsilon": cfg.get("epsilon"),
        "center": cfg.get("center"),
        "scale": cfg.get("scale"),
    }
    _print_dict(info, indent=2)
    for weight in layer.weights:
        print(f"    weight: {weight.name} shape={weight.shape}")


def describe_padding(layer: ZeroPadding2D):
    cfg = layer.get_config()
    info = {"padding": cfg.get("padding")}
    _print_dict(info, indent=2)


def describe_pool(layer: keras.layers.Layer):
    cfg = layer.get_config()
    info = {k: cfg.get(k)
            for k in ("pool_size", "strides", "padding") if k in cfg}
    if not info:
        info = cfg
    _print_dict(info, indent=2)


DESCRIBE_DISPATCH = {
    Conv2D: describe_conv,
    BatchNormalization: describe_bn,
    ZeroPadding2D: describe_padding,
    MaxPool2D: describe_pool,
    AveragePooling2D: describe_pool,
    GlobalAveragePooling2D: describe_pool,
}


def describe_generic(layer: keras.layers.Layer):
    cfg = layer.get_config()
    print("  config keys:", sorted(cfg.keys()))
    if layer.weights:
        for weight in layer.weights:
            print(f"    weight: {weight.name} shape={weight.shape}")


def iter_matching_layers(model: keras.Model, names: Optional[Iterable[str]]):
    if names:
        targets = set(names)
    else:
        targets = None
    for idx, layer in enumerate(model.layers):
        if targets is None or layer.name in targets:
            yield idx, layer


def inspect_model(model_path: str, layer_names: Optional[Iterable[str]]):
    model = load_keras_model(model_path)
    print(f"Loaded model: {model.name}")
    print("=" * 80)

    for idx, layer in iter_matching_layers(model, layer_names):
        describe_fn = None
        for klass, handler in DESCRIBE_DISPATCH.items():
            if isinstance(layer, klass):
                describe_fn = handler
                break

        if describe_fn is None:
            describe_fn = describe_generic

        print(f"[{idx:03d}] ({layer.__class__.__name__}) {layer.name}")
        describe_fn(layer)
        print("-" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect Conv/BN/ZeroPadding/Pooling layers of a Keras model."
    )
    parser.add_argument(
        "--model-path",
        default="model/resnet18_mnist.h5",
        help="Path to the Keras .h5 model to inspect.",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        help="Specific layer names to inspect. If omitted, all matching layer types are listed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect_model(args.model_path, args.layers)
