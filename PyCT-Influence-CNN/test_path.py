#!/usr/bin/env python3
"""
Utility to inspect the layer ordering of a (ResNet) Keras model and compare it
with the topological execution order used by dnn_predict_common.
"""

import argparse
import sys

import keras

from dnn_predict_common import _collect_execution_order, _get_inbound_layers


def _format_shape(shape):
    if shape is None:
        return "None"
    return str(shape)


def _describe_layers(layers, title):
    print(f"\n=== {title} ===")
    for idx, layer in enumerate(layers):
        inbound = [parent.name for parent in _get_inbound_layers(layer)]
        print(
            f"[{idx:03d}] {layer.name:<35} {layer.__class__.__name__:<30} "
            f"output={_format_shape(getattr(layer, 'output_shape', None))} "
            f"inbound={inbound}"
        )


def _build_resnet18(input_shape, classes):
    try:
        from keras_resnet.models import ResNet18
    except ModuleNotFoundError:
        sys.exit("Missing dependency keras-resnet. Install with `pip install keras-resnet`.")

    inputs = keras.layers.Input(shape=input_shape)
    return ResNet18(inputs=inputs, classes=classes, include_top=True, freeze_bn=False)


def _load_model(model_path):
    try:
        from keras_resnet.models import ResNet2D18
        from keras_resnet.layers import BatchNormalization
    except ModuleNotFoundError:
        ResNet2D18 = None
        BatchNormalization = None

    custom_objects = {}
    if ResNet2D18 is not None:
        custom_objects["ResNet2D18"] = ResNet2D18
    if BatchNormalization is not None:
        custom_objects["BatchNormalization"] = BatchNormalization

    try:
        return keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to load {model_path}: {exc}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect model layer ordering.")
    parser.add_argument(
        "--model-path",
        help="Path to a saved Keras .h5 model. If omitted, builds a fresh ResNet18.",
    )
    parser.add_argument(
        "--input-shape",
        default="28,28,1",
        help="Input shape for fallback ResNet build (comma-separated).",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=10,
        help="Number of classes for fallback ResNet build.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = None
    if args.model_path:
        model = _load_model(args.model_path)

    if model is None:
        shape = tuple(int(dim) for dim in args.input_shape.split(","))
        model = _build_resnet18(shape, args.classes)
        print("Loaded fallback ResNet18 model (random weights).")
    else:
        print(f"Loaded model from {args.model_path}.")

    _describe_layers(model.layers, "Raw model.layers order")

    execution_order = _collect_execution_order(model)
    _describe_layers(execution_order, "Topological execution order")


if __name__ == "__main__":
    main()
