#!/usr/bin/env python3
"""
Sanity test that NNModel assigns unique SSA keys per layer instance and
that residual Add layers correctly reference those keys.
"""

import os
import tempfile

import keras

import dnn_predict_common as dpc  # type: ignore
from dnnct.myDNN import AddLayer  # type: ignore


def build_toy_resnet():
    inputs = keras.layers.Input(shape=(8, 8, 1), name="input_tensor")

    x = keras.layers.Conv2D(4, 3, padding="same",
                            name="block1_conv")(inputs)
    x = keras.layers.Activation("relu", name="block1_relu")(x)

    y = keras.layers.Conv2D(4, 3, padding="same",
                            name="block2_conv")(x)
    y = keras.layers.Activation("relu", name="block2_relu")(y)

    shortcut = keras.layers.Conv2D(
        4, 1, padding="same", name="shortcut_conv")(inputs)
    shortcut = keras.layers.Activation(
        "relu", name="shortcut_relu")(shortcut)

    merged = keras.layers.Add(name="residual_add")([y, shortcut])
    merged = keras.layers.Flatten(name="flatten")(merged)
    outputs = keras.layers.Dense(2, activation="softmax", name="pred")(merged)

    return keras.Model(inputs, outputs, name="toy_resnet")


def run_test():
    model = build_toy_resnet()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "toy_resnet.h5")
        model.save(model_path)
        dpc.init_model(model_path)

    my_model = getattr(dpc, "myModel", None)
    if my_model is None:
        raise RuntimeError("myModel was not initialized")

    keys = getattr(my_model, "my_layer_keys", [])
    unique_keys = set(keys)
    print("=== SSA keys in myModel ===")
    for idx, key in enumerate(keys):
        print(f"{idx:03d}: {key}")
    if len(keys) != len(unique_keys):
        raise AssertionError("SSA keys are not unique.")

    add_layers = [layer for layer in my_model.layers if isinstance(layer, AddLayer)]
    if not add_layers:
        raise AssertionError("No AddLayer instances were registered.")

    key_set = set(keys)
    for add in add_layers:
        for source in add.input_from:
            if source not in key_set and source != "layer_input":
                raise AssertionError(f"AddLayer references missing source {source}")

    print("SSA test passed: unique keys and valid residual inputs.")


if __name__ == "__main__":
    run_test()
