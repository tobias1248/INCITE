#!/usr/bin/env python3
"""Unit tests for logging-related refactors in dnnct.myDNN."""

from __future__ import annotations

import logging
import sys
import types
import unittest
from typing import Any, List

import numpy as np
from unittest import mock

if "keras" not in sys.modules:
    keras_module = types.ModuleType("keras")
    layers_module = types.ModuleType("keras.layers")
    for name in [
        "Dense",
        "Conv1D",
        "Conv2D",
        "LocallyConnected1D",
        "LocallyConnected2D",
        "Flatten",
        "ELU",
        "Activation",
        "ReLU",
        "MaxPool2D",
        "MaxPooling2D",
        "RandomCrop",
        "RandomFlip",
        "Dropout",
        "ZeroPadding2D",
        "LSTM",
        "Embedding",
        "BatchNormalization",
        "LayerNormalization",
        "SimpleRNN",
        "MultiHeadAttention",
        "Add",
        "AveragePooling1D",
        "GlobalAveragePooling2D",
        "GlobalAveragePooling1D",
        "Reshape",
    ]:
        setattr(layers_module, name, type(name, (), {}))
    keras_module.layers = layers_module
    keras_module.backend = types.SimpleNamespace(clear_session=lambda: None)
    sys.modules["keras"] = keras_module
    sys.modules["keras.layers"] = layers_module

import dnnct.myDNN as mydnn


class _AddLayerDenseStub:
    def __init__(self):
        self._kernel = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self._bias = np.array([0.0, 0.0], dtype=np.float32)

    def get_weights(self):
        return [self._kernel, self._bias]

    def get_config(self):
        return {"activation": "linear"}


class _DummyLayer:
    def __init__(self):
        self._output = None

    def forward(self, tensor_in):
        self._output = [value + 1 for value in tensor_in]
        return self._output

    def getOutput(self):
        return self._output


class MyDNNLoggingTests(unittest.TestCase):
    def test_act_sigmoid_boundaries(self) -> None:
        self.assertEqual(mydnn.act_sigmoid(0), 0.5)
        self.assertEqual(mydnn.act_sigmoid(6), 1.0)
        self.assertEqual(mydnn.act_sigmoid(-6), 0.0)

    def test_actfunc_sigmoid_matches_act_sigmoid(self) -> None:
        value = 0.25
        self.assertAlmostEqual(
            mydnn.actFunc(value, "sigmoid"),
            mydnn.act_sigmoid(value),
        )

    def test_activation_layer_forward_tanh(self) -> None:
        layer = mydnn.ActivationLayer("tanh")
        output = layer.forward([[0, 1], [-1, 2]])
        expected = [
            [mydnn.act_tanh(0), mydnn.act_tanh(1)],
            [mydnn.act_tanh(-1), mydnn.act_tanh(2)],
        ]
        for row_out, row_exp in zip(output, expected):
            self.assertTrue(
                np.allclose(row_out, row_exp),
                msg=f"{row_out} != {row_exp}",
            )

    def test_nnmodel_forward_logs_layers(self) -> None:
        model = mydnn.NNModel()
        model.layers = [_DummyLayer(), _DummyLayer()]
        tensor = [0, 1, 2]
        with mock.patch("dnnct.myDNN.register_current_layer_number", lambda *_: None):
            with mock.patch(
                "dnnct.myDNN.to_Keras_layer_number",
                side_effect=lambda idx: idx,
            ):
                with self.assertLogs("ct.model", level=logging.INFO) as captured:
                    result = model.forward(tensor)
        self.assertEqual(result, [2, 3, 4])
        self.assertTrue(any("DNN start forwarding" in msg for msg in captured.output))
        self.assertTrue(any("DNN finish forwarding" in msg for msg in captured.output))

    def test_nnmodel_add_layer_appends_activation(self) -> None:
        original_dense = mydnn.Dense
        mydnn.Dense = _AddLayerDenseStub  # type: ignore[assignment]
        try:
            model = mydnn.NNModel()
            stub_layer = _AddLayerDenseStub()
            with self.assertLogs("ct.model", level=logging.DEBUG) as captured:
                added = model.addLayer(stub_layer)
            self.assertEqual(added, 2)
            self.assertEqual(len(model.layers), 2)
            self.assertTrue(
                any("Add Activation Layer" in msg for msg in captured.output),
                msg="Expected activation-layer log entry",
            )
        finally:
            mydnn.Dense = original_dense  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
