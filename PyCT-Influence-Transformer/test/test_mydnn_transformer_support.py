#!/usr/bin/env python3
"""Unit tests for transformer-specific helpers in dnnct.myDNN."""

from __future__ import annotations

import unittest

import numpy as np

import dnnct.myDNN as mydnn


class MyDNNTransformerSupportTests(unittest.TestCase):
    def test_activation_layer_supports_gelu(self) -> None:
        layer = mydnn.ActivationLayer("gelu")
        values = [-4.0, -1.0, 0.0, 1.0, 4.0]
        output = layer.forward(values.copy())

        self.assertEqual(output[0], 0.0)  # <= -3 saturated
        self.assertAlmostEqual(output[1], -1.0 / 3.0, places=6)
        self.assertEqual(output[2], 0.0)
        self.assertAlmostEqual(output[3], 2.0 / 3.0, places=6)
        self.assertEqual(output[4], 4.0)  # >= 3 behaves like identity

    def test_layer_norm_layer_shape_and_centering(self) -> None:
        layer = mydnn.LayerNormLayer(gamma=[1.0, 1.0, 1.0], beta=[0.0, 0.0, 0.0], epsilon=1e-6)
        output = layer.forward([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])

        self.assertEqual(len(output), 2)
        self.assertEqual(len(output[0]), 3)
        self.assertAlmostEqual(sum(output[0]) / 3.0, 0.0, places=6)
        for value in output[1]:
            self.assertAlmostEqual(value, 0.0, places=6)

    def test_add_position_embedding_layer(self) -> None:
        pos_embedding = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
        layer = mydnn.AddPositionEmbeddingLayer(pos_embedding)
        output = layer.forward([[1.0, 2.0], [3.0, 4.0]])

        self.assertTrue(np.allclose(output, [[1.1, 2.2], [3.3, 4.4]], atol=1e-6))

    def test_sequence_pooling_layer(self) -> None:
        kernel = np.array([[1.0], [0.0]], dtype=np.float32)
        bias = np.array([0.0], dtype=np.float32)
        layer = mydnn.SequencePoolingLayer(kernel, bias)
        output = layer.forward([[1.0, 2.0], [3.0, 4.0]])

        self.assertEqual(len(output), 2)
        self.assertGreater(output[0], 2.0)
        self.assertLess(output[0], 3.0)
        self.assertGreater(output[1], 3.0)
        self.assertLess(output[1], 4.0)


if __name__ == "__main__":
    unittest.main()
