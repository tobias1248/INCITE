#!/usr/bin/env python3
"""Unit tests for transformer-specific helpers in dnnct.myDNN."""

from __future__ import annotations

import unittest

import numpy as np
from libct.utils import ConcolicObject
from unittest import mock

import dnnct.myDNN as mydnn


class _TensorLike:
    def __init__(self, value):
        self._value = np.array(value, dtype=np.float32)

    def numpy(self):
        return self._value


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

    def test_add_cls_token_layer_rank2(self) -> None:
        cls_token = np.array([[[10.0, 20.0]]], dtype=np.float32)
        layer = mydnn.AddClsTokenLayer(cls_token)
        output = layer.forward([[1.0, 2.0], [3.0, 4.0]])

        self.assertTrue(np.allclose(output, [[10.0, 20.0], [1.0, 2.0], [3.0, 4.0]], atol=1e-6))

    def test_add_cls_token_layer_rank3(self) -> None:
        cls_token = np.array([[[9.0, 8.0]]], dtype=np.float32)
        layer = mydnn.AddClsTokenLayer(cls_token)
        output = layer.forward(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )

        self.assertEqual(np.array(output).shape, (2, 3, 2))
        self.assertTrue(np.allclose(output[0][0], [9.0, 8.0], atol=1e-6))
        self.assertTrue(np.allclose(output[1][0], [9.0, 8.0], atol=1e-6))

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

    def test_extract_cls_token_layer_rank2(self) -> None:
        layer = mydnn.ExtractClsTokenLayer()
        output = layer.forward([[10.0, 20.0], [1.0, 2.0], [3.0, 4.0]])

        self.assertTrue(np.allclose(output, [10.0, 20.0], atol=1e-6))

    def test_extract_cls_token_layer_rank3(self) -> None:
        layer = mydnn.ExtractClsTokenLayer()
        output = layer.forward(
            [
                [[10.0, 20.0], [1.0, 2.0]],
                [[30.0, 40.0], [3.0, 4.0]],
            ]
        )

        self.assertEqual(np.array(output).shape, (2, 2))
        self.assertTrue(np.allclose(output[0], [10.0, 20.0], atol=1e-6))
        self.assertTrue(np.allclose(output[1], [30.0, 40.0], atol=1e-6))

    def test_cls_token_flow_with_position_embedding(self) -> None:
        cls_layer = mydnn.AddClsTokenLayer(np.array([[[1.0, 1.5]]], dtype=np.float32))
        pos_layer = mydnn.AddPositionEmbeddingLayer(
            np.array([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]], dtype=np.float32)
        )
        extract_layer = mydnn.ExtractClsTokenLayer()

        with_cls = cls_layer.forward([[2.0, 3.0], [4.0, 5.0]])
        with_pos = pos_layer.forward(with_cls)
        output = extract_layer.forward(with_pos)

        self.assertTrue(np.allclose(output, [1.1, 1.7], atol=1e-6))

    def test_act_softmax_returns_concrete_float_values(self) -> None:
        values = [
            ConcolicObject(0.1, ["x"]),
            ConcolicObject(0.4, ["y"]),
            ConcolicObject(-0.2, ["z"]),
        ]
        output = mydnn.act_softmax(values)

        self.assertTrue(all(type(value) is float for value in output))
        self.assertAlmostEqual(sum(output), 1.0, places=6)

    def test_mha_softmax_returns_concrete_float_values(self) -> None:
        layer = mydnn.MultiHeadAttentionLayer(
            num_heads=1,
            key_dim_per_heads=2,
            wq=_TensorLike(np.zeros((2, 1, 2))),
            bq=_TensorLike(np.zeros((1, 2))),
            wk=_TensorLike(np.zeros((2, 1, 2))),
            bk=_TensorLike(np.zeros((1, 2))),
            wv=_TensorLike(np.zeros((2, 1, 2))),
            bv=_TensorLike(np.zeros((1, 2))),
            output_weights=_TensorLike(np.zeros((1, 2, 2))),
            output_bias=_TensorLike(np.zeros((2,))),
        )
        meta = {
            "feature_dim": 2,
            "token_shape": (2,),
            "mode": "sequence",
        }
        scores = [
            [ConcolicObject(0.3, ["a"]), ConcolicObject(0.1, ["b"])],
            [ConcolicObject(-0.2, ["c"]), ConcolicObject(0.2, ["d"])],
        ]
        output = layer.softmax(scores, meta)

        self.assertEqual(len(output), 2)
        self.assertTrue(all(type(value) is float for row in output for value in row))
        self.assertAlmostEqual(sum(output[0]), 1.0, places=6)
        self.assertAlmostEqual(sum(output[1]), 1.0, places=6)

    def test_attention_position_registration_is_query_only(self) -> None:
        layer = mydnn.MultiHeadAttentionLayer(
            num_heads=1,
            key_dim_per_heads=2,
            wq=_TensorLike(np.zeros((2, 1, 2))),
            bq=_TensorLike(np.zeros((1, 2))),
            wk=_TensorLike(np.zeros((2, 1, 2))),
            bk=_TensorLike(np.zeros((1, 2))),
            wv=_TensorLike(np.zeros((2, 1, 2))),
            bv=_TensorLike(np.zeros((1, 2))),
            output_weights=_TensorLike(np.zeros((1, 2, 2))),
            output_bias=_TensorLike(np.zeros((2,))),
        )
        meta = {
            "feature_dim": 4,
            "token_shape": (8,),
            "mode": "sequence",
        }
        with mock.patch("dnnct.myDNN.register_current_indices") as mocked_register:
            layer._register_attention_position(3, 99, meta)
        mocked_register.assert_called_once()
        indices = mocked_register.call_args.args[0]
        self.assertEqual(indices, [(3, 0), (3, 1), (3, 2), (3, 3)])

    def test_attention_position_registration_is_spatial_query_only(self) -> None:
        layer = mydnn.MultiHeadAttentionLayer(
            num_heads=1,
            key_dim_per_heads=1,
            wq=_TensorLike(np.zeros((1, 1, 1))),
            bq=_TensorLike(np.zeros((1, 1))),
            wk=_TensorLike(np.zeros((1, 1, 1))),
            bk=_TensorLike(np.zeros((1, 1))),
            wv=_TensorLike(np.zeros((1, 1, 1))),
            bv=_TensorLike(np.zeros((1, 1))),
            output_weights=_TensorLike(np.zeros((1, 1, 1))),
            output_bias=_TensorLike(np.zeros((1,))),
            attention_axes=(1, 2),
            sample_rank=3,
        )
        meta = {
            "feature_dim": 1,
            "token_shape": (2, 2),
            "mode": "spatial_2d",
        }
        with mock.patch("dnnct.myDNN.register_current_indices") as mocked_register:
            layer._register_attention_position(3, 0, meta)
        mocked_register.assert_called_once_with([(1, 1, 0)])

    def test_mha_forward_supports_spatial_attention_shape(self) -> None:
        layer = mydnn.MultiHeadAttentionLayer(
            num_heads=1,
            key_dim_per_heads=1,
            wq=_TensorLike(np.ones((1, 1, 1))),
            bq=_TensorLike(np.zeros((1, 1))),
            wk=_TensorLike(np.ones((1, 1, 1))),
            bk=_TensorLike(np.zeros((1, 1))),
            wv=_TensorLike(np.ones((1, 1, 1))),
            bv=_TensorLike(np.zeros((1, 1))),
            output_weights=_TensorLike(np.ones((1, 1, 1))),
            output_bias=_TensorLike(np.zeros((1,))),
            attention_axes=(1, 2),
            sample_rank=3,
        )
        output = layer.forward(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
            ]
        )

        self.assertEqual(np.array(output).shape, (2, 2, 1))


if __name__ == "__main__":
    unittest.main()
