#!/usr/bin/env python3
"""Unit tests for transformer-specific helpers in dnnct.myDNN."""

from __future__ import annotations

import unittest

import numpy as np
from libct.predicate import Predicate
from libct.utils import ConcolicObject

import dnnct.myDNN as mydnn


class _TensorLike:
    def __init__(self, value):
        self._value = np.array(value, dtype=np.float32)

    def numpy(self):
        return self._value


def _unit_mha_layer(*, sample_rank=None, attention_axes=None):
    return mydnn.MultiHeadAttentionLayer(
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
        attention_axes=attention_axes,
        sample_rank=sample_rank,
    )


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

    def test_average_pooling_1d_layer(self) -> None:
        layer = mydnn.AveragePooling1DLayer(pool_size=4, stride=4)
        output = layer.forward(
            [
                [1.0, 10.0],
                [3.0, 14.0],
                [5.0, 18.0],
                [7.0, 22.0],
                [9.0, 26.0],
                [11.0, 30.0],
                [13.0, 34.0],
                [15.0, 38.0],
            ]
        )

        self.assertTrue(np.allclose(output, [[4.0, 16.0], [12.0, 32.0]], atol=1e-6))

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
        scores = [
            [ConcolicObject(0.3, ["a"]), ConcolicObject(0.1, ["b"])],
            [ConcolicObject(-0.2, ["c"]), ConcolicObject(0.2, ["d"])],
        ]
        output = layer.softmax(scores)

        self.assertEqual(len(output), 2)
        self.assertTrue(all(type(value) is float for row in output for value in row))
        self.assertAlmostEqual(sum(output[0]), 1.0, places=6)
        self.assertAlmostEqual(sum(output[1]), 1.0, places=6)

    def test_mha_freezes_qk_but_keeps_value_projection_symbolic(self) -> None:
        class _NoBranchPath:
            def add_branch(self, _branch) -> None:
                raise AssertionError("frozen Q/K attention must not create branches")

        class _Engine:
            symbolic_enabled = True
            path = _NoBranchPath()

        engine = _Engine()
        layer = _unit_mha_layer(sample_rank=2)
        query = [
            [ConcolicObject(1.0, "query_0_VAR", engine)],
            [ConcolicObject(2.0, "query_1_VAR", engine)],
        ]
        key = [
            [ConcolicObject(1.5, "key_0_VAR", engine)],
            [ConcolicObject(0.5, "key_1_VAR", engine)],
        ]
        value = [
            [ConcolicObject(3.0, "value_0_VAR", engine)],
            [ConcolicObject(4.0, "value_1_VAR", engine)],
        ]

        output = layer._forward_attention(query, key, value)
        expression = Predicate.get_formula_deep(output[0][0].expr)

        self.assertEqual(layer.symbolic_attention_mode, "frozen_qk")
        self.assertNotIn("query_", expression)
        self.assertNotIn("key_", expression)
        self.assertIn("value_", expression)

    def test_mha_recomputes_frozen_attention_weights_for_each_input(self) -> None:
        layer = _unit_mha_layer(sample_rank=2)

        first = layer.forward([[1.0], [2.0]])
        second = layer.forward([[2.0], [1.0]])

        self.assertFalse(np.allclose(first, second, atol=1e-6))

    def test_mha_applies_attention_mask(self) -> None:
        layer = _unit_mha_layer(sample_rank=2)

        output = layer.forward(
            [[1.0], [2.0]],
            mask=[[True, False], [False, True]],
        )

        self.assertTrue(np.allclose(output, [[1.0], [2.0]], atol=1e-6))

    def test_mha_rejects_fully_masked_attention_row(self) -> None:
        layer = _unit_mha_layer(sample_rank=2)

        with self.assertRaisesRegex(ValueError, "cannot hide every key"):
            layer.forward(
                [[1.0], [2.0]],
                mask=[[False, False], [True, True]],
            )

    def test_mha_applies_per_sample_batch_masks(self) -> None:
        layer = _unit_mha_layer(sample_rank=2)

        output = layer.forward(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
            ],
            mask=[
                [[True, False], [False, True]],
                [[False, True], [True, False]],
            ],
        )

        self.assertTrue(
            np.allclose(
                output,
                [
                    [[1.0], [2.0]],
                    [[4.0], [3.0]],
                ],
                atol=1e-6,
            )
        )

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

    def test_mha_forward_caches_sequence_output(self) -> None:
        layer = _unit_mha_layer(sample_rank=2)

        output = layer.forward([[1.0], [2.0]])

        self.assertTrue(np.allclose(layer.getOutput(), output, atol=1e-6))

    def test_mha_forward_caches_spatial_output(self) -> None:
        layer = _unit_mha_layer(sample_rank=3, attention_axes=(1, 2))

        output = layer.forward(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
            ]
        )

        self.assertEqual(np.array(layer.getOutput()).shape, (2, 2, 1))
        self.assertTrue(np.allclose(layer.getOutput(), output, atol=1e-6))

    def test_mha_forward_caches_full_batch_output(self) -> None:
        layer = _unit_mha_layer(sample_rank=2)

        output = layer.forward(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
            ]
        )

        self.assertEqual(np.array(layer.getOutput()).shape, (2, 2, 1))
        self.assertTrue(np.allclose(layer.getOutput(), output, atol=1e-6))

    def test_nnmodel_get_layer_output_reads_mha_output(self) -> None:
        layer = _unit_mha_layer(sample_rank=2)
        output = layer.forward([[1.0], [2.0]])
        model = mydnn.NNModel()
        model.layers = [layer]

        self.assertTrue(np.allclose(model.getLayOutput(0), output, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
