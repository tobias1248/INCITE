from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dnnct.myDNN as mydnn


class _TensorLike:
    def __init__(self, value):
        self._value = np.array(value, dtype=np.float32)

    def numpy(self):
        return self._value


def _mha_layer(
    *,
    key_dim_per_heads: int = 1,
    ternary_config: mydnn.TernaryRuntimeConfig,
) -> mydnn.MultiHeadAttentionLayer:
    return mydnn.MultiHeadAttentionLayer(
        num_heads=1,
        key_dim_per_heads=key_dim_per_heads,
        wq=_TensorLike(np.array([[[2.0] * key_dim_per_heads], [[0.1] * key_dim_per_heads], [[-4.0] * key_dim_per_heads]])),
        bq=_TensorLike(np.zeros((1, key_dim_per_heads))),
        wk=_TensorLike(np.ones((3, 1, key_dim_per_heads))),
        bk=_TensorLike(np.zeros((1, key_dim_per_heads))),
        wv=_TensorLike(np.ones((3, 1, key_dim_per_heads))),
        bv=_TensorLike(np.zeros((1, key_dim_per_heads))),
        output_weights=_TensorLike(np.array([[[2.0], [0.1], [-4.0]]])),
        output_bias=_TensorLike(np.array([0.5])),
        ternary_config=ternary_config,
    )


def test_compute_delta_returns_scaled_mean_abs_weights() -> None:
    weights = np.array([[1.0, -3.0], [2.0, -2.0]], dtype=float)

    delta = mydnn.compute_delta(weights, 0.75)

    assert delta == np.mean(np.abs(weights)) * 0.75


def test_dense_layer_ternary_forward_rank1() -> None:
    layer = mydnn.DenseLayer(
        np.array([[2.0, 0.1, -4.0]], dtype=float),
        np.array([0.5], dtype=float),
        (1, 3),
        ternary_config=mydnn.TernaryRuntimeConfig(enabled=True, threshold_scale=0.75),
    )

    output = layer.forward([1.0, 2.0, 3.0])

    assert output == [-1.5]


def test_dense_layer_ternary_forward_rank2() -> None:
    layer = mydnn.DenseLayer(
        np.array([[2.0, 0.1, -4.0]], dtype=float),
        np.array([0.5], dtype=float),
        (1, 3),
        ternary_config=mydnn.TernaryRuntimeConfig(enabled=True, threshold_scale=0.75),
    )

    output = layer.forward([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert output == [[-1.5], [-1.5]]


def test_dense_layer_disabled_ternary_preserves_full_precision() -> None:
    weights = np.array([[2.0, 0.1, -4.0]], dtype=float)
    bias = np.array([0.5], dtype=float)
    layer = mydnn.DenseLayer(
        weights,
        bias,
        (1, 3),
        ternary_config=mydnn.TernaryRuntimeConfig(enabled=False, threshold_scale=0.75),
    )

    output = layer.forward([1.0, 2.0, 3.0])

    expected = [0.5 + 1.0 * 2.0 + 2.0 * 0.1 + 3.0 * -4.0]
    assert output == expected


def test_conv2d_layer_ternary_forward() -> None:
    weights = np.array([[[[2.0], [0.1]], [[-4.0], [0.0]]]], dtype=float)
    bias = np.array([1.0], dtype=float)
    layer = mydnn.Conv2DLayer(
        weights,
        bias,
        weights.shape,
        ternary_config=mydnn.TernaryRuntimeConfig(enabled=True, threshold_scale=0.75),
    )

    output = layer.forward(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ]
    )

    assert output == [[[ -1.0 ]]]


def test_conv2d_layer_disabled_ternary_preserves_full_precision() -> None:
    weights = np.array([[[[2.0], [0.1]], [[-4.0], [0.0]]]], dtype=float)
    bias = np.array([1.0], dtype=float)
    layer = mydnn.Conv2DLayer(
        weights,
        bias,
        weights.shape,
        ternary_config=mydnn.TernaryRuntimeConfig(enabled=False, threshold_scale=0.75),
    )

    output = layer.forward(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ]
    )

    expected = 1.0 + 1.0 * 2.0 + 2.0 * 0.1 + 3.0 * -4.0 + 4.0 * 0.0
    assert output == [[[expected]]]


def test_mha_transform_projection_uses_ternary_weights() -> None:
    layer = _mha_layer(
        ternary_config=mydnn.TernaryRuntimeConfig(enabled=True, threshold_scale=0.75),
    )

    output = layer.transform_and_split(
        [[1.0, 2.0, 3.0]],
        layer.WQ,
        layer.BQ,
        feature_dim=3,
        delta=layer._delta_wq,
    )

    assert output == [[[-2.0]]]


def test_mha_transform_projection_disabled_ternary_preserves_full_precision() -> None:
    layer = _mha_layer(
        ternary_config=mydnn.TernaryRuntimeConfig(enabled=False, threshold_scale=0.75),
    )

    output = layer.transform_and_split(
        [[1.0, 2.0, 3.0]],
        layer.WQ,
        layer.BQ,
        feature_dim=3,
        delta=layer._delta_wq,
    )

    expected = 1.0 * 2.0 + 2.0 * 0.1 + 3.0 * -4.0
    assert output == [[[expected]]]


def test_mha_output_projection_uses_ternary_weights() -> None:
    layer = _mha_layer(
        key_dim_per_heads=3,
        ternary_config=mydnn.TernaryRuntimeConfig(enabled=True, threshold_scale=0.75),
    )

    output = layer.output_transform(
        [[1.0, 2.0, 3.0]],
        layer.WO,
        layer.BO,
        delta=layer._delta_wo,
    )

    assert output == [[-1.5]]
