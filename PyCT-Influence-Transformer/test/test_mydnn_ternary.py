from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dnnct.myDNN as mydnn


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
