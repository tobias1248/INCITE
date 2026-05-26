#!/usr/bin/env python3
"""Tests for myDNN Conv2D inference semantics."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from keras import layers

import dnnct.myDNN as mydnn


def _assert_conv2d_matches_keras(
    input_values: np.ndarray,
    kernel_values: np.ndarray,
    *,
    strides: tuple[int, int],
    padding: str,
) -> None:
    keras_layer = layers.Conv2D(
        filters=int(kernel_values.shape[-1]),
        kernel_size=tuple(int(dim) for dim in kernel_values.shape[:2]),
        strides=strides,
        padding=padding,
        use_bias=False,
    )
    keras_layer.build((None, *input_values.shape))
    keras_layer.set_weights([kernel_values.astype(np.float32)])

    my_weights = kernel_values.transpose(3, 0, 1, 2)
    my_layer = mydnn.Conv2DLayer(
        my_weights,
        np.zeros(my_weights.shape[0], dtype=np.float32),
        my_weights.shape,
        stride=strides,
        padding=padding,
    )

    expected = keras_layer(input_values[np.newaxis, ...]).numpy()[0]
    actual = np.asarray(my_layer.forward(input_values.tolist()), dtype=np.float32)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(my_layer.getOutput(), dtype=np.float32), expected)


def test_conv2d_same_padding_stride_two_matches_keras() -> None:
    input_values = np.arange(1, 17, dtype=np.float32).reshape(4, 4, 1)
    kernel_values = np.ones((3, 3, 1, 1), dtype=np.float32)

    _assert_conv2d_matches_keras(
        input_values,
        kernel_values,
        strides=(2, 2),
        padding="same",
    )


def test_conv2d_same_padding_stride_one_stays_keras_compatible() -> None:
    input_values = np.arange(1, 26, dtype=np.float32).reshape(5, 5, 1)
    kernel_values = np.arange(1, 10, dtype=np.float32).reshape(3, 3, 1, 1)

    _assert_conv2d_matches_keras(
        input_values,
        kernel_values,
        strides=(1, 1),
        padding="same",
    )


def test_conv2d_valid_padding_stride_two_stays_keras_compatible() -> None:
    input_values = np.arange(1, 26, dtype=np.float32).reshape(5, 5, 1)
    kernel_values = np.arange(1, 10, dtype=np.float32).reshape(3, 3, 1, 1)

    _assert_conv2d_matches_keras(
        input_values,
        kernel_values,
        strides=(2, 2),
        padding="valid",
    )


def test_conv2d_one_by_one_same_padding_stride_two_matches_keras() -> None:
    input_values = np.arange(1, 17, dtype=np.float32).reshape(4, 4, 1)
    kernel_values = np.asarray([[[[2.0]]]], dtype=np.float32)

    _assert_conv2d_matches_keras(
        input_values,
        kernel_values,
        strides=(2, 2),
        padding="same",
    )
