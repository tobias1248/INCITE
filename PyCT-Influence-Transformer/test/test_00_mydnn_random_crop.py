#!/usr/bin/env python3
"""Tests for myDNN RandomCrop inference semantics."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from keras import layers

import dnnct.myDNN as mydnn


def _assert_random_crop_matches_smart_resize(
    input_shape: tuple[int, int, int],
    target_shape: tuple[int, int],
) -> None:
    values = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
    layer = mydnn.RandomCropLayer(target_shape[0], target_shape[1])

    actual = np.asarray(layer.forward(values), dtype=np.float32)
    expected = np.asarray(
        tf.keras.preprocessing.image.smart_resize(values, target_shape),
        dtype=np.float32,
    )

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(layer.getOutput(), dtype=np.float32), expected)


def test_random_crop_layer_matches_smart_resize_for_padded_cifar_shape() -> None:
    _assert_random_crop_matches_smart_resize((40, 40, 1), (32, 32))


def test_random_crop_layer_matches_smart_resize_for_non_square_shape() -> None:
    _assert_random_crop_matches_smart_resize((5, 8, 2), (4, 4))


def test_random_crop_layer_matches_smart_resize_when_upsampling() -> None:
    _assert_random_crop_matches_smart_resize((3, 4, 1), (6, 8))


def test_random_crop_layer_matches_keras_random_crop_inference_when_same_size() -> None:
    values = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    actual = np.asarray(mydnn.RandomCropLayer(3, 4).forward(values), dtype=np.float32)
    expected = layers.RandomCrop(3, 4)(values, training=False).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_random_crop_mapping_uses_random_crop_layer() -> None:
    model = mydnn.NNModel()

    added = model.addLayer(layers.RandomCrop(2, 3))

    assert added == 1
    assert isinstance(model.layers[0], mydnn.RandomCropLayer)


def test_random_flip_mapping_stays_noop_layer() -> None:
    model = mydnn.NNModel()

    added = model.addLayer(layers.RandomFlip("horizontal"))

    assert added == 1
    assert isinstance(model.layers[0], mydnn.NoOpLayer)
