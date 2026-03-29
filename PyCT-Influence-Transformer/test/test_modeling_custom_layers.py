from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.custom_layers import (
    AddClsToken,
    AddPositionEmbedding,
    DropPath,
    ExtractClsToken,
    SequencePooling,
    get_transformer_custom_objects,
)


class _FakeKerasOps:
    @staticmethod
    def shape(inputs):
        return np.array(inputs).shape

    @staticmethod
    def broadcast_to(value, shape):
        return np.broadcast_to(np.array(value), shape)

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate([np.array(v) for v in values], axis=axis)

    @staticmethod
    def softmax(values, axis=0):
        values = np.array(values, dtype=np.float32)
        shifted = values - np.max(values, axis=axis, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=axis, keepdims=True)

    @staticmethod
    def sum(values, axis=0):
        return np.sum(np.array(values), axis=axis)


class _FakeDense:
    def __init__(self, units, name=None):
        self.units = units
        self.name = name

    def __call__(self, inputs):
        values = np.array(inputs, dtype=np.float32)
        return np.sum(values, axis=-1, keepdims=True)


def test_transformer_custom_objects_contains_expected_aliases(monkeypatch):
    layer_map = get_transformer_custom_objects()

    assert layer_map["AddClsToken"] is AddClsToken
    assert layer_map["Custom>AddClsToken"] is AddClsToken
    assert layer_map["AddPositionEmbedding"] is AddPositionEmbedding
    assert layer_map["ExtractClsToken"] is ExtractClsToken
    assert layer_map["DropPath"] is DropPath
    assert layer_map["SequencePooling"] is SequencePooling


def test_add_cls_token_uses_keras_ops_path(monkeypatch):
    monkeypatch.setattr('modeling.custom_layers.keras.ops', _FakeKerasOps, raising=False)

    layer = AddClsToken()
    layer.cls_token = np.array([[[9.0, 8.0]]], dtype=np.float32)
    output = layer.call(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))

    assert output.shape == (1, 3, 2)
    assert np.allclose(output[0, 0], [9.0, 8.0])
    assert np.allclose(output[0, 1], [1.0, 2.0])


def test_sequence_pooling_uses_keras_ops_path(monkeypatch):
    monkeypatch.setattr('modeling.custom_layers.keras.ops', _FakeKerasOps, raising=False)
    monkeypatch.setattr('modeling.custom_layers.keras.layers.Dense', _FakeDense)

    layer = SequencePooling()
    output = layer.call(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))

    assert output.shape == (1, 2)
    assert output[0, 0] > 2.0
    assert output[0, 1] > 3.0


def test_drop_path_config_round_trip():
    layer = DropPath(drop_prob=0.25)

    config = layer.get_config()

    assert config["drop_prob"] == 0.25
