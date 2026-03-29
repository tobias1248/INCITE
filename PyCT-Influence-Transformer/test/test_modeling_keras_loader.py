from __future__ import annotations

from pathlib import Path
import sys
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import modeling.keras_loader as loader


class _FakeDataset:
    def __init__(self, shape):
        self.shape = shape


class _FakeWeightsGroup:
    def __init__(self, datasets):
        self._datasets = datasets

    def visititems(self, visitor):
        for name, dataset in self._datasets.items():
            visitor(name, dataset)


class _FakeHandle:
    def __init__(self, *, attrs=None, model_weights=None):
        self.attrs = attrs or {}
        self._model_weights = model_weights

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, key):
        if key == 'model_weights':
            return self._model_weights
        return None


def test_read_model_config_decodes_json(monkeypatch):
    handle = _FakeHandle(attrs={'model_config': b'{"class_name": "ResNet2D18"}'})
    monkeypatch.setattr(loader, 'h5py', mock.Mock(File=lambda *_args, **_kwargs: handle))

    config = loader.read_model_config('model/mock.h5')

    assert config == {'class_name': 'ResNet2D18'}


def test_extract_specs_from_weights_reads_channels_and_classes(monkeypatch):
    fake_h5 = mock.Mock()
    fake_h5.Dataset = _FakeDataset
    fake_h5.File = lambda *_args, **_kwargs: _FakeHandle(
        model_weights=_FakeWeightsGroup(
            {
                'conv/kernel:0': _FakeDataset((3, 3, 5, 16)),
                'dense/kernel:0': _FakeDataset((16, 10)),
                'dense_hidden/kernel:0': _FakeDataset((16, 32)),
            }
        )
    )
    monkeypatch.setattr(loader, 'h5py', fake_h5)

    specs = loader.extract_specs_from_weights('model/mock.h5')

    assert specs['input_channel_only'] == 5
    assert specs['num_classes'] == 10


def test_infer_resnet_specs_uses_overrides(monkeypatch):
    monkeypatch.setattr(loader, 'read_model_config', lambda _path: {'class_name': 'ResNet2D18'})
    monkeypatch.setattr(loader, 'extract_specs_from_weights', lambda _path: {})

    specs = loader.infer_resnet_specs(
        'model/resnet18_custom.h5',
        input_shape_override=(64, 64, 3),
        num_classes_override=7,
    )

    assert specs == {'depth': 18, 'input_shape': (64, 64, 3), 'num_classes': 7}


def test_load_model_with_compat_delegates_to_keras_when_direct_load_succeeds(monkeypatch):
    loaded = object()
    monkeypatch.setattr(loader, 'get_resnet_custom_objects', lambda: {'A': object()})
    monkeypatch.setattr(loader.keras.models, 'load_model', lambda *args, **kwargs: loaded)

    result = loader.load_model_with_compat('model/mock.h5')

    assert result is loaded


def test_load_model_with_compat_uses_resnet_fallback(monkeypatch):
    class _Boom(ValueError):
        pass

    def raise_load(*_args, **_kwargs):
        raise _Boom("Unknown layer: 'ResNet2D18'")

    rebuilt = mock.Mock()
    monkeypatch.setattr(loader, 'get_resnet_custom_objects', lambda: {})
    monkeypatch.setattr(loader.keras.models, 'load_model', raise_load)
    monkeypatch.setattr(loader, 'infer_resnet_specs', lambda *_args, **_kwargs: {'depth': 18, 'input_shape': (32, 32, 3), 'num_classes': 10})
    monkeypatch.setattr(loader, 'build_resnet_model', lambda *_args, **_kwargs: rebuilt)

    result = loader.load_model_with_compat('model/mock.h5')

    rebuilt.load_weights.assert_called_once_with('model/mock.h5')
    assert result is rebuilt


def test_load_model_with_compat_raises_when_fallback_specs_missing(monkeypatch):
    class _Boom(ValueError):
        pass

    def raise_load(*_args, **_kwargs):
        raise _Boom("Unknown layer: 'ResNet2D18'")

    monkeypatch.setattr(loader, 'get_resnet_custom_objects', lambda: {})
    monkeypatch.setattr(loader.keras.models, 'load_model', raise_load)
    monkeypatch.setattr(loader, 'infer_resnet_specs', lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match='Failed to rebuild ResNet model'):
        loader.load_model_with_compat('model/mock.h5')
