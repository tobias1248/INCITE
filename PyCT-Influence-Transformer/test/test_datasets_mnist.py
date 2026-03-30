from __future__ import annotations

from pathlib import Path
from types import ModuleType
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets.mnist as mnist_mod
from datasets.keras_cache import DatasetAvailabilityError


def test_mnist_dataset_init_uses_local_loader_and_normalizes(monkeypatch) -> None:
    calls = []
    fake_module = ModuleType("tensorflow.keras.datasets.mnist")

    def fake_load_data(*, path):
        calls.append(path)
        x_train = np.array([[[0, 255]]], dtype=np.uint8)
        y_train = np.array([0], dtype=np.uint8)
        x_test = np.array([[[0, 255]], [[255, 0]]], dtype=np.uint8)
        y_test = np.array([1, 2], dtype=np.uint8)
        return (x_train, y_train), (x_test, y_test)

    fake_module.load_data = fake_load_data
    monkeypatch.setattr(mnist_mod, "prepare_keras_cache_env", lambda: None)
    monkeypatch.setattr(mnist_mod, "resolve_mnist_path", lambda: "/tmp/mnist.npz")
    monkeypatch.setitem(sys.modules, "tensorflow.keras.datasets.mnist", fake_module)

    dataset = mnist_mod.MnistDataset()

    assert calls == ["/tmp/mnist.npz"]
    assert dataset.x_test.shape == (2, 1, 2, 1)
    assert float(dataset.x_test[0, 0, 1, 0]) == 1.0


def test_mnist_dataset_init_propagates_missing_cache(monkeypatch) -> None:
    monkeypatch.setattr(mnist_mod, "prepare_keras_cache_env", lambda: None)
    monkeypatch.setattr(
        mnist_mod,
        "resolve_mnist_path",
        lambda: (_ for _ in ()).throw(DatasetAvailabilityError("missing mnist cache")),
    )

    with pytest.raises(DatasetAvailabilityError, match="missing mnist cache"):
        mnist_mod.MnistDataset()


def test_get_mnist_test_data_and_set_condict_enables_attack_pixels(monkeypatch) -> None:
    monkeypatch.setenv("PYCT_BG_PER_CLASS", "1")
    monkeypatch.setenv("PYCT_BG_SEED", "123")
    dataset = mnist_mod.MnistDataset.__new__(mnist_mod.MnistDataset)
    dataset.x_test = np.array(
        [
            [[[0.0], [0.1]]],
            [[[0.2], [0.3]]],
            [[[0.4], [0.5]]],
            [[[0.6], [0.7]]],
        ],
        dtype=np.float32,
    )
    dataset.y_test = np.array([0, 0, 1, 1], dtype=np.int64)

    in_dict, con_dict, input_for_shap, background = dataset.get_mnist_test_data_and_set_condict(
        0,
        attack_pixels=[(0, 1, 0)],
    )

    assert in_dict["v_0_1_0"] == pytest.approx(0.1)
    assert con_dict["v_0_0_0"] == 0
    assert con_dict["v_0_1_0"] == 1
    assert input_for_shap.shape == (1, 2, 1)
    assert background.shape[0] == 2


def test_get_mnist_test_data_respects_background_env(monkeypatch) -> None:
    monkeypatch.setenv("PYCT_BG_PER_CLASS", "1")
    monkeypatch.setenv("PYCT_BG_SEED", "42")
    dataset = mnist_mod.MnistDataset.__new__(mnist_mod.MnistDataset)
    dataset.x_test = np.array(
        [
            [[[0.0]]],
            [[[1.0]]],
            [[[2.0]]],
            [[[3.0]]],
        ],
        dtype=np.float32,
    )
    dataset.y_test = np.array([0, 0, 1, 1], dtype=np.int64)

    first = dataset.get_mnist_test_data(0)[3]
    second = dataset.get_mnist_test_data(0)[3]

    assert np.array_equal(first, second)
    assert first.shape[0] == 2
