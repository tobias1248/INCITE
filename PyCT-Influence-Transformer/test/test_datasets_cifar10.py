from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets.cifar10 as cifar_mod
from datasets.keras_cache import DatasetAvailabilityError


def test_cifar10_dataset_init_uses_local_loader_and_normalizes(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_loader(datasets_dir):
        calls.append(datasets_dir)
        x_train = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        y_train = np.array([[0]], dtype=np.uint8)
        x_test = np.full((2, 2, 2, 3), 255, dtype=np.uint8)
        y_test = np.array([[1], [2]], dtype=np.uint8)
        return (x_train, y_train), (x_test, y_test)

    monkeypatch.setattr(cifar_mod, "prepare_keras_cache_env", lambda: None)
    monkeypatch.setattr(cifar_mod, "ensure_local_dataset_files", lambda *_args: tmp_path)
    monkeypatch.setattr(cifar_mod, "load_local_cifar10", fake_loader)

    dataset = cifar_mod.Cifar10Dataset()

    assert calls == [tmp_path]
    assert dataset.x_test.shape == (2, 2, 2, 3)
    assert float(dataset.x_test[0, 0, 0, 0]) == 1.0


def test_cifar10_dataset_init_propagates_missing_cache(monkeypatch) -> None:
    monkeypatch.setattr(cifar_mod, "prepare_keras_cache_env", lambda: None)
    monkeypatch.setattr(
        cifar_mod,
        "ensure_local_dataset_files",
        lambda *_args: (_ for _ in ()).throw(DatasetAvailabilityError("missing cifar cache")),
    )

    with pytest.raises(DatasetAvailabilityError, match="missing cifar cache"):
        cifar_mod.Cifar10Dataset()


def test_get_cifar10_test_data_and_set_condict_enables_attack_pixels(monkeypatch) -> None:
    monkeypatch.setenv("PYCT_BG_PER_CLASS", "1")
    monkeypatch.setenv("PYCT_BG_SEED", "11")
    dataset = cifar_mod.Cifar10Dataset.__new__(cifar_mod.Cifar10Dataset)
    dataset.x_test = np.array(
        [
            [[[0.0, 0.1, 0.2]]],
            [[[0.3, 0.4, 0.5]]],
            [[[0.6, 0.7, 0.8]]],
            [[[0.9, 1.0, 1.1]]],
        ],
        dtype=np.float32,
    )
    dataset.y_test = np.array([[0], [0], [1], [1]], dtype=np.int64)

    in_dict, con_dict, input_for_shap, background = dataset.get_cifar10_test_data_and_set_condict(
        2,
        attack_pixels=[(0, 0, 2)],
    )

    assert in_dict["v_0_0_1"] == pytest.approx(0.7)
    assert con_dict["v_0_0_0"] == 0
    assert con_dict["v_0_0_2"] == 1
    assert input_for_shap.shape == (1, 1, 3)
    assert background.shape[0] == 2


def test_cifar10_get_test_data_respects_background_env(monkeypatch) -> None:
    monkeypatch.setenv("PYCT_BG_PER_CLASS", "1")
    monkeypatch.setenv("PYCT_BG_SEED", "9")
    dataset = cifar_mod.Cifar10Dataset.__new__(cifar_mod.Cifar10Dataset)
    dataset.x_test = np.array(
        [
            [[[0.0, 0.0, 0.0]]],
            [[[1.0, 1.0, 1.0]]],
            [[[2.0, 2.0, 2.0]]],
            [[[3.0, 3.0, 3.0]]],
        ],
        dtype=np.float32,
    )
    dataset.y_test = np.array([[0], [0], [1], [1]], dtype=np.int64)

    first = dataset.get_cifar10_test_data_and_set_condict(0, attack_pixels=[])[3]
    second = dataset.get_cifar10_test_data_and_set_condict(0, attack_pixels=[])[3]

    assert np.array_equal(first, second)
    assert first.shape[0] == 2
