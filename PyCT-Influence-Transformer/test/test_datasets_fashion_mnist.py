from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets.fashion_mnist as fashion_mod
from datasets.keras_cache import DatasetAvailabilityError


def test_fashion_mnist_dataset_init_uses_local_loader_and_normalizes(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_loader(datasets_dir):
        calls.append(datasets_dir)
        x_train = np.array([[[0, 255]]], dtype=np.uint8)
        y_train = np.array([0], dtype=np.uint8)
        x_test = np.array([[[255, 0]], [[0, 255]]], dtype=np.uint8)
        y_test = np.array([1, 2], dtype=np.uint8)
        return (x_train, y_train), (x_test, y_test)

    monkeypatch.setattr(fashion_mod, "prepare_keras_cache_env", lambda: None)
    monkeypatch.setattr(fashion_mod, "ensure_local_dataset_files", lambda *_args: tmp_path)
    monkeypatch.setattr(fashion_mod, "load_local_fashion_mnist", fake_loader)

    dataset = fashion_mod.FashionMnistDataset()

    assert calls == [tmp_path]
    assert dataset.x_test.shape == (2, 1, 2, 1)
    assert float(dataset.x_test[0, 0, 0, 0]) == 1.0


def test_fashion_mnist_dataset_init_propagates_missing_cache(monkeypatch) -> None:
    monkeypatch.setattr(fashion_mod, "prepare_keras_cache_env", lambda: None)
    monkeypatch.setattr(
        fashion_mod,
        "ensure_local_dataset_files",
        lambda *_args: (_ for _ in ()).throw(DatasetAvailabilityError("missing fashion cache")),
    )

    with pytest.raises(DatasetAvailabilityError, match="missing fashion cache"):
        fashion_mod.FashionMnistDataset()


def test_get_fashion_mnist_test_data_and_set_condict_enables_attack_pixels(monkeypatch) -> None:
    monkeypatch.setenv("PYCT_BG_PER_CLASS", "1")
    monkeypatch.setenv("PYCT_BG_SEED", "7")
    dataset = fashion_mod.FashionMnistDataset.__new__(fashion_mod.FashionMnistDataset)
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

    in_dict, con_dict, input_for_shap, background = dataset.get_fashion_mnist_test_data_and_set_condict(
        1,
        attack_pixels=[(0, 0, 0)],
    )

    assert in_dict["v_0_1_0"] == pytest.approx(0.3)
    assert con_dict["v_0_0_0"] == 1
    assert con_dict["v_0_1_0"] == 0
    assert input_for_shap.shape == (1, 2, 1)
    assert background.shape[0] == 2


def test_fashion_mnist_dataset_cache_layout_uses_normal_directory_shape() -> None:
    assert all(path.startswith("fashion-mnist/") for path in fashion_mod._FASHION_MNIST_FILES)
