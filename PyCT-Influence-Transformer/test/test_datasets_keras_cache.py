from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.keras_cache import (
    DatasetAvailabilityError,
    ensure_local_dataset_files,
    get_keras_datasets_dir,
    resolve_mnist_path,
)


def test_get_keras_datasets_dir_respects_pyct_keras_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv('PYCT_KERAS_HOME', str(tmp_path / 'keras-home'))
    monkeypatch.delenv('KERAS_HOME', raising=False)

    datasets_dir = get_keras_datasets_dir()

    assert datasets_dir == tmp_path / 'keras-home' / 'datasets'


def test_resolve_mnist_path_requires_local_cache_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv('PYCT_KERAS_HOME', str(tmp_path / 'keras-home'))
    monkeypatch.delenv('PYCT_ALLOW_DATASET_DOWNLOAD', raising=False)

    with pytest.raises(DatasetAvailabilityError, match='MNIST dataset cache is missing'):
        resolve_mnist_path()


def test_resolve_mnist_path_allows_download_when_enabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv('PYCT_KERAS_HOME', str(tmp_path / 'keras-home'))
    monkeypatch.setenv('PYCT_ALLOW_DATASET_DOWNLOAD', '1')

    assert resolve_mnist_path() == 'mnist.npz'


def test_ensure_local_dataset_files_raises_with_missing_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv('PYCT_KERAS_HOME', str(tmp_path / 'keras-home'))
    monkeypatch.delenv('PYCT_ALLOW_DATASET_DOWNLOAD', raising=False)

    with pytest.raises(DatasetAvailabilityError, match='fashion_mnist dataset cache is missing'):
        ensure_local_dataset_files('fashion_mnist', ['fashion-mnist/train-images-idx3-ubyte.gz'])


def test_ensure_local_dataset_files_accepts_normal_fashion_mnist_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv('PYCT_KERAS_HOME', str(tmp_path / 'keras-home'))
    monkeypatch.delenv('PYCT_ALLOW_DATASET_DOWNLOAD', raising=False)
    cached_file = tmp_path / 'keras-home' / 'datasets' / 'fashion-mnist' / 'train-images-idx3-ubyte.gz'
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    cached_file.write_bytes(b'cached')

    resolved = ensure_local_dataset_files('fashion_mnist', ['fashion-mnist/train-images-idx3-ubyte.gz'])

    assert resolved == tmp_path / 'keras-home' / 'datasets'
