from __future__ import annotations

from pathlib import Path
import gzip
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.local_keras_loaders import load_local_cifar10, load_local_fashion_mnist


def _write_idx_images(path: Path, images: np.ndarray) -> None:
    header = np.array([2051, images.shape[0], images.shape[1], images.shape[2]], dtype='>i4').tobytes()
    with gzip.open(path, 'wb') as handle:
        handle.write(header)
        handle.write(images.astype(np.uint8).tobytes())


def _write_idx_labels(path: Path, labels: np.ndarray) -> None:
    header = np.array([2049, labels.shape[0]], dtype='>i4').tobytes()
    with gzip.open(path, 'wb') as handle:
        handle.write(header)
        handle.write(labels.astype(np.uint8).tobytes())


def test_load_local_fashion_mnist_reads_standard_cache_layout(tmp_path) -> None:
    base = tmp_path / 'fashion-mnist'
    base.mkdir(parents=True)
    train_images = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
    test_images = np.arange(28 * 28, dtype=np.uint8).reshape(1, 28, 28)
    train_labels = np.array([3, 4], dtype=np.uint8)
    test_labels = np.array([9], dtype=np.uint8)
    _write_idx_images(base / 'train-images-idx3-ubyte.gz', train_images)
    _write_idx_images(base / 't10k-images-idx3-ubyte.gz', test_images)
    _write_idx_labels(base / 'train-labels-idx1-ubyte.gz', train_labels)
    _write_idx_labels(base / 't10k-labels-idx1-ubyte.gz', test_labels)

    (x_train, y_train), (x_test, y_test) = load_local_fashion_mnist(tmp_path)

    assert x_train.shape == (2, 28, 28)
    assert x_test.shape == (1, 28, 28)
    np.testing.assert_array_equal(y_train, train_labels)
    np.testing.assert_array_equal(y_test, test_labels)


def _write_cifar_batch(path: Path, labels, seed: int) -> None:
    rng = np.random.default_rng(seed)
    payload = {
        b'data': rng.integers(0, 255, size=(1, 3072), dtype=np.uint8),
        b'labels': list(labels),
    }
    with path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def test_load_local_cifar10_reads_extracted_batch_directory(tmp_path) -> None:
    base = tmp_path / 'cifar-10-batches-py'
    base.mkdir(parents=True)
    for idx in range(1, 6):
        _write_cifar_batch(base / f'data_batch_{idx}', [idx], seed=idx)
    _write_cifar_batch(base / 'test_batch', [7], seed=99)
    (base / 'batches.meta').write_bytes(b'meta')

    (x_train, y_train), (x_test, y_test) = load_local_cifar10(tmp_path)

    assert x_train.shape == (5, 32, 32, 3)
    assert y_train.shape == (5, 1)
    assert x_test.shape == (1, 32, 32, 3)
    assert y_test.shape == (1, 1)
    assert int(y_test[0, 0]) == 7
