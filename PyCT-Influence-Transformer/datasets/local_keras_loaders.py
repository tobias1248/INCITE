from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np


def _read_idx_images_gz(path: Path) -> np.ndarray:
    with gzip.open(path, 'rb') as handle:
        data = np.frombuffer(handle.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def _read_idx_labels_gz(path: Path) -> np.ndarray:
    with gzip.open(path, 'rb') as handle:
        return np.frombuffer(handle.read(), np.uint8, offset=8)


def load_local_fashion_mnist(datasets_dir: Path):
    base = datasets_dir / 'fashion-mnist'
    x_train = _read_idx_images_gz(base / 'train-images-idx3-ubyte.gz')
    y_train = _read_idx_labels_gz(base / 'train-labels-idx1-ubyte.gz')
    x_test = _read_idx_images_gz(base / 't10k-images-idx3-ubyte.gz')
    y_test = _read_idx_labels_gz(base / 't10k-labels-idx1-ubyte.gz')
    return (x_train, y_train), (x_test, y_test)


def _load_cifar_batch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open('rb') as handle:
        payload = pickle.load(handle, encoding='bytes')
    data = payload[b'data'].reshape(-1, 3, 32, 32)
    labels = np.asarray(payload[b'labels'], dtype=np.uint8)
    return data, labels


def load_local_cifar10(datasets_dir: Path):
    base = datasets_dir / 'cifar-10-batches-py'
    train_batches = []
    train_labels = []
    for idx in range(1, 6):
        batch_data, batch_labels = _load_cifar_batch(base / f'data_batch_{idx}')
        train_batches.append(batch_data)
        train_labels.append(batch_labels)
    x_train = np.concatenate(train_batches, axis=0)
    y_train = np.concatenate(train_labels, axis=0).reshape(-1, 1)
    x_test, y_test = _load_cifar_batch(base / 'test_batch')
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)
    y_test = y_test.reshape(-1, 1)
    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)
    return (x_train, y_train), (x_test, y_test)


__all__ = ['load_local_fashion_mnist', 'load_local_cifar10']
