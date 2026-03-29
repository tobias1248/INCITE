from __future__ import annotations

import numpy as np

from datasets.common import (
    enable_attack_pixels,
    get_background_per_class,
    get_background_seed,
    select_background_per_class,
    tensor_to_in_dict_and_con_dict,
)
from datasets.keras_cache import ensure_local_dataset_files, prepare_keras_cache_env
from datasets.local_keras_loaders import load_local_fashion_mnist

_FASHION_MNIST_FILES = (
    'fashion-mnist/train-labels-idx1-ubyte.gz',
    'fashion-mnist/train-images-idx3-ubyte.gz',
    'fashion-mnist/t10k-labels-idx1-ubyte.gz',
    'fashion-mnist/t10k-images-idx3-ubyte.gz',
)


class FashionMnistDataset:
    def __init__(self) -> None:
        prepare_keras_cache_env()
        datasets_dir = ensure_local_dataset_files('fashion_mnist', _FASHION_MNIST_FILES)
        (x_train, y_train), (x_test, y_test) = load_local_fashion_mnist(datasets_dir)
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        del x_train
        del y_train

        self.x_test = np.expand_dims(x_test, -1)
        self.y_test = y_test

    def get_fashion_mnist_test_data(self, idx):
        test_img = self.x_test[idx]
        return tensor_to_in_dict_and_con_dict(test_img)

    def get_fashion_mnist_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict, con_dict = self.get_fashion_mnist_test_data(idx)
        input_for_shap = self.x_test[idx]
        background_dataset_for_shap = select_background_per_class(
            self.x_test,
            self.y_test,
            per_class=get_background_per_class(),
            seed=get_background_seed(),
        )
        enable_attack_pixels(con_dict, attack_pixels)
        return in_dict, con_dict, input_for_shap, background_dataset_for_shap


__all__ = ['FashionMnistDataset']
