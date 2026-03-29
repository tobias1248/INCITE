from __future__ import annotations

import numpy as np

from datasets.common import (
    enable_attack_pixels,
    get_background_per_class,
    get_background_seed,
    select_background_per_class,
    tensor_to_in_dict_and_con_dict,
)
from datasets.keras_cache import prepare_keras_cache_env, resolve_mnist_path


class MnistDataset:
    def __init__(self) -> None:
        prepare_keras_cache_env()
        from tensorflow.keras.datasets.mnist import load_data

        (x_train, y_train), (x_test, y_test) = load_data(path=resolve_mnist_path())
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        del x_train
        del y_train

        self.x_test = np.expand_dims(x_test, -1)
        self.y_test = y_test

    def get_mnist_test_data(self, idx):
        test_img = self.x_test[idx]
        in_dict, con_dict = tensor_to_in_dict_and_con_dict(test_img)
        input_for_shap = test_img
        background_dataset_for_shap = select_background_per_class(
            self.x_test,
            self.y_test,
            per_class=get_background_per_class(),
            seed=get_background_seed(),
        )
        return in_dict, con_dict, input_for_shap, background_dataset_for_shap

    def get_mnist_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict, con_dict, input_for_shap, background_dataset_for_shap = self.get_mnist_test_data(idx)
        enable_attack_pixels(con_dict, attack_pixels)
        return in_dict, con_dict, input_for_shap, background_dataset_for_shap


__all__ = ["MnistDataset"]
