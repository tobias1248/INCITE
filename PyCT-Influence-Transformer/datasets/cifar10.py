from __future__ import annotations

from datasets.common import (
    enable_attack_pixels,
    get_background_per_class,
    get_background_seed,
    select_background_per_class,
    tensor_to_in_dict_and_con_dict,
)
from datasets.keras_cache import ensure_local_dataset_files, prepare_keras_cache_env
from datasets.local_keras_loaders import load_local_cifar10

_CIFAR10_FILES = tuple(
    [f'cifar-10-batches-py/data_batch_{idx}' for idx in range(1, 6)]
    + ['cifar-10-batches-py/test_batch', 'cifar-10-batches-py/batches.meta']
)


class Cifar10Dataset:
    def __init__(self) -> None:
        prepare_keras_cache_env()
        datasets_dir = ensure_local_dataset_files('cifar10', _CIFAR10_FILES)
        (x_train, y_train), (x_test, y_test) = load_local_cifar10(datasets_dir)
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        del x_train
        del y_train

        self.x_test = x_test
        self.y_test = y_test

    def get_cifar10_test_data(self, idx):
        return tensor_to_in_dict_and_con_dict(self.x_test[idx])

    def get_cifar10_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict, con_dict = self.get_cifar10_test_data(idx)
        input_for_shap = self.x_test[idx]
        background_dataset_for_shap = select_background_per_class(
            self.x_test,
            self.y_test,
            per_class=get_background_per_class(),
            seed=get_background_seed(),
        )
        enable_attack_pixels(con_dict, attack_pixels)
        return in_dict, con_dict, input_for_shap, background_dataset_for_shap


__all__ = ['Cifar10Dataset']
