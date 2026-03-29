"""Dataset adapters for the modularized task builders."""

from datasets.cifar10 import Cifar10Dataset
from datasets.common import (
    enable_attack_pixels,
    get_background_per_class,
    get_background_seed,
    select_background_per_class,
    tensor_to_in_dict_and_con_dict,
)
from datasets.fashion_mnist import FashionMnistDataset
from datasets.legacy_sequence import IMDB_Dataset, MSstock_Dataset, RNN_MnistDataset
from datasets.mnist import MnistDataset

__all__ = [
    "Cifar10Dataset",
    "FashionMnistDataset",
    "IMDB_Dataset",
    "MSstock_Dataset",
    "MnistDataset",
    "RNN_MnistDataset",
    "enable_attack_pixels",
    "get_background_per_class",
    "get_background_seed",
    "select_background_per_class",
    "tensor_to_in_dict_and_con_dict",
]
