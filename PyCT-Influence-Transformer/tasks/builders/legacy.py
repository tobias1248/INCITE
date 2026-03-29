from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from datasets.legacy_sequence import IMDB_Dataset, MSstock_Dataset, RNN_MnistDataset
from datasets.mnist import MnistDataset
from tasks.builders.common import (
    generate_inputs,
    make_coordinate_provider,
    make_payload_builder,
    make_shap_provider,
    queue_modes,
)
from tasks.types import TaskGenerationSpec


def _dataset_sample_shape(dataset: Any) -> Tuple[int, ...]:
    sample = np.asarray(dataset.x_test[0])
    if sample.ndim == 3 and sample.shape[-1] == 1:
        sample = np.squeeze(sample, axis=-1)
    return tuple(int(dim) for dim in sample.shape)


def _deterministic_random_provider(dataset_factory, ton_values: Sequence[int], *, base_seed: int = 2024):
    dataset = dataset_factory()
    return make_coordinate_provider(_dataset_sample_shape(dataset), ton_values, base_seed=base_seed)


def pyct_shap_1_4_8_16_32(model_name: str, first_n_img: int) -> List[Dict[str, Any]]:
    shap_pixels = np.load(f"./shap_value/{model_name}/mnist_sort_shap_pixel.npy")
    spec = TaskGenerationSpec(
        dataset_factory=MnistDataset,
        attack_pixel_fn=make_shap_provider(shap_pixels),
        queue_modes=queue_modes(include_stack=True),
        ton_values=[1],
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"mnist_test_{idx}",
            "exp_name": f"shap_{ton}",
        },
        payload_builder=make_payload_builder("get_mnist_test_data_and_set_condict"),
    )
    return generate_inputs(model_name, first_n_img, spec).inputs


def pyct_random_1_4_8_16_32(model_name: str, first_n_img: int) -> List[Dict[str, Any]]:
    ton_values = [1, 4, 8, 16, 32]
    spec = TaskGenerationSpec(
        dataset_factory=MnistDataset,
        attack_pixel_fn=_deterministic_random_provider(MnistDataset, ton_values),
        queue_modes=queue_modes(include_stack=True),
        ton_values=ton_values,
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"mnist_test_{idx}",
            "exp_name": f"random_{ton}",
        },
        payload_builder=make_payload_builder("get_mnist_test_data_and_set_condict"),
    )
    return generate_inputs(model_name, first_n_img, spec).inputs


def pyct_rnn_random_1_4_8_16_32(model_name: str, first_n_img: int) -> List[Dict[str, Any]]:
    ton_values = [1, 4, 8, 16, 32]
    spec = TaskGenerationSpec(
        dataset_factory=RNN_MnistDataset,
        attack_pixel_fn=_deterministic_random_provider(RNN_MnistDataset, ton_values),
        queue_modes=queue_modes(include_stack=True),
        ton_values=ton_values,
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"mnist_test_{idx}",
            "exp_name": f"random_{ton}",
        },
        payload_builder=make_payload_builder("get_mnist_test_data_and_set_condict"),
    )
    return generate_inputs(model_name, first_n_img, spec).inputs


def pyct_rnn_shap_1_4_8_16_32(model_name: str, first_n_img: int) -> List[Dict[str, Any]]:
    shap_pixels = np.load(f"./shap_value/{model_name}/mnist_sort_shap_pixel.npy")
    spec = TaskGenerationSpec(
        dataset_factory=RNN_MnistDataset,
        attack_pixel_fn=make_shap_provider(shap_pixels),
        queue_modes=queue_modes(include_stack=True),
        ton_values=[1, 4, 8, 16, 32],
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"mnist_test_{idx}",
            "exp_name": f"shap_{ton}",
        },
        payload_builder=make_payload_builder("get_mnist_test_data_and_set_condict"),
    )
    return generate_inputs(model_name, first_n_img, spec).inputs


def stock_shap_1_2_3_4_8_limit_range02(model_name: str, first_n_img: int) -> List[Dict[str, Any]]:
    limit_p = 0.2
    shap_pixels = np.load(f"./shap_value/{model_name}/stock_sort_shap_pixel.npy")
    spec = TaskGenerationSpec(
        dataset_factory=MSstock_Dataset,
        attack_pixel_fn=make_shap_provider(shap_pixels),
        queue_modes=queue_modes(include_stack=True),
        ton_values=[1, 2, 3, 4, 8],
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"stock_test_{idx}",
            "exp_name": f"limit_{limit_p}/shap_{ton}",
            "save_smt": True,
        },
        payload_builder=make_payload_builder(
            "get_stock_test_data_and_set_condict",
            extra_fields={"limit_change_percentage": limit_p},
        ),
    )
    return generate_inputs(model_name, first_n_img, spec).inputs


def stock_random_1_2_3_4_8_range02(model_name: str, first_n_img: int) -> List[Dict[str, Any]]:
    limit_p = 0.2
    ton_values = [1, 2, 3, 4, 8]
    spec = TaskGenerationSpec(
        dataset_factory=MSstock_Dataset,
        attack_pixel_fn=_deterministic_random_provider(MSstock_Dataset, ton_values),
        queue_modes=queue_modes(include_stack=True),
        ton_values=ton_values,
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"stock_test_{idx}",
            "exp_name": f"limit_{limit_p}/random_{ton}",
        },
        payload_builder=make_payload_builder(
            "get_stock_test_data_and_set_condict",
            extra_fields={"limit_change_percentage": limit_p},
        ),
    )
    return generate_inputs(model_name, first_n_img, spec).inputs


def imdb_shap_1_2_3_4_8_range02(
    model_name: str,
    first_n_img: int,
    model_type: str = "cnn",
) -> List[Dict[str, Any]]:
    limit_p = 0.2
    shap_pixels = np.load(f"./shap_value/{model_name}/imdb_sort_shap_pixel.npy")

    def save_exp_builder(idx, ton, mode):
        exp_name = f"limit_up_86400_{limit_p}/shap_{ton}"
        if model_type == "cnn":
            exp_name = f"lstm_limit_{limit_p}/shap_{ton}"
        return {
            "input_name": f"imdb_test_{idx}",
            "exp_name": exp_name,
            "only_first_forward": True,
        }

    spec = TaskGenerationSpec(
        dataset_factory=IMDB_Dataset,
        attack_pixel_fn=make_shap_provider(shap_pixels),
        queue_modes=queue_modes(include_queue=True, include_stack=False),
        ton_values=[128, 256, 512],
        save_exp_builder=save_exp_builder,
        payload_builder=make_payload_builder(
            "get_imdb_test_data_and_set_condict",
            extra_fields={
                "limit_change_percentage": limit_p,
                "only_first_forward": True,
            },
        ),
    )
    return generate_inputs(model_name, first_n_img, spec).inputs


def imdb_transformer_shap_1_2_3_4_8_range02(
    model_name: str,
    first_n_img: int,
    model_type: str = "cnn",
) -> List[Dict[str, Any]]:
    limit_p = 0.2
    shap_pixels = np.load(f"./shap_value/{model_name}/imdb_sort_shap_pixel.npy")

    def save_exp_builder(idx, ton, mode):
        exp_name = f"limit_86400_{limit_p}/shap_{ton}"
        if model_type == "cnn":
            exp_name = f"lstm_limit_{limit_p}/shap_{ton}"
        return {
            "input_name": f"imdb_test_{idx}",
            "exp_name": exp_name,
        }

    spec = TaskGenerationSpec(
        dataset_factory=IMDB_Dataset,
        attack_pixel_fn=make_shap_provider(shap_pixels),
        queue_modes=queue_modes(include_queue=True, include_stack=False),
        ton_values=[1],
        save_exp_builder=save_exp_builder,
        payload_builder=make_payload_builder(
            "get_imdb_test_data_and_set_condict",
            extra_fields={"limit_change_percentage": limit_p},
        ),
        skip_existing=False,
    )
    return generate_inputs(model_name, first_n_img, spec).inputs


def mnist_lstm_1_2_3_4_8_range02(
    model_name: str,
    first_n_img: int,
    model_type: str = "tnn",
) -> List[Dict[str, Any]]:
    limit_p = 0.2
    shap_pixels = np.load("./shap_value/mnist_lstm_09785/mnist_sort_shap_pixel.npy")

    def save_exp_builder(idx, ton, mode):
        exp_name = f"limit_mnist_shap_3600_{limit_p}/shap_{ton}"
        if model_type == "cnn":
            exp_name = f"lstm_limit_{limit_p}/shap_{ton}"
        return {
            "input_name": f"imdb_test_{idx}",
            "exp_name": exp_name,
        }

    spec = TaskGenerationSpec(
        dataset_factory=RNN_MnistDataset,
        attack_pixel_fn=make_shap_provider(shap_pixels),
        queue_modes=queue_modes(include_queue=True, include_stack=False),
        ton_values=[1, 2, 4, 8],
        save_exp_builder=save_exp_builder,
        payload_builder=make_payload_builder(
            "get_mnist_test_data_and_set_condict",
            extra_fields={"limit_change_percentage": limit_p},
        ),
    )
    return generate_inputs(model_name, first_n_img, spec).inputs


def mnist_lstm_15_1_2_3_4_8_range02(
    model_name: str,
    first_n_img: int,
    model_type: str = "tnn",
) -> List[Dict[str, Any]]:
    limit_p = 0.2
    shap_pixels = np.load("./shap_value/mnist_lstm_09785/mnist_sort_shap_pixel.npy")

    def save_exp_builder(idx, ton, mode):
        exp_name = f"limit_mnist_0.25_shap_3600_{limit_p}/shap_{ton}"
        if model_type == "cnn":
            exp_name = f"lstm_limit_{limit_p}/shap_{ton}"
        return {
            "input_name": f"imdb_test_{idx}",
            "exp_name": exp_name,
        }

    spec = TaskGenerationSpec(
        dataset_factory=RNN_MnistDataset,
        attack_pixel_fn=make_shap_provider(shap_pixels),
        queue_modes=queue_modes(include_queue=True, include_stack=False),
        ton_values=[1, 2, 4, 8],
        save_exp_builder=save_exp_builder,
        payload_builder=make_payload_builder(
            "get_mnist_test_data_and_set_condict",
            extra_fields={"limit_change_percentage": limit_p},
        ),
    )
    return generate_inputs(model_name, first_n_img, spec).inputs


def sentiment_lstm_lstm_15_1_2_3_4_8_range02(
    model_name: str,
    first_n_img: int,
    model_type: str = "tnn",
) -> List[Dict[str, Any]]:
    limit_p = 0.2
    shap_pixels = np.load("./shap_value/imdb_LSTM_08509/imdb_sort_shap_pixel.npy")

    def save_exp_builder(idx, ton, mode):
        exp_name = f"limit_mnist_shap_7200_{limit_p}/shap_{ton}"
        if model_type == "cnn":
            exp_name = f"lstm_limit_{limit_p}/shap_{ton}"
        return {
            "input_name": f"imdb_test_{idx}",
            "exp_name": exp_name,
        }

    spec = TaskGenerationSpec(
        dataset_factory=IMDB_Dataset,
        attack_pixel_fn=make_shap_provider(shap_pixels),
        queue_modes=queue_modes(include_queue=True, include_stack=False),
        ton_values=[1, 2, 4, 8],
        save_exp_builder=save_exp_builder,
        payload_builder=make_payload_builder(
            "get_imdb_test_data_and_set_condict",
            extra_fields={"limit_change_percentage": limit_p},
        ),
    )
    return generate_inputs(model_name, first_n_img, spec).inputs


__all__ = [
    "pyct_shap_1_4_8_16_32",
    "pyct_random_1_4_8_16_32",
    "pyct_rnn_random_1_4_8_16_32",
    "pyct_rnn_shap_1_4_8_16_32",
    "stock_shap_1_2_3_4_8_limit_range02",
    "stock_random_1_2_3_4_8_range02",
    "imdb_shap_1_2_3_4_8_range02",
    "imdb_transformer_shap_1_2_3_4_8_range02",
    "mnist_lstm_1_2_3_4_8_range02",
    "mnist_lstm_15_1_2_3_4_8_range02",
    "sentiment_lstm_lstm_15_1_2_3_4_8_range02",
]
