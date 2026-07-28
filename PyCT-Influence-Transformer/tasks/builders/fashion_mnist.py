from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from datasets.fashion_mnist import FashionMnistDataset
from explainability.pixel_provider import JsonShapPixelProvider
from explainability.shap_contract import DEFAULT_TARGET_CLASS_SHAP_ROOT
from explainability.shap_calculator import ShapValuesCalculator
from tasks.builders.common import (
    QueueMode,
    iter_cases,
    log,
    make_coordinate_provider,
    normalize_indices,
    normalize_ton_sequence,
    sparsify_con_dict,
)
from tasks.paths import get_save_dir_from_save_exp


def fashion_mnist_transformer_shap(
    model_name: str,
    first_n_img: Iterable[int],
    force: bool = False,
    *,
    ton_values: Optional[Sequence[int]] = None,
    ton: Optional[int] = None,
    exp_prefix: Optional[str] = None,
    attack_mode: str = "shap",
) -> List[Dict[str, object]]:
    ton_sequence = normalize_ton_sequence(ton_values, fallback=ton or 1)
    dataset = FashionMnistDataset()
    sample_shape = tuple(int(dim) for dim in dataset.x_test.shape[1:])

    pixel_provider = JsonShapPixelProvider(
        model_name=model_name,
        shap_root=DEFAULT_TARGET_CLASS_SHAP_ROOT,
        coordinate_dims=3,
        coordinate_bounds=sample_shape,
    )
    queue_mode = QueueMode("priority_queue", "priority_queue")

    prefix = f"{exp_prefix.strip('/')}/" if exp_prefix else ""
    indices = normalize_indices(first_n_img)

    inputs: List[Dict[str, object]] = []
    skipped = 0

    for idx in indices:
        ton_plans: List[Dict[str, object]] = []
        base_in_dict: Optional[Dict[str, object]] = None
        input_for_shap = None
        background_dataset_for_shap = None
        input_name = f"case_{idx}"

        for ton_value in ton_sequence:
            save_exp = {
                "input_name": input_name,
                "exp_name": f"{prefix}shap_{ton_value}",
                "idx": idx,
                "attack_mode": attack_mode,
            }
            save_dir = get_save_dir_from_save_exp(
                save_exp,
                model_name,
                attack_mode,
                only_first_forward=False,
            )
            if not force and os.path.exists(save_dir):
                skipped += 1
                continue

            attack_pixels = pixel_provider.top_pixels(idx, ton_value)
            (
                in_dict,
                con_dict,
                input_for_shap,
                background_dataset_for_shap,
            ) = dataset.get_fashion_mnist_test_data_and_set_condict(idx, attack_pixels)
            if base_in_dict is None:
                base_in_dict = in_dict
            ton_plans.append(
                {
                    "ton": ton_value,
                    "con_dict": sparsify_con_dict(con_dict),
                    "save_exp": save_exp,
                }
            )

        if not ton_plans or base_in_dict is None:
            continue

        entry: Dict[str, object] = {
            "model_name": model_name,
            "idx": idx,
            "in_dict": base_in_dict,
            "input_for_shap": input_for_shap,
            "background_dataset_for_shap": background_dataset_for_shap,
            "solve_order_stack": queue_mode.solve_order_stack,
            "shap_value_pre_calculated": True,
            "popped_log_attack_mode": attack_mode,
            "ton_plans": ton_plans,
        }
        inputs.append(entry)

    log.info("built inputs=%s skipped=%s", len(inputs), skipped)
    return inputs


def fashion_mnist_transformer_random(
    model_name: str,
    first_n_img: Iterable[int],
    *,
    ton_values: Sequence[int],
    force: bool = False,
    base_seed: int = 2024,
    exp_prefix: Optional[str] = None,
    attack_mode: str = "random",
) -> List[Dict[str, object]]:
    ton_sequence = normalize_ton_sequence(ton_values)

    dataset = FashionMnistDataset()
    sample_shape = tuple(int(dim) for dim in dataset.x_test.shape[1:])
    coordinate_provider = make_coordinate_provider(sample_shape, ton_sequence, base_seed=base_seed)
    queue_mode = QueueMode("priority_queue", "priority_queue")

    prefix = exp_prefix.strip("/") if exp_prefix else "random_select"
    indices = normalize_indices(first_n_img)

    inputs: List[Dict[str, object]] = []
    skipped = 0

    for idx in indices:
        ton_plans: List[Dict[str, object]] = []
        base_in_dict: Optional[Dict[str, object]] = None
        input_for_shap = None
        background_dataset_for_shap = None
        input_name = f"case_{idx}"

        for ton_value in ton_sequence:
            save_exp = {
                "input_name": input_name,
                "exp_name": f"{prefix}/random_{ton_value}",
                "idx": idx,
                "attack_mode": attack_mode,
            }
            save_dir = get_save_dir_from_save_exp(
                save_exp,
                model_name,
                attack_mode,
                only_first_forward=False,
            )
            if not force and os.path.exists(save_dir):
                skipped += 1
                continue

            attack_pixels = [list(coord) for coord in coordinate_provider(idx, ton_value)]
            (
                in_dict,
                con_dict,
                input_for_shap,
                background_dataset_for_shap,
            ) = dataset.get_fashion_mnist_test_data_and_set_condict(
                idx,
                [tuple(pixel) for pixel in attack_pixels],
            )
            if base_in_dict is None:
                base_in_dict = in_dict
            ton_plans.append(
                {
                    "ton": ton_value,
                    "con_dict": sparsify_con_dict(con_dict),
                    "save_exp": save_exp,
                }
            )

        if not ton_plans or base_in_dict is None:
            continue

        entry: Dict[str, object] = {
            "model_name": model_name,
            "idx": idx,
            "in_dict": base_in_dict,
            "input_for_shap": input_for_shap,
            "background_dataset_for_shap": background_dataset_for_shap,
            "solve_order_stack": queue_mode.solve_order_stack,
            "shap_value_pre_calculated": True,
            "popped_log_attack_mode": attack_mode,
            "ton_plans": ton_plans,
        }
        inputs.append(entry)

    log.info("built inputs=%s skipped=%s", len(inputs), skipped)
    return inputs


def fashion_mnist_transformer_shap_calculate_all(
    model_name: str,
    first_n_img: int,
    *,
    force_refresh: bool = False,
    explainer_type: str = "gradient",
    output_root: str = DEFAULT_TARGET_CLASS_SHAP_ROOT,
) -> List[Dict[str, object]]:
    dataset = FashionMnistDataset()
    indices = normalize_indices(first_n_img)
    artifacts: List[Dict[str, object]] = []

    for idx in iter_cases(indices, desc="fashion_mnist SHAP"):
        (
            _,
            _,
            input_for_shap,
            background_dataset_for_shap,
        ) = dataset.get_fashion_mnist_test_data_and_set_condict(idx, [])

        calculator = ShapValuesCalculator(
            model_path=f"./model/{model_name}.h5",
            background_dataset=background_dataset_for_shap,
            input_data=np.expand_dims(input_for_shap, axis=0),
            idx=idx,
            explainer_type=explainer_type,
            output_root=output_root,
        )
        cache_exists = calculator.cache_path.is_file()
        calculator.ensure(
            assume_cached=cache_exists and not force_refresh,
            force_refresh=force_refresh,
        )
        artifacts.append(
            {
                "idx": idx,
                "output_path": str(calculator.cache_path),
                "was_cached": bool(calculator.last_timing["was_cached"]),
                "computed": bool(calculator.last_timing["computed"]),
                "compute_seconds": float(calculator.last_timing["compute_seconds"]),
            }
        )

    return artifacts


__all__ = [
    "fashion_mnist_transformer_shap",
    "fashion_mnist_transformer_random",
    "fashion_mnist_transformer_shap_calculate_all",
    "JsonShapPixelProvider",
]
