import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional UX dependency
    tqdm = None

from libct.shapInfl import ShapValuesCalculator
from libct.shap_pixel_provider import JsonShapPixelProvider

log = logging.getLogger("ct.experiment")

__all__ = [
    "get_save_dir_from_save_exp",
    "QueueMode",
    "TaskGenerationSpec",
    "GenerationResult",
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
    "fashion_mnist_transformer_shap",
    "fashion_mnist_transformer_shap_calculate_all",
    "mnist_transformer_shap_calculate_all",
    "cifar10_cal_shap_specs",
    "fashion_mnist_transformer_random",
    "mnist_transformer_shap",
    "mnist_transformer_random",
    "cifar10_transformer_shap",
    "cifar10_transformer_random",
]


def get_save_dir_from_save_exp(
    save_exp: Dict[str, str],
    model_name: str,
    attack_mode: str,
    *,
    only_first_forward: bool = False,
    timeout: Optional[int] = None,
    constraint_build_timeout: Optional[bool] = None,
    constraint_build_timeout_seconds: Optional[int] = None,
    score_alpha: Optional[float] = None,
    symbolic_path_threshold: Optional[int] = None,
) -> str:
    def _format_alpha(value: Optional[Any]) -> str:
        if value is None:
            return "aNA"
        try:
            val = float(value)
        except (TypeError, ValueError):
            return f"a{value}"
        scaled = val * 10.0
        if abs(scaled - round(scaled)) < 1e-6:
            return f"a{int(round(scaled)):02d}"
        cleaned = f"{val:g}".replace(".", "p").replace("-", "m")
        return f"a{cleaned}"

    def _resolve_value(key: str, explicit: Optional[Any], env_key: str) -> Optional[Any]:
        if explicit is not None:
            return explicit
        if isinstance(save_exp, dict) and key in save_exp:
            return save_exp.get(key)
        return os.environ.get(env_key)

    def _resolve_bool(key: str, explicit: Optional[bool], env_key: str) -> Optional[bool]:
        if explicit is not None:
            return bool(explicit)
        if isinstance(save_exp, dict) and key in save_exp:
            return bool(save_exp.get(key))
        raw = os.environ.get(env_key)
        if raw is None:
            return None
        return raw.lower() not in {"0", "false", "no", "off"}

    def _format_component(value: Optional[Any]) -> str:
        if value is None:
            return "na"
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    def _format_build_timeout_component(enabled: Optional[bool], seconds: Optional[Any]) -> str:
        if enabled is False:
            return "0"
        if seconds is None:
            return "30"
        try:
            return str(int(seconds))
        except (TypeError, ValueError):
            return _format_component(seconds)

    base_model = f"{model_name}_only_first_forward" if only_first_forward else model_name
    timeout_val = _resolve_value("timeout", timeout, "PYCT_TIMEOUT")
    build_timeout_enabled = _resolve_bool(
        "constraint_build_timeout",
        constraint_build_timeout,
        "PYCT_CONSTRAINT_BUILD_TIMEOUT_ENABLED",
    )
    build_timeout_seconds_val = _resolve_value(
        "constraint_build_timeout_seconds",
        constraint_build_timeout_seconds,
        "PYCT_CONSTRAINT_BUILD_TIMEOUT_SECONDS",
    )
    alpha_val = _resolve_value("score_alpha", score_alpha, "PYCT_SCORE_ALPHA")
    threshold_val = _resolve_value("symbolic_path_threshold", symbolic_path_threshold, "PYCT_SYMBOLIC_PATH_THRESHOLD")
    alpha_component = _format_alpha(alpha_val)
    build_timeout_component = _format_build_timeout_component(
        build_timeout_enabled,
        build_timeout_seconds_val,
    )
    base_dir = "{}_{}_{}_{}_{}_{}".format(
        base_model,
        attack_mode,
        _format_component(timeout_val),
        build_timeout_component,
        alpha_component,
        _format_component(threshold_val),
    )
    idx = save_exp.get("idx")
    if idx is None:
        # try to infer from input_name fallback
        input_name = save_exp.get("input_name", "")
        if input_name.startswith("case_"):
            try:
                idx = int(input_name.split("_")[-1])
            except ValueError:
                idx = "unknown"
        else:
            idx = "unknown"
    case_name = save_exp.get("input_name", f"case_{idx}")
    return os.path.join("exp", base_dir, case_name)


def _always_false(_: Dict[str, Any], __: Any) -> bool:
    return False


@dataclass(frozen=True)
class QueueMode:
    solve_order_stack: Any
    identifier: str


@dataclass
class TaskGenerationSpec:
    dataset_factory: Callable[[], Any]
    attack_pixel_fn: Callable[[int, int], List[Any]]
    queue_modes: Sequence[QueueMode]
    ton_values: Sequence[int]
    save_exp_builder: Callable[[int, int, QueueMode], Dict[str, Any]]
    payload_builder: Callable[[Any, int, List[Any], int, QueueMode], Dict[str, Any]]
    skip_existing: bool = True
    save_dir_flag: Callable[[Dict[str, Any], QueueMode], bool] = field(default=_always_false)


@dataclass
class GenerationResult:
    inputs: List[Dict[str, Any]]
    skipped: int = 0


def _normalize_indices(first_n_img: Any) -> List[int]:
    if isinstance(first_n_img, int):
        return list(range(first_n_img))
    if isinstance(first_n_img, range):
        return list(first_n_img)
    if isinstance(first_n_img, Iterable):
        return list(first_n_img)
    raise TypeError(f"Unsupported type for first_n_img: {type(first_n_img)!r}")


def _iter_cases(indices: Sequence[int], *, desc: str) -> Iterable[int]:
    if tqdm is None:
        return indices
    return tqdm(indices, desc=desc, unit="case", dynamic_ncols=True)


def _normalize_ton_sequence(
    ton_values: Optional[Sequence[int]],
    *,
    fallback: Optional[int] = None,
) -> Tuple[int, ...]:
    if ton_values is None:
        ton_values = [fallback] if fallback is not None else None
    if ton_values is None:
        raise ValueError("ton_values must be provided or fallback must be set.")

    sequence: List[int] = []
    for value in ton_values:
        ton = int(value)
        if ton < 1:
            raise ValueError("ton_values must all be >= 1.")
        if ton not in sequence:
            sequence.append(ton)
    if not sequence:
        raise ValueError("ton_values must contain at least one value.")
    return tuple(sequence)


def _sparsify_con_dict(con_dict: Dict[str, Any]) -> Dict[str, int]:
    """Keep only enabled coordinates to reduce payload size in transformer ton plans."""
    sparse: Dict[str, int] = {}
    for key, value in con_dict.items():
        if value:
            sparse[key] = 1
    return sparse


def _make_shap_provider(shap_array: np.ndarray) -> Callable[[int, int], List[Any]]:
    def provider(idx: int, ton: int) -> List[Any]:
        return shap_array[idx, :ton].tolist()

    return provider


def _make_random_provider(random_array: np.ndarray) -> Callable[[int, int], List[Any]]:
    def provider(idx: int, ton: int) -> List[Any]:
        return random_array[idx, :ton].tolist()

    return provider


def _make_coordinate_provider(
    sample_shape: Tuple[int, ...],
    ton_values: Sequence[int],
    *,
    base_seed: int = 2024,
) -> Callable[[int, int], List[Tuple[int, ...]]]:
    if not ton_values:
        raise ValueError("ton_values must be non-empty for coordinate provider generation.")
    max_ton = max(ton_values)
    dims = len(sample_shape)

    def provider(idx: int, ton: int) -> List[Tuple[int, ...]]:
        if ton > max_ton:
            raise ValueError(f"Requested ton {ton} exceeds configured maximum {max_ton}.")
        rng = np.random.default_rng(base_seed + idx)
        seen: Set[Tuple[int, ...]] = set()
        coords: List[Tuple[int, ...]] = []
        while len(coords) < ton:
            candidate = tuple(int(rng.integers(0, upper)) for upper in sample_shape)
            if candidate in seen:
                continue
            coords.append(candidate)
            seen.add(candidate)
        return coords

    return provider


def _queue_modes(*, include_queue: bool = True, include_stack: bool = False) -> List[QueueMode]:
    modes: List[QueueMode] = []
    if include_queue:
        modes.append(QueueMode(False, "queue"))
    if include_stack:
        modes.append(QueueMode(True, "stack"))
    return modes


def _make_payload_builder(
    method_name: str,
    *,
    extra_fields: Optional[Dict[str, Any]] = None,
    extra_factory: Optional[Callable[[int, int, QueueMode], Dict[str, Any]]] = None,
) -> Callable[[Any, int, List[Any], int, QueueMode], Dict[str, Any]]:
    extra_fields = extra_fields or {}

    def builder(
        dataset: Any,
        idx: int,
        attack_pixels: List[Any],
        ton: int,
        mode: QueueMode,
    ) -> Dict[str, Any]:
        method = getattr(dataset, method_name)
        result = method(idx, attack_pixels)
        if not isinstance(result, tuple) or len(result) < 2:
            raise ValueError(f"{method_name} must return a tuple (in_dict, con_dict, ...)")
        in_dict, con_dict = result[:2]
        payload: Dict[str, Any] = {
            "idx": idx,
            "in_dict": in_dict,
            "con_dict": con_dict,
            "solve_order_stack": mode.solve_order_stack,
        }
        if extra_factory:
            payload.update(extra_factory(idx, ton, mode))
        if extra_fields:
            payload.update(extra_fields)
        return payload

    return builder


def _generate_inputs(
    model_name: str,
    first_n_img: Any,
    spec: TaskGenerationSpec,
    *,
    skip_existing_override: Optional[bool] = None,
) -> GenerationResult:
    dataset = spec.dataset_factory()
    indices = _normalize_indices(first_n_img)
    skip_existing = spec.skip_existing if skip_existing_override is None else skip_existing_override

    inputs: List[Dict[str, Any]] = []
    skipped = 0

    for mode in spec.queue_modes:
        for ton in spec.ton_values:
            for idx in indices:
                save_exp = spec.save_exp_builder(idx, ton, mode)
                save_exp.setdefault("idx", idx)
                attack_mode = save_exp.get("attack_mode", mode.identifier)
                save_dir = get_save_dir_from_save_exp(
                    save_exp,
                    model_name,
                    attack_mode,
                    only_first_forward=spec.save_dir_flag(save_exp, mode),
                )
                if skip_existing and os.path.exists(save_dir):
                    skipped += 1
                    continue
                attack_pixels = spec.attack_pixel_fn(idx, ton)
                payload = spec.payload_builder(dataset, idx, attack_pixels, ton, mode)
                entry: Dict[str, Any] = {"model_name": model_name, "save_exp": save_exp}
                entry.update(payload)
                inputs.append(entry)

    return GenerationResult(inputs=inputs, skipped=skipped)


# ----- Fashion-MNIST Transformer -------------------------------------------


def fashion_mnist_transformer_shap(
    model_name: str,
    first_n_img: Iterable[int],
    force: bool = False,
    *,
    ton_values: Optional[Sequence[int]] = None,
    ton: Optional[int] = None,
    exp_prefix: Optional[str] = None,
    attack_mode: str = "shap",
) -> List[Dict[str, Any]]:
    from utils.dataset import FashionMnistDataset

    ton_sequence = _normalize_ton_sequence(ton_values, fallback=ton or 1)
    dataset = FashionMnistDataset()
    sample_shape = tuple(int(dim) for dim in dataset.x_test.shape[1:])

    pixel_provider = JsonShapPixelProvider(
        model_name=model_name,
        shap_root="shap_value_all_layer",
        coordinate_dims=3,
        coordinate_bounds=sample_shape,
    )
    queue_mode = QueueMode("priority_queue", "priority_queue")

    prefix = f"{exp_prefix.strip('/')}/" if exp_prefix else ""
    indices = _normalize_indices(first_n_img)

    inputs: List[Dict[str, Any]] = []
    skipped = 0

    for idx in indices:
        ton_plans: List[Dict[str, Any]] = []
        base_in_dict: Optional[Dict[str, Any]] = None
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
                    "con_dict": _sparsify_con_dict(con_dict),
                    "save_exp": save_exp,
                }
            )

        if not ton_plans or base_in_dict is None:
            continue

        entry: Dict[str, Any] = {
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


def mnist_transformer_shap(
    model_name: str,
    first_n_img: Iterable[int],
    force: bool = False,
    *,
    ton_values: Optional[Sequence[int]] = None,
    ton: Optional[int] = None,
    exp_prefix: Optional[str] = None,
    attack_mode: str = "shap",
) -> List[Dict[str, Any]]:
    from utils.dataset import MnistDataset

    ton_sequence = _normalize_ton_sequence(ton_values, fallback=ton or 1)
    dataset = MnistDataset()
    sample_shape = tuple(int(dim) for dim in dataset.x_test.shape[1:])

    pixel_provider = JsonShapPixelProvider(
        model_name=model_name,
        shap_root="shap_value_all_layer",
        coordinate_dims=3,
        coordinate_bounds=sample_shape,
    )
    queue_mode = QueueMode("priority_queue", "priority_queue")

    prefix = f"{exp_prefix.strip('/')}/" if exp_prefix else ""
    indices = _normalize_indices(first_n_img)

    inputs: List[Dict[str, Any]] = []
    skipped = 0

    for idx in indices:
        ton_plans: List[Dict[str, Any]] = []
        base_in_dict: Optional[Dict[str, Any]] = None
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
            ) = dataset.get_mnist_test_data(idx)
            for i, j, k in attack_pixels:
                key = f"v_{i}_{j}_{k}"
                if key in con_dict:
                    con_dict[key] = 1
            if base_in_dict is None:
                base_in_dict = in_dict
            ton_plans.append(
                {
                    "ton": ton_value,
                    "con_dict": _sparsify_con_dict(con_dict),
                    "save_exp": save_exp,
                }
            )

        if not ton_plans or base_in_dict is None:
            continue

        entry: Dict[str, Any] = {
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


def cifar10_transformer_shap(
    model_name: str,
    first_n_img: Iterable[int],
    force: bool = False,
    *,
    ton_values: Optional[Sequence[int]] = None,
    ton: Optional[int] = None,
    exp_prefix: Optional[str] = None,
    attack_mode: str = "shap",
    pixel_selector: str = "pixel-shap",
) -> List[Dict[str, Any]]:
    from utils.dataset import Cifar10Dataset

    ton_sequence = _normalize_ton_sequence(ton_values, fallback=ton or 1)
    if pixel_selector in {"patch-shap", "token-shap"} and tuple(ton_sequence) != (1,):
        raise ValueError(f"{pixel_selector} supports only --pixel-search 1 in v1.")
    dataset = Cifar10Dataset()
    sample_shape = tuple(int(dim) for dim in dataset.x_test.shape[1:])

    pixel_provider = JsonShapPixelProvider(
        model_name=model_name,
        shap_root="shap_value_all_layer",
        selector=pixel_selector,
        coordinate_dims=3,
        coordinate_bounds=sample_shape,
    )
    queue_mode = QueueMode("priority_queue", "priority_queue")

    prefix = f"{exp_prefix.strip('/')}/" if exp_prefix else ""
    indices = _normalize_indices(first_n_img)

    inputs: List[Dict[str, Any]] = []
    skipped = 0

    for idx in indices:
        ton_plans: List[Dict[str, Any]] = []
        base_in_dict: Optional[Dict[str, Any]] = None
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
            ) = dataset.get_cifar10_test_data_and_set_condict(idx, attack_pixels)
            if base_in_dict is None:
                base_in_dict = in_dict
            ton_plans.append(
                {
                    "ton": ton_value,
                    "con_dict": _sparsify_con_dict(con_dict),
                    "save_exp": save_exp,
                }
            )

        if not ton_plans or base_in_dict is None:
            continue

        entry: Dict[str, Any] = {
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
) -> List[Dict[str, Any]]:
    from utils.dataset import FashionMnistDataset

    ton_sequence = _normalize_ton_sequence(ton_values)

    dataset = FashionMnistDataset()
    sample_shape = tuple(int(dim) for dim in dataset.x_test.shape[1:])
    coordinate_provider = _make_coordinate_provider(sample_shape, ton_sequence, base_seed=base_seed)
    queue_mode = QueueMode("priority_queue", "priority_queue")

    prefix = exp_prefix.strip("/") if exp_prefix else "random_select"
    indices = _normalize_indices(first_n_img)

    inputs: List[Dict[str, Any]] = []
    skipped = 0

    for idx in indices:
        ton_plans: List[Dict[str, Any]] = []
        base_in_dict: Optional[Dict[str, Any]] = None
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
                    "con_dict": _sparsify_con_dict(con_dict),
                    "save_exp": save_exp,
                }
            )

        if not ton_plans or base_in_dict is None:
            continue

        entry: Dict[str, Any] = {
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


def mnist_transformer_random(
    model_name: str,
    first_n_img: Iterable[int],
    *,
    ton_values: Sequence[int],
    force: bool = False,
    base_seed: int = 2024,
    exp_prefix: Optional[str] = None,
    attack_mode: str = "random",
) -> List[Dict[str, Any]]:
    from utils.dataset import MnistDataset

    ton_sequence = _normalize_ton_sequence(ton_values)

    dataset = MnistDataset()
    sample_shape = tuple(int(dim) for dim in dataset.x_test.shape[1:])
    coordinate_provider = _make_coordinate_provider(sample_shape, ton_sequence, base_seed=base_seed)
    queue_mode = QueueMode("priority_queue", "priority_queue")

    prefix = exp_prefix.strip("/") if exp_prefix else "random_select"
    indices = _normalize_indices(first_n_img)

    inputs: List[Dict[str, Any]] = []
    skipped = 0

    for idx in indices:
        ton_plans: List[Dict[str, Any]] = []
        base_in_dict: Optional[Dict[str, Any]] = None
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
            ) = dataset.get_mnist_test_data(idx)
            for i, j, k in attack_pixels:
                key = f"v_{i}_{j}_{k}"
                if key in con_dict:
                    con_dict[key] = 1
            if base_in_dict is None:
                base_in_dict = in_dict
            ton_plans.append(
                {
                    "ton": ton_value,
                    "con_dict": _sparsify_con_dict(con_dict),
                    "save_exp": save_exp,
                }
            )

        if not ton_plans or base_in_dict is None:
            continue

        entry: Dict[str, Any] = {
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


def cifar10_transformer_random(
    model_name: str,
    first_n_img: Iterable[int],
    *,
    ton_values: Sequence[int],
    force: bool = False,
    base_seed: int = 2024,
    exp_prefix: Optional[str] = None,
    attack_mode: str = "random",
) -> List[Dict[str, Any]]:
    from utils.dataset import Cifar10Dataset

    ton_sequence = _normalize_ton_sequence(ton_values)

    dataset = Cifar10Dataset()
    sample_shape = tuple(int(dim) for dim in dataset.x_test.shape[1:])
    coordinate_provider = _make_coordinate_provider(sample_shape, ton_sequence, base_seed=base_seed)
    queue_mode = QueueMode("priority_queue", "priority_queue")

    prefix = exp_prefix.strip("/") if exp_prefix else "random_select"
    indices = _normalize_indices(first_n_img)

    inputs: List[Dict[str, Any]] = []
    skipped = 0

    for idx in indices:
        ton_plans: List[Dict[str, Any]] = []
        base_in_dict: Optional[Dict[str, Any]] = None
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
            ) = dataset.get_cifar10_test_data_and_set_condict(
                idx,
                [tuple(pixel) for pixel in attack_pixels],
            )
            if base_in_dict is None:
                base_in_dict = in_dict
            ton_plans.append(
                {
                    "ton": ton_value,
                    "con_dict": _sparsify_con_dict(con_dict),
                    "save_exp": save_exp,
                }
            )

        if not ton_plans or base_in_dict is None:
            continue

        entry: Dict[str, Any] = {
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
    output_root: str = "shap_value_all_layer",
) -> List[Dict[str, Any]]:
    from utils.dataset import FashionMnistDataset

    dataset = FashionMnistDataset()
    indices = _normalize_indices(first_n_img)
    artifacts: List[Dict[str, Any]] = []

    for idx in _iter_cases(indices, desc="fashion_mnist SHAP"):
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
                "was_cached": cache_exists and not force_refresh,
            }
        )

    return artifacts


def mnist_transformer_shap_calculate_all(
    model_name: str,
    first_n_img: int,
    *,
    force_refresh: bool = False,
    explainer_type: str = "gradient",
    output_root: str = "shap_value_all_layer",
) -> List[Dict[str, Any]]:
    from utils.dataset import MnistDataset

    dataset = MnistDataset()
    indices = _normalize_indices(first_n_img)
    artifacts: List[Dict[str, Any]] = []

    for idx in _iter_cases(indices, desc="mnist SHAP"):
        (
            _,
            _,
            input_for_shap,
            background_dataset_for_shap,
        ) = dataset.get_mnist_test_data(idx)

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
                "was_cached": cache_exists and not force_refresh,
            }
        )

    return artifacts


def cifar10_cal_shap_specs(
    model_name: str,
    first_n_img: int,
    *,
    force_refresh: bool = False,
    explainer_type: str = "gradient",
    output_root: str = "shap_value_all_layer",
) -> List[Dict[str, Any]]:
    from utils.dataset import Cifar10Dataset

    dataset = Cifar10Dataset()
    indices = _normalize_indices(first_n_img)
    artifacts: List[Dict[str, Any]] = []

    for idx in _iter_cases(indices, desc="cifar10 SHAP"):
        (
            _,
            _,
            input_for_shap,
            background_dataset_for_shap,
        ) = dataset.get_cifar10_test_data_and_set_condict(idx, [])

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
                "was_cached": cache_exists and not force_refresh,
            }
        )

    return artifacts


# ----- MNIST (CNN) -----------------------------------------------------------


def pyct_shap_1_4_8_16_32(model_name: str, first_n_img: int) -> List[Dict[str, Any]]:
    from utils.dataset import MnistDataset

    shap_pixels = np.load(f"./shap_value/{model_name}/mnist_sort_shap_pixel.npy")
    spec = TaskGenerationSpec(
        dataset_factory=MnistDataset,
        attack_pixel_fn=_make_shap_provider(shap_pixels),
        queue_modes=_queue_modes(include_stack=True),
        ton_values=[1],
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"mnist_test_{idx}",
            "exp_name": f"shap_{ton}",
        },
        payload_builder=_make_payload_builder("get_mnist_test_data_and_set_condict"),
    )
    return _generate_inputs(model_name, first_n_img, spec).inputs


def pyct_random_1_4_8_16_32(model_name: str, first_n_img: int) -> List[Dict[str, Any]]:
    from utils.dataset import MnistDataset
    from utils.gen_random_pixel_location import mnist_test_data_10000

    random_pixels = mnist_test_data_10000()
    spec = TaskGenerationSpec(
        dataset_factory=MnistDataset,
        attack_pixel_fn=_make_random_provider(random_pixels),
        queue_modes=_queue_modes(include_stack=True),
        ton_values=[1, 4, 8, 16, 32],
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"mnist_test_{idx}",
            "exp_name": f"random_{ton}",
        },
        payload_builder=_make_payload_builder("get_mnist_test_data_and_set_condict"),
    )
    return _generate_inputs(model_name, first_n_img, spec).inputs


# ----- MNIST (RNN) -----------------------------------------------------------


def pyct_rnn_random_1_4_8_16_32(model_name: str, first_n_img: int) -> List[Dict[str, Any]]:
    from utils.dataset import RNN_MnistDataset
    from utils.gen_random_pixel_location import rnn_mnist_test_data_10000

    random_pixels = rnn_mnist_test_data_10000()
    spec = TaskGenerationSpec(
        dataset_factory=RNN_MnistDataset,
        attack_pixel_fn=_make_random_provider(random_pixels),
        queue_modes=_queue_modes(include_stack=True),
        ton_values=[1, 4, 8, 16, 32],
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"mnist_test_{idx}",
            "exp_name": f"random_{ton}",
        },
        payload_builder=_make_payload_builder("get_mnist_test_data_and_set_condict"),
    )
    return _generate_inputs(model_name, first_n_img, spec).inputs


def pyct_rnn_shap_1_4_8_16_32(model_name: str, first_n_img: int) -> List[Dict[str, Any]]:
    from utils.dataset import RNN_MnistDataset

    shap_pixels = np.load(f"./shap_value/{model_name}/mnist_sort_shap_pixel.npy")
    spec = TaskGenerationSpec(
        dataset_factory=RNN_MnistDataset,
        attack_pixel_fn=_make_shap_provider(shap_pixels),
        queue_modes=_queue_modes(include_stack=True),
        ton_values=[1, 4, 8, 16, 32],
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"mnist_test_{idx}",
            "exp_name": f"shap_{ton}",
        },
        payload_builder=_make_payload_builder("get_mnist_test_data_and_set_condict"),
    )
    return _generate_inputs(model_name, first_n_img, spec).inputs


# ----- Stock ----------------------------------------------------------------


def stock_shap_1_2_3_4_8_limit_range02(model_name: str, first_n_img: int) -> List[Dict[str, Any]]:
    from utils.dataset import MSstock_Dataset

    limit_p = 0.2
    shap_pixels = np.load(f"./shap_value/{model_name}/stock_sort_shap_pixel.npy")
    spec = TaskGenerationSpec(
        dataset_factory=MSstock_Dataset,
        attack_pixel_fn=_make_shap_provider(shap_pixels),
        queue_modes=_queue_modes(include_stack=True),
        ton_values=[1, 2, 3, 4, 8],
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"stock_test_{idx}",
            "exp_name": f"limit_{limit_p}/shap_{ton}",
            "save_smt": True,
        },
        payload_builder=_make_payload_builder(
            "get_stock_test_data_and_set_condict",
            extra_fields={"limit_change_percentage": limit_p},
        ),
    )
    return _generate_inputs(model_name, first_n_img, spec).inputs


def stock_random_1_2_3_4_8_range02(model_name: str, first_n_img: int) -> List[Dict[str, Any]]:
    from utils.dataset import MSstock_Dataset
    from utils.gen_random_pixel_location import lstm_stock_strategy_502

    limit_p = 0.2
    random_pixels = lstm_stock_strategy_502()
    spec = TaskGenerationSpec(
        dataset_factory=MSstock_Dataset,
        attack_pixel_fn=_make_random_provider(random_pixels),
        queue_modes=_queue_modes(include_stack=True),
        ton_values=[1, 2, 3, 4, 8],
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"stock_test_{idx}",
            "exp_name": f"limit_{limit_p}/random_{ton}",
        },
        payload_builder=_make_payload_builder(
            "get_stock_test_data_and_set_condict",
            extra_fields={"limit_change_percentage": limit_p},
        ),
    )
    return _generate_inputs(model_name, first_n_img, spec).inputs


# ----- IMDB -----------------------------------------------------------------


def imdb_shap_1_2_3_4_8_range02(
    model_name: str,
    first_n_img: int,
    model_type: str = "cnn",
) -> List[Dict[str, Any]]:
    from utils.dataset import IMDB_Dataset

    limit_p = 0.2
    shap_pixels = np.load(f"./shap_value/{model_name}/imdb_sort_shap_pixel.npy")

    def save_exp_builder(idx: int, ton: int, mode: QueueMode) -> Dict[str, Any]:
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
        attack_pixel_fn=_make_shap_provider(shap_pixels),
        queue_modes=_queue_modes(include_queue=True, include_stack=False),
        ton_values=[128, 256, 512],
        save_exp_builder=save_exp_builder,
        payload_builder=_make_payload_builder(
            "get_imdb_test_data_and_set_condict",
            extra_fields={"limit_change_percentage": limit_p},
        ),
    )
    return _generate_inputs(model_name, first_n_img, spec).inputs


def imdb_transformer_shap_1_2_3_4_8_range02(
    model_name: str,
    first_n_img: int,
    model_type: str = "cnn",
) -> List[Dict[str, Any]]:
    from utils.dataset import IMDB_Dataset

    limit_p = 0.2
    shap_pixels = np.load(f"./shap_value/{model_name}/imdb_sort_shap_pixel.npy")

    def save_exp_builder(idx: int, ton: int, mode: QueueMode) -> Dict[str, Any]:
        exp_name = f"limit_86400_{limit_p}/shap_{ton}"
        if model_type == "cnn":
            exp_name = f"lstm_limit_{limit_p}/shap_{ton}"
        return {
            "input_name": f"imdb_test_{idx}",
            "exp_name": exp_name,
        }

    spec = TaskGenerationSpec(
        dataset_factory=IMDB_Dataset,
        attack_pixel_fn=_make_shap_provider(shap_pixels),
        queue_modes=_queue_modes(include_queue=True, include_stack=False),
        ton_values=[1],
        save_exp_builder=save_exp_builder,
        payload_builder=_make_payload_builder(
            "get_imdb_test_data_and_set_condict",
            extra_fields={"limit_change_percentage": limit_p},
        ),
        skip_existing=False,
    )
    return _generate_inputs(model_name, first_n_img, spec).inputs


# ----- LSTM variants --------------------------------------------------------


def mnist_lstm_1_2_3_4_8_range02(
    model_name: str,
    first_n_img: int,
    model_type: str = "tnn",
) -> List[Dict[str, Any]]:
    from utils.dataset import RNN_MnistDataset

    limit_p = 0.2
    shap_pixels = np.load("./shap_value/mnist_lstm_09785/mnist_sort_shap_pixel.npy")

    def save_exp_builder(idx: int, ton: int, mode: QueueMode) -> Dict[str, Any]:
        exp_name = f"limit_mnist_shap_3600_{limit_p}/shap_{ton}"
        if model_type == "cnn":
            exp_name = f"lstm_limit_{limit_p}/shap_{ton}"
        return {
            "input_name": f"imdb_test_{idx}",
            "exp_name": exp_name,
        }

    spec = TaskGenerationSpec(
        dataset_factory=RNN_MnistDataset,
        attack_pixel_fn=_make_shap_provider(shap_pixels),
        queue_modes=_queue_modes(include_queue=True, include_stack=False),
        ton_values=[1, 2, 4, 8],
        save_exp_builder=save_exp_builder,
        payload_builder=_make_payload_builder(
            "get_mnist_test_data_and_set_condict",
            extra_fields={"limit_change_percentage": limit_p},
        ),
    )
    return _generate_inputs(model_name, first_n_img, spec).inputs


def mnist_lstm_15_1_2_3_4_8_range02(
    model_name: str,
    first_n_img: int,
    model_type: str = "tnn",
) -> List[Dict[str, Any]]:
    from utils.dataset import RNN_MnistDataset

    limit_p = 0.2
    shap_pixels = np.load("./shap_value/mnist_lstm_09785/mnist_sort_shap_pixel.npy")

    def save_exp_builder(idx: int, ton: int, mode: QueueMode) -> Dict[str, Any]:
        exp_name = f"limit_mnist_0.25_shap_3600_{limit_p}/shap_{ton}"
        if model_type == "cnn":
            exp_name = f"lstm_limit_{limit_p}/shap_{ton}"
        return {
            "input_name": f"imdb_test_{idx}",
            "exp_name": exp_name,
        }

    spec = TaskGenerationSpec(
        dataset_factory=RNN_MnistDataset,
        attack_pixel_fn=_make_shap_provider(shap_pixels),
        queue_modes=_queue_modes(include_queue=True, include_stack=False),
        ton_values=[1, 2, 4, 8],
        save_exp_builder=save_exp_builder,
        payload_builder=_make_payload_builder(
            "get_mnist_test_data_and_set_condict",
            extra_fields={"limit_change_percentage": limit_p},
        ),
    )
    return _generate_inputs(model_name, first_n_img, spec).inputs


def sentiment_lstm_lstm_15_1_2_3_4_8_range02(
    model_name: str,
    first_n_img: int,
    model_type: str = "tnn",
) -> List[Dict[str, Any]]:
    from utils.dataset import IMDB_Dataset

    limit_p = 0.2
    shap_pixels = np.load("./shap_value/imdb_LSTM_08509/imdb_sort_shap_pixel.npy")

    def save_exp_builder(idx: int, ton: int, mode: QueueMode) -> Dict[str, Any]:
        exp_name = f"limit_mnist_shap_7200_{limit_p}/shap_{ton}"
        if model_type == "cnn":
            exp_name = f"lstm_limit_{limit_p}/shap_{ton}"
        return {
            "input_name": f"imdb_test_{idx}",
            "exp_name": exp_name,
        }

    spec = TaskGenerationSpec(
        dataset_factory=IMDB_Dataset,
        attack_pixel_fn=_make_shap_provider(shap_pixels),
        queue_modes=_queue_modes(include_queue=True, include_stack=False),
        ton_values=[1, 2, 4, 8],
        save_exp_builder=save_exp_builder,
        payload_builder=_make_payload_builder(
            "get_imdb_test_data_and_set_condict",
            extra_fields={"limit_change_percentage": limit_p},
        ),
    )
    return _generate_inputs(model_name, first_n_img, spec).inputs
