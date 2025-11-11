import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from libct.shapInfl import ShapValuesCalculator

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
    "fashion_mnist_transformer_random",
]


def get_save_dir_from_save_exp(
    save_exp: Dict[str, str],
    model_name: str,
    s_or_q: str,
    *,
    only_first_forward: bool = False,
) -> str:
    if only_first_forward:
        return os.path.join(
            "exp",
            f"{model_name}_only_first_forward",
            s_or_q,
            save_exp["exp_name"],
            save_exp["input_name"],
        )
    return os.path.join("exp", model_name, s_or_q, save_exp["exp_name"], save_exp["input_name"])


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
                save_dir = get_save_dir_from_save_exp(
                    save_exp,
                    model_name,
                    mode.identifier,
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
    exp_prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from utils.dataset import FashionMnistDataset

    shap_pixels = np.load(f"./shap_value/{model_name}/{model_name}_sort_pixel_3d.npy")
    queue_mode = QueueMode("priority_queue", "priority_queue")

    def payload_builder(
        dataset: Any,
        idx: int,
        attack_pixels: List[Any],
        ton: int,
        mode: QueueMode,
    ) -> Dict[str, Any]:
        in_dict, con_dict, input_for_shap, background_dataset_for_shap = (
            dataset.get_fashion_mnist_test_data_and_set_condict(idx, attack_pixels)
        )
        return {
            "idx": idx,
            "in_dict": in_dict,
            "con_dict": con_dict,
            "input_for_shap": input_for_shap,
            "background_dataset_for_shap": background_dataset_for_shap,
            "solve_order_stack": mode.solve_order_stack,
            "shap_value_pre_calculated": True,
            "popped_log_attack_mode": "shap",
        }

    prefix = f"{exp_prefix.strip('/')}/" if exp_prefix else ""
    spec = TaskGenerationSpec(
        dataset_factory=FashionMnistDataset,
        attack_pixel_fn=_make_shap_provider(shap_pixels),
        queue_modes=[queue_mode],
        ton_values=[1],
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"fashion_mnist_test_{idx}",
            "exp_name": f"{prefix}shap_{ton}",
        },
        payload_builder=payload_builder,
    )

    result = _generate_inputs(
        model_name,
        first_n_img,
        spec,
        skip_existing_override=not force,
    )
    log.info("built inputs=%s skipped=%s", len(result.inputs), result.skipped)
    return result.inputs


def fashion_mnist_transformer_random(
    model_name: str,
    first_n_img: Iterable[int],
    *,
    ton_values: Sequence[int],
    force: bool = False,
    base_seed: int = 2024,
    exp_prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from utils.dataset import FashionMnistDataset

    ton_values = tuple(ton_values)
    if not ton_values:
        raise ValueError("ton_values must contain at least one value.")

    dataset = FashionMnistDataset()
    sample_shape = tuple(int(dim) for dim in dataset.x_test.shape[1:])
    coordinate_provider = _make_coordinate_provider(sample_shape, ton_values, base_seed=base_seed)
    queue_mode = QueueMode("priority_queue", "priority_queue")

    def attack_pixel_fn(idx: int, ton: int) -> List[Any]:
        return [list(coord) for coord in coordinate_provider(idx, ton)]

    def payload_builder(
        dataset_obj: Any,
        idx: int,
        attack_pixels: List[Any],
        ton: int,
        mode: QueueMode,
    ) -> Dict[str, Any]:
        (
            in_dict,
            con_dict,
            input_for_shap,
            background_dataset_for_shap,
        ) = dataset_obj.get_fashion_mnist_test_data_and_set_condict(
            idx,
            [tuple(pixel) for pixel in attack_pixels],
        )
        return {
            "idx": idx,
            "in_dict": in_dict,
            "con_dict": con_dict,
            "input_for_shap": input_for_shap,
            "background_dataset_for_shap": background_dataset_for_shap,
            "solve_order_stack": mode.solve_order_stack,
            "shap_value_pre_calculated": True,
            "popped_log_attack_mode": "random",
        }

    prefix = exp_prefix.strip("/") if exp_prefix else "random_select"
    spec = TaskGenerationSpec(
        dataset_factory=lambda: dataset,
        attack_pixel_fn=attack_pixel_fn,
        queue_modes=[queue_mode],
        ton_values=list(ton_values),
        save_exp_builder=lambda idx, ton, mode: {
            "input_name": f"fashion_mnist_test_{idx}",
            "exp_name": f"{prefix}/random_{ton}",
        },
        payload_builder=payload_builder,
    )

    result = _generate_inputs(
        model_name,
        first_n_img,
        spec,
        skip_existing_override=not force,
    )
    log.info("built inputs=%s skipped=%s", len(result.inputs), result.skipped)
    return result.inputs


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

    for idx in indices:
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
