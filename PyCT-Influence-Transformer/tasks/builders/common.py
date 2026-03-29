from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional UX dependency
    tqdm = None

from tasks.paths import get_save_dir_from_save_exp
from tasks.types import GenerationResult, QueueMode, TaskGenerationSpec

log = logging.getLogger("ct.experiment")


def normalize_indices(first_n_img: Any) -> List[int]:
    if isinstance(first_n_img, int):
        return list(range(first_n_img))
    if isinstance(first_n_img, range):
        return list(first_n_img)
    if isinstance(first_n_img, Iterable):
        return list(first_n_img)
    raise TypeError(f"Unsupported type for first_n_img: {type(first_n_img)!r}")


def iter_cases(indices: Sequence[int], *, desc: str) -> Iterable[int]:
    if tqdm is None:
        return indices
    return tqdm(indices, desc=desc, unit="case", dynamic_ncols=True)


def normalize_ton_sequence(
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


def sparsify_con_dict(con_dict: Dict[str, Any]) -> Dict[str, int]:
    sparse: Dict[str, int] = {}
    for key, value in con_dict.items():
        if value:
            sparse[key] = 1
    return sparse


def make_shap_provider(shap_array: np.ndarray) -> Callable[[int, int], List[Any]]:
    def provider(idx: int, ton: int) -> List[Any]:
        return shap_array[idx, :ton].tolist()

    return provider


def make_random_provider(random_array: np.ndarray) -> Callable[[int, int], List[Any]]:
    def provider(idx: int, ton: int) -> List[Any]:
        return random_array[idx, :ton].tolist()

    return provider


def make_coordinate_provider(
    sample_shape: Tuple[int, ...],
    ton_values: Sequence[int],
    *,
    base_seed: int = 2024,
) -> Callable[[int, int], List[Tuple[int, ...]]]:
    if not ton_values:
        raise ValueError("ton_values must be non-empty for coordinate provider generation.")
    max_ton = max(ton_values)

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


def queue_modes(*, include_queue: bool = True, include_stack: bool = False) -> List[QueueMode]:
    modes: List[QueueMode] = []
    if include_queue:
        modes.append(QueueMode(False, "queue"))
    if include_stack:
        modes.append(QueueMode(True, "stack"))
    return modes


def make_payload_builder(
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


def generate_inputs(
    model_name: str,
    first_n_img: Any,
    spec: TaskGenerationSpec,
    *,
    skip_existing_override: Optional[bool] = None,
) -> GenerationResult:
    dataset = spec.dataset_factory()
    indices = normalize_indices(first_n_img)
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


__all__ = [
    "log",
    "normalize_indices",
    "iter_cases",
    "normalize_ton_sequence",
    "sparsify_con_dict",
    "make_shap_provider",
    "make_random_provider",
    "make_coordinate_provider",
    "queue_modes",
    "make_payload_builder",
    "generate_inputs",
    "QueueMode",
    "TaskGenerationSpec",
    "GenerationResult",
]
