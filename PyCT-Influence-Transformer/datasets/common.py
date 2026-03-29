from __future__ import annotations

from typing import Iterable, Sequence
import os

import numpy as np


def get_background_per_class(default: int = 3) -> int:
    try:
        value = int(os.environ.get("PYCT_BG_PER_CLASS", default))
    except ValueError:
        value = default
    return max(value, 1)


def get_background_seed(default: int = 2233) -> int:
    try:
        return int(os.environ.get("PYCT_BG_SEED", default))
    except ValueError:
        return default


def select_background_per_class(
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    per_class: int,
    seed: int = 0,
) -> np.ndarray:
    y_flat = np.asarray(y_test).reshape(-1)
    rng = np.random.default_rng(seed)
    selected_indices = []
    for label in sorted(set(int(v) for v in y_flat)):
        candidates = np.where(y_flat == label)[0]
        if candidates.size == 0:
            continue
        if candidates.size <= per_class:
            chosen = candidates
        else:
            chosen = rng.choice(candidates, size=per_class, replace=False)
        selected_indices.extend(chosen.tolist())
    if not selected_indices:
        return x_test[:per_class]
    return x_test[np.array(selected_indices)]


def tensor_to_in_dict_and_con_dict(sample: np.ndarray) -> tuple[dict[str, float], dict[str, int]]:
    in_dict: dict[str, float] = {}
    con_dict: dict[str, int] = {}
    for index in np.ndindex(sample.shape):
        key = "v_" + "_".join(str(int(part)) for part in index)
        in_dict[key] = float(sample[index])
        con_dict[key] = 0
    return in_dict, con_dict


def enable_attack_pixels(
    con_dict: dict[str, int],
    attack_pixels: Iterable[Sequence[int]],
) -> None:
    for pixel in attack_pixels:
        key = "v_" + "_".join(str(int(part)) for part in pixel)
        if key in con_dict:
            con_dict[key] = 1


__all__ = [
    "get_background_per_class",
    "get_background_seed",
    "select_background_per_class",
    "tensor_to_in_dict_and_con_dict",
    "enable_attack_pixels",
]
