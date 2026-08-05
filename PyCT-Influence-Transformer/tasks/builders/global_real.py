from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from datasets.cifar10 import Cifar10Dataset
from explainability.input_shap_sign import (
    BOUNDS_MODE_CLIP,
    BOUNDS_MODE_STRICT,
    BOUNDS_MODES,
    TargetClassInputShapProvider,
    build_sign_mask,
    derive_valid_shift_interval,
)
from libct.global_real import GLOBAL_X_INPUT_NAME
from modeling.keras_loader import load_model_with_compat
from tasks.builders.common import log, normalize_indices
from tasks.paths import get_save_dir_from_save_exp


def _predict_class(model, sample: np.ndarray) -> int:
    predictions = np.asarray(model.predict(sample[np.newaxis, ...], verbose=0))
    if predictions.ndim != 2 or predictions.shape[0] != 1 or predictions.shape[1] < 2:
        raise ValueError(
            "global-real requires one multiclass prediction row; "
            f"got {predictions.shape}"
        )
    if not np.isfinite(predictions).all():
        raise ValueError("global-real reference predictions contain NaN or Inf")
    return int(np.argmax(predictions[0]))


def _sign_mapping(sign_mask: np.ndarray) -> Dict[str, int]:
    return {
        "v_" + "_".join(str(int(part)) for part in index): int(sign_mask[index])
        for index in np.ndindex(sign_mask.shape)
    }


def cifar10_global_real(
    model_name: str,
    first_n_img: Iterable[int],
    *,
    force: bool = False,
    attack_mode: str = "global-real",
    requested_min: float = -0.1,
    requested_max: float = 0.1,
    bounds_mode: str = BOUNDS_MODE_CLIP,
    shap_sign_epsilon: float = 0.0,
    shap_output_root: str = "shap_target_class",
) -> List[Dict[str, object]]:
    if not math.isfinite(requested_min) or not math.isfinite(requested_max):
        raise ValueError("global X bounds must be finite")
    if requested_min > requested_max:
        raise ValueError("global X minimum must be <= maximum")
    if not requested_min <= 0.0 <= requested_max:
        raise ValueError("global X bounds must include 0")
    if bounds_mode not in BOUNDS_MODES:
        raise ValueError(f"global X bounds_mode must be one of {', '.join(BOUNDS_MODES)}")
    if not math.isfinite(shap_sign_epsilon) or shap_sign_epsilon < 0:
        raise ValueError("SHAP sign epsilon must be finite and non-negative")

    dataset = Cifar10Dataset()
    indices = normalize_indices(first_n_img)
    model_path = Path("model") / f"{model_name}.h5"
    model = load_model_with_compat(str(model_path))
    provider = TargetClassInputShapProvider(
        model_path=model_path,
        output_root=Path(shap_output_root),
    )
    background = dataset.get_cifar10_test_data_and_set_condict(0, [])[3]

    inputs: List[Dict[str, object]] = []
    skipped = 0
    for idx in indices:
        input_name = f"case_{idx}"
        save_exp = {
            "input_name": input_name,
            "exp_name": "global_real",
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

        sample = np.asarray(dataset.x_test[idx], dtype=np.float32)
        target_class = _predict_class(model, sample)
        attribution = provider.ensure(
            case_index=idx,
            sample=sample,
            background=background,
            target_class=target_class,
        )
        sign_mask = build_sign_mask(
            attribution.values,
            epsilon=shap_sign_epsilon,
        )
        effective_min = float(requested_min)
        effective_max = float(requested_max)
        if bounds_mode == BOUNDS_MODE_STRICT:
            effective_min, effective_max = derive_valid_shift_interval(
                sample,
                sign_mask,
                requested_min=requested_min,
                requested_max=requested_max,
            )

        in_dict, _ = dataset.get_cifar10_test_data(idx)
        in_dict[GLOBAL_X_INPUT_NAME] = 0.0
        global_real_config = {
            "variable_name": GLOBAL_X_INPUT_NAME,
            "requested_min": float(requested_min),
            "requested_max": float(requested_max),
            "effective_min": float(effective_min),
            "effective_max": float(effective_max),
            "bounds_mode": bounds_mode,
            "shap_sign_epsilon": float(shap_sign_epsilon),
            "shap_target_class": target_class,
            "shap_cache_path": str(attribution.cache_path),
            "nonzero_sign_count": int(np.count_nonzero(sign_mask)),
            "sign_by_input": _sign_mapping(sign_mask),
        }
        inputs.append(
            {
                "model_name": model_name,
                "idx": idx,
                "in_dict": in_dict,
                "con_dict": {GLOBAL_X_INPUT_NAME: 1},
                "solve_order_stack": "priority_queue",
                "input_for_shap": sample,
                "background_dataset_for_shap": background,
                "shap_value_pre_calculated": True,
                "shap_output_root": shap_output_root,
                "popped_log_attack_mode": attack_mode,
                "global_real_config": global_real_config,
                "save_exp": save_exp,
            }
        )

    log.info("built global-real inputs=%s skipped=%s", len(inputs), skipped)
    return inputs


__all__ = ["cifar10_global_real"]
