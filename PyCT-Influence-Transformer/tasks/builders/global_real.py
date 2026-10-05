from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from libct.aces_like import (
    ACES_LIKE_COLOR_SPACE,
    ACES_LIKE_CURVE_VERSION,
    ACES_LIKE_GAMUT_MAPPER,
    ACES_LIKE_PWL_ERROR_METRIC,
    ACES_LIKE_PWL_VALIDATOR_VERSION,
    ACES_LIKE_SHIFT_KINDS,
    DEFAULT_PWL_ERROR_TOLERANCE,
    DEFAULT_PWL_MAX_SEGMENTS,
)
from libct.global_real_probe import (
    DEFAULT_PROBE_INITIAL_POINTS,
    DEFAULT_PROBE_MAX_REFINEMENTS,
    DEFAULT_PROBE_TOLERANCE_FRACTION,
)

from datasets.cifar10 import Cifar10Dataset
from explainability.input_shap_sign import (
    BOUNDS_MODE_CLIP,
    BOUNDS_MODE_STRICT,
    BOUNDS_MODES,
    TargetClassInputShapProvider,
    build_sign_mask,
    derive_valid_affine_shift_interval,
)
from libct.global_real import GLOBAL_X_INPUT_NAME, build_aces_like_global_real_config
from tasks.builders.common import log, normalize_indices
from tasks.paths import get_save_dir_from_save_exp


GLOBAL_SHIFT_KINDS: Tuple[str, ...] = (
    "shap-sign",
    "brightness",
    "contrast",
) + ACES_LIKE_SHIFT_KINDS


def _coefficient_mapping(coefficients: np.ndarray) -> Dict[str, float]:
    return {
        "v_" + "_".join(str(int(part)) for part in index): float(coefficients[index])
        for index in np.ndindex(coefficients.shape)
    }


def _contrast_coefficients(sample: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    if sample.ndim < 1 or sample.shape[-1] < 1:
        raise ValueError(f"contrast shift requires a channel axis, got {sample.shape}")
    spatial_axes = tuple(range(sample.ndim - 1))
    channel_means = np.mean(sample, axis=spatial_axes, keepdims=True, dtype=np.float64)
    return (
        np.asarray(sample, dtype=np.float64) - channel_means,
        channel_means.reshape(-1).tolist(),
    )


def _cifar10_global_real_affine(
    model_name: str,
    first_n_img: Iterable[int],
    *,
    force: bool = False,
    attack_mode: str = "global-real",
    requested_min: float = -0.1,
    requested_max: float = 0.1,
    bounds_mode: str = BOUNDS_MODE_CLIP,
    shift_kind: str = "shap-sign",
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
    if shift_kind not in GLOBAL_SHIFT_KINDS:
        raise ValueError(f"global shift kind must be one of {', '.join(GLOBAL_SHIFT_KINDS)}")
    if not math.isfinite(shap_sign_epsilon) or shap_sign_epsilon < 0:
        raise ValueError("SHAP sign epsilon must be finite and non-negative")

    dataset = Cifar10Dataset()
    indices = normalize_indices(first_n_img)
    provider = None
    background = None
    if shift_kind == "shap-sign":
        model_path = Path("model") / f"{model_name}.h5"
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
        extra_metadata: Dict[str, object] = {"global_shift_kind": shift_kind}
        if shift_kind == "shap-sign":
            assert provider is not None
            attribution = provider.load_cached(
                case_index=idx,
                sample=sample,
                background=background,
            )
            coefficients = build_sign_mask(
                attribution.values,
                epsilon=shap_sign_epsilon,
            ).astype(np.float64)
            extra_metadata.update(
                {
                    "shap_sign_epsilon": float(shap_sign_epsilon),
                    "shap_target_class": attribution.target_class,
                    "shap_cache_path": str(attribution.cache_path),
                    "nonzero_sign_count": int(np.count_nonzero(coefficients)),
                }
            )
        elif shift_kind == "brightness":
            coefficients = np.ones_like(sample, dtype=np.float64)
        else:
            coefficients, channel_means = _contrast_coefficients(sample)
            extra_metadata["contrast_channel_means"] = channel_means

        effective_min = float(requested_min)
        effective_max = float(requested_max)
        if bounds_mode == BOUNDS_MODE_STRICT:
            effective_min, effective_max = derive_valid_affine_shift_interval(
                sample,
                coefficients,
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
            "global_shift_kind": shift_kind,
            "coefficient_by_input": _coefficient_mapping(coefficients),
            **extra_metadata,
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

    log.info(
        "built global-real inputs=%s skipped=%s kind=%s",
        len(inputs),
        skipped,
        shift_kind,
    )
    return inputs


def cifar10_global_real(
    model_name: str,
    first_n_img: Iterable[int],
    *,
    force: bool = False,
    attack_mode: str = "global-real",
    requested_min: float = -0.1,
    requested_max: float = 0.1,
    bounds_mode: str = BOUNDS_MODE_CLIP,
    shift_kind: str = "shap-sign",
    shap_sign_epsilon: float = 0.0,
    shap_output_root: str = "shap_target_class",
    pwl_max_segments: int = DEFAULT_PWL_MAX_SEGMENTS,
    pwl_error_tolerance: float = DEFAULT_PWL_ERROR_TOLERANCE,
    probe_enabled: bool = True,
    probe_initial_points: int = DEFAULT_PROBE_INITIAL_POINTS,
    probe_max_refinements: int = DEFAULT_PROBE_MAX_REFINEMENTS,
    probe_tolerance_fraction: float = DEFAULT_PROBE_TOLERANCE_FRACTION,
    defer_pwl: bool = False,
) -> List[Dict[str, object]]:
    if shift_kind not in ACES_LIKE_SHIFT_KINDS:
        return _cifar10_global_real_affine(
            model_name,
            first_n_img,
            force=force,
            attack_mode=attack_mode,
            requested_min=requested_min,
            requested_max=requested_max,
            bounds_mode=bounds_mode,
            shift_kind=shift_kind,
            shap_sign_epsilon=shap_sign_epsilon,
            shap_output_root=shap_output_root,
        )
    if bounds_mode != BOUNDS_MODE_CLIP:
        raise ValueError("ACES-like global shifts require bounds_mode='clip'")
    if not math.isfinite(requested_min) or not math.isfinite(requested_max):
        raise ValueError("global X bounds must be finite")
    if requested_min >= requested_max or not requested_min <= 0.0 <= requested_max:
        raise ValueError("ACES-like global X bounds must satisfy min < max and include 0")
    if (
        not isinstance(pwl_max_segments, int)
        or isinstance(pwl_max_segments, bool)
        or pwl_max_segments < 1
    ):
        raise ValueError("ACES-like PWL max_segments must be an integer >= 1")
    if not math.isfinite(pwl_error_tolerance) or pwl_error_tolerance <= 0.0:
        raise ValueError("ACES-like PWL error_tolerance must be finite and positive")
    if isinstance(probe_initial_points, bool) or probe_initial_points < 1:
        raise ValueError("GlobalReal probe_initial_points must be an integer >= 1")
    if isinstance(probe_max_refinements, bool) or probe_max_refinements < 0:
        raise ValueError("GlobalReal probe_max_refinements must be an integer >= 0")
    if not math.isfinite(probe_tolerance_fraction) or probe_tolerance_fraction <= 0.0:
        raise ValueError("GlobalReal probe_tolerance_fraction must be finite and positive")

    dataset = Cifar10Dataset()
    inputs: List[Dict[str, object]] = []
    skipped = 0
    for idx in normalize_indices(first_n_img):
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
        sample = np.asarray(dataset.x_test[idx], dtype=np.float64)
        in_dict, _ = dataset.get_cifar10_test_data(idx)
        in_dict[GLOBAL_X_INPUT_NAME] = 0.0
        global_real_config = {
            "variable_name": GLOBAL_X_INPUT_NAME,
            "requested_min": float(requested_min),
            "requested_max": float(requested_max),
            "effective_min": float(requested_min),
            "effective_max": float(requested_max),
            "bounds_mode": bounds_mode,
            "transform_mode": "aces-like-pwl",
            "global_shift_kind": shift_kind,
            "pwl_max_segments": int(pwl_max_segments),
            "pwl_error_tolerance": float(pwl_error_tolerance),
            "pwl_error_metric": ACES_LIKE_PWL_ERROR_METRIC,
            "pwl_validator_version": ACES_LIKE_PWL_VALIDATOR_VERSION,
            "aces_like_color_space": ACES_LIKE_COLOR_SPACE,
            "aces_like_curve_version": ACES_LIKE_CURVE_VERSION,
            "aces_like_gamut_mapper": ACES_LIKE_GAMUT_MAPPER,
            "probe_enabled": bool(probe_enabled),
            "probe_initial_points": int(probe_initial_points),
            "probe_max_refinements": int(probe_max_refinements),
            "probe_tolerance_fraction": float(probe_tolerance_fraction),
        }
        if defer_pwl:
            # Hybrid builds the other axis only after DE selects its seed.
            global_real_config["pwl_deferred"] = True
        else:
            global_real_config = build_aces_like_global_real_config(sample, global_real_config)
        inputs.append(
            {
                "model_name": model_name,
                "idx": idx,
                "in_dict": in_dict,
                "con_dict": {GLOBAL_X_INPUT_NAME: 1},
                "solve_order_stack": "priority_queue",
                "input_for_shap": sample,
                "background_dataset_for_shap": None,
                "shap_value_pre_calculated": True,
                "shap_output_root": shap_output_root,
                "popped_log_attack_mode": attack_mode,
                "global_real_config": global_real_config,
                "save_exp": save_exp,
            }
        )
    log.info("built global-real inputs=%s skipped=%s kind=%s", len(inputs), skipped, shift_kind)
    return inputs


__all__ = ["GLOBAL_SHIFT_KINDS", "cifar10_global_real"]
