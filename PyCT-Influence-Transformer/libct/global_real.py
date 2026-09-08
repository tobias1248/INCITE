from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from libct.aces_like import (
    ACES_LIKE_COLOR_SPACE,
    ACES_LIKE_CURVE_VERSION,
    ACES_LIKE_GAMUT_MAPPER,
    ACES_LIKE_PWL_ERROR_METRIC,
    ACES_LIKE_PWL_VALIDATOR_VERSION,
    ACES_LIKE_SHIFT_KINDS,
    AcesLikeTransformError,
    DEFAULT_PWL_ERROR_TOLERANCE,
    DEFAULT_PWL_MAX_SEGMENTS,
    apply_aces_like_transform,
    build_adaptive_pwl_approximation,
)
from libct.utils import ConcolicObject, get_in_dict_shape, unwrap


GLOBAL_X_INPUT_NAME = "__pyct_global_x"
GLOBAL_X_SMT_NAME = f"{GLOBAL_X_INPUT_NAME}_VAR"
BOUNDS_MODE_CLIP = "clip"
BOUNDS_MODE_STRICT = "strict"
BOUNDS_MODES = (BOUNDS_MODE_CLIP, BOUNDS_MODE_STRICT)
TRANSFORM_MODE_AFFINE = "affine"
TRANSFORM_MODE_ACES_LIKE_PWL = "aces-like-pwl"
TRANSFORM_MODES = (TRANSFORM_MODE_AFFINE, TRANSFORM_MODE_ACES_LIKE_PWL)


def _coerce_exact_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        raise ValueError(f"ACES-like {name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"ACES-like {name} must be an integer") from exc
    if integer != value:
        raise ValueError(f"ACES-like {name} must be an integer")
    return integer


def validate_global_real_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    variable_name = str(config.get("variable_name", GLOBAL_X_INPUT_NAME))
    if variable_name != GLOBAL_X_INPUT_NAME:
        raise ValueError(
            f"global real variable_name must be {GLOBAL_X_INPUT_NAME!r}, "
            f"got {variable_name!r}"
        )

    bounds_mode = str(config.get("bounds_mode", BOUNDS_MODE_CLIP))
    if bounds_mode not in BOUNDS_MODES:
        raise ValueError(
            f"global real bounds_mode must be one of {', '.join(BOUNDS_MODES)}"
        )

    lower = float(config["effective_min"])
    upper = float(config["effective_max"])
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("global real effective bounds must be finite")
    if lower > upper:
        raise ValueError("global real effective_min must be <= effective_max")
    if not lower <= 0.0 <= upper:
        raise ValueError("global real effective bounds must include X=0")

    transform_mode = str(config.get("transform_mode", TRANSFORM_MODE_AFFINE))
    if transform_mode not in TRANSFORM_MODES:
        raise ValueError(
            f"global real transform_mode must be one of {', '.join(TRANSFORM_MODES)}"
        )

    normalized = dict(config)
    normalized.update(
        {
            "variable_name": variable_name,
            "bounds_mode": bounds_mode,
            "effective_min": lower,
            "effective_max": upper,
            "transform_mode": transform_mode,
        }
    )

    if transform_mode == TRANSFORM_MODE_ACES_LIKE_PWL:
        if bounds_mode != BOUNDS_MODE_CLIP:
            raise ValueError(
                "ACES-like transform requires bounds_mode='clip'; strict is not defined"
            )
        kind = str(config.get("global_shift_kind", ""))
        if kind not in ACES_LIKE_SHIFT_KINDS:
            raise ValueError(
                "ACES-like transform requires global_shift_kind to be one of "
                + ", ".join(ACES_LIKE_SHIFT_KINDS)
            )
        raw_knots = config.get("pwl_knots")
        if not isinstance(raw_knots, Sequence) or isinstance(raw_knots, (str, bytes)):
            raise ValueError("ACES-like transform requires a non-empty pwl_knots sequence")
        knots = [float(value) for value in raw_knots]
        if len(knots) < 2 or not all(math.isfinite(value) for value in knots):
            raise ValueError("ACES-like pwl_knots must contain at least two finite values")
        if any(right <= left for left, right in zip(knots, knots[1:])):
            raise ValueError("ACES-like pwl_knots must be strictly increasing")
        if not math.isclose(knots[0], lower, abs_tol=1e-12):
            raise ValueError("ACES-like pwl_knots must start at effective_min")
        if not math.isclose(knots[-1], upper, abs_tol=1e-12):
            raise ValueError("ACES-like pwl_knots must end at effective_max")
        if not any(math.isclose(value, 0.0, abs_tol=1e-12) for value in knots):
            raise ValueError("ACES-like pwl_knots must include X=0")
        tolerance = float(config.get("pwl_error_tolerance", DEFAULT_PWL_ERROR_TOLERANCE))
        max_segments = _coerce_exact_int(
            config.get("pwl_max_segments", DEFAULT_PWL_MAX_SEGMENTS),
            "pwl_max_segments",
        )
        max_abs_error = float(config.get("pwl_max_abs_error", 0.0))
        if "pwl_segment_count" not in config:
            raise ValueError("ACES-like transform requires pwl_segment_count metadata")
        segment_count = _coerce_exact_int(
            config["pwl_segment_count"], "pwl_segment_count"
        )
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("ACES-like pwl_error_tolerance must be finite and positive")
        if max_segments < 1:
            raise ValueError("ACES-like pwl_max_segments must be an integer >= 1")
        if segment_count < 1 or segment_count != len(knots) - 1:
            raise ValueError(
                "ACES-like pwl_segment_count must match the recorded table"
            )
        if max_segments < segment_count:
            raise ValueError(
                "ACES-like pwl_max_segments is smaller than the recorded table"
            )
        if not math.isfinite(max_abs_error) or max_abs_error < 0.0:
            raise ValueError("ACES-like pwl_max_abs_error must be finite and non-negative")

        expected_metadata = {
            "aces_like_color_space": ACES_LIKE_COLOR_SPACE,
            "aces_like_curve_version": ACES_LIKE_CURVE_VERSION,
            "aces_like_gamut_mapper": ACES_LIKE_GAMUT_MAPPER,
            "pwl_error_metric": ACES_LIKE_PWL_ERROR_METRIC,
            "pwl_validator_version": ACES_LIKE_PWL_VALIDATOR_VERSION,
        }
        normalized_metadata = {}
        for key, expected in expected_metadata.items():
            if key not in config:
                raise ValueError(f"ACES-like transform requires {key} metadata")
            actual = str(config[key])
            if actual != expected:
                raise ValueError(
                    f"ACES-like {key} must be {expected!r}, got {actual!r}"
                )
            normalized_metadata[key] = actual
        normalized.update(
            {
                "pwl_knots": knots,
                "pwl_error_tolerance": tolerance,
                "pwl_max_segments": int(max_segments),
                "pwl_max_abs_error": max_abs_error,
                "pwl_segment_count": int(segment_count),
                **normalized_metadata,
                "coefficient_by_input": {},
            }
        )
        normalized.pop("sign_by_input", None)
        return normalized

    raw_coefficients = config.get("coefficient_by_input")
    raw_signs = config.get("sign_by_input")
    if raw_coefficients is not None and raw_signs is not None:
        raise ValueError(
            "global real config cannot define both coefficient_by_input "
            "and sign_by_input"
        )

    coefficients: Dict[str, float] = {}
    if raw_coefficients is None:
        if not isinstance(raw_signs, Mapping) or not raw_signs:
            raise ValueError(
                "global real coefficient_by_input or sign_by_input must be a non-empty mapping"
            )
        for name, value in raw_signs.items():
            if not isinstance(name, str) or not name.startswith("v_"):
                raise ValueError(f"invalid global real input name: {name!r}")
            sign = int(value)
            if sign not in (-1, 0, 1):
                raise ValueError(f"global real sign for {name!r} must be -1, 0, or 1")
            coefficients[name] = float(sign)
    else:
        if not isinstance(raw_coefficients, Mapping) or not raw_coefficients:
            raise ValueError("global real coefficient_by_input must be a non-empty mapping")
        for name, value in raw_coefficients.items():
            if not isinstance(name, str) or not name.startswith("v_"):
                raise ValueError(f"invalid global real input name: {name!r}")
            coefficient = float(value)
            if not math.isfinite(coefficient):
                raise ValueError(f"global real coefficient for {name!r} must be finite")
            coefficients[name] = coefficient

    normalized.pop("sign_by_input", None)
    normalized["coefficient_by_input"] = coefficients
    return normalized


def _input_mapping_to_rgb(
    primitive_inputs: Mapping[str, Any],
) -> Tuple[np.ndarray, Dict[str, Tuple[int, ...]]]:
    pixel_inputs = {
        name: value
        for name, value in primitive_inputs.items()
        if isinstance(name, str) and name.startswith("v_")
    }
    shape = get_in_dict_shape(pixel_inputs)
    if len(shape) < 1 or shape[-1] != 3:
        raise ValueError(
            "ACES-like global real transform requires predictor inputs with a "
            "final RGB channel axis"
        )
    expected_names = {
        "v_" + "_".join(str(index) for index in indices)
        for indices in np.ndindex(shape)
    }
    if set(pixel_inputs) != expected_names:
        missing = sorted(expected_names - set(pixel_inputs))[:3]
        extra = sorted(set(pixel_inputs) - expected_names)[:3]
        raise ValueError(
            "ACES-like RGB input mapping is incomplete "
            f"(missing={missing}, extra={extra})"
        )

    rgb = np.empty(shape, dtype=np.float64)
    coordinates: Dict[str, Tuple[int, ...]] = {}
    for name, raw_value in pixel_inputs.items():
        indices = tuple(int(part) for part in name.split("_")[1:])
        value = float(unwrap(raw_value))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"ACES-like RGB input {name!r} must be finite and in [0, 1]")
        rgb[indices] = value
        coordinates[name] = indices
    return rgb, coordinates


def _build_aces_like_approximation(
    primitive_inputs: Mapping[str, Any],
    config: Mapping[str, Any],
):
    rgb, coordinates = _input_mapping_to_rgb(primitive_inputs)
    try:
        approximation = build_adaptive_pwl_approximation(
            rgb,
            kind=config["global_shift_kind"],
            x_min=config["effective_min"],
            x_max=config["effective_max"],
            max_segments=config["pwl_max_segments"],
            error_tolerance=config["pwl_error_tolerance"],
        )
    except AcesLikeTransformError as exc:
        raise ValueError(f"unable to build ACES-like PWL approximation: {exc}") from exc

    configured_knots = np.asarray(config["pwl_knots"], dtype=np.float64)
    if configured_knots.shape != approximation.knots.shape or not np.allclose(
        configured_knots,
        approximation.knots,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "ACES-like PWL knots do not match the deterministic approximation "
            "built from the input"
        )
    configured_error = float(config.get("pwl_max_abs_error", 0.0))
    if not math.isclose(
        approximation.max_abs_error,
        configured_error,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "ACES-like PWL sampled error does not match the recorded configuration"
        )
    return approximation, rgb, coordinates


def _pwl_symbolic_expression(
    shared_x: Any,
    approximation: Any,
    coordinates: Tuple[int, ...],
):
    branch = None
    for index in range(approximation.segment_count - 1, -1, -1):
        slope, intercept = approximation.affine_for_segment(index)
        candidate = [
            "+",
            ["*", shared_x, f"{float(slope[coordinates]):.15f}"],
            f"{float(intercept[coordinates]):.15f}",
        ]
        if branch is None:
            branch = candidate
        else:
            branch = [
                "ite",
                [
                    "<=",
                    shared_x,
                    f"{float(approximation.knots[index + 1]):.15f}",
                ],
                candidate,
                branch,
            ]
    assert branch is not None
    return branch


def solver_variable_bounds(config: Mapping[str, Any]) -> Dict[str, Tuple[float, float]]:
    normalized = validate_global_real_config(config)
    return {
        GLOBAL_X_SMT_NAME: (
            normalized["effective_min"],
            normalized["effective_max"],
        )
    }


def build_concolic_global_real_kwargs(
    engine: Any,
    primitive_inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    config = validate_global_real_config(engine.global_real_config)
    variable_name = config["variable_name"]
    if variable_name not in primitive_inputs:
        raise ValueError(f"global real input is missing {variable_name!r}")

    shift_value = float(primitive_inputs[variable_name])
    shared_x = ConcolicObject(shift_value, GLOBAL_X_SMT_NAME, engine)
    engine.concolic_name_list.append(GLOBAL_X_SMT_NAME)
    engine.concolic_flag_dict[GLOBAL_X_SMT_NAME] = 1

    if config["transform_mode"] == TRANSFORM_MODE_ACES_LIKE_PWL:
        approximation, _rgb, coordinates = _build_aces_like_approximation(
            primitive_inputs,
            config,
        )
        kwargs: Dict[str, Any] = {}
        concrete_rgb = approximation.evaluate(shift_value)
        for name, raw_value in primitive_inputs.items():
            if name == variable_name:
                continue
            if name not in coordinates:
                kwargs[name] = raw_value
                continue
            engine.concolic_flag_dict[f"{name}_VAR"] = 0
            coordinate = coordinates[name]
            expression = _pwl_symbolic_expression(shared_x, approximation, coordinate)
            kwargs[name] = ConcolicObject(
                float(concrete_rgb[coordinate]),
                expression,
                engine,
            )
        return kwargs

    coefficients = config["coefficient_by_input"]
    pixel_names = {
        name for name in primitive_inputs if isinstance(name, str) and name.startswith("v_")
    }
    if pixel_names != set(coefficients):
        missing = sorted(pixel_names - set(coefficients))[:3]
        extra = sorted(set(coefficients) - pixel_names)[:3]
        raise ValueError(
            "global real coefficient mapping does not match predictor inputs "
            f"(missing={missing}, extra={extra})"
        )

    kwargs: Dict[str, Any] = {}
    for name, raw_value in primitive_inputs.items():
        if name == variable_name:
            continue
        if name not in coefficients:
            kwargs[name] = raw_value
            continue

        engine.concolic_flag_dict[f"{name}_VAR"] = 0
        base = float(raw_value)
        coefficient = coefficients[name]
        if coefficient == 0.0:
            kwargs[name] = base
            continue
        affine = shared_x * coefficient + base
        if config["bounds_mode"] == BOUNDS_MODE_STRICT:
            kwargs[name] = affine
            continue

        concrete = min(max(float(unwrap(affine)), 0.0), 1.0)
        expression = [
            "ite",
            ["<", affine, "0.0"],
            "0.0",
            ["ite", [">", affine, "1.0"], "1.0", affine],
        ]
        kwargs[name] = ConcolicObject(float(concrete), expression, engine)
    return kwargs


def materialize_global_real_details(
    primitive_inputs: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], float, int, Dict[str, Any]]:
    normalized = validate_global_real_config(config)
    if normalized["transform_mode"] != TRANSFORM_MODE_ACES_LIKE_PWL:
        materialized, shift, clipped_count = materialize_global_real_arguments(
            primitive_inputs, normalized
        )
        return materialized, shift, clipped_count, {
            "transform_mode": TRANSFORM_MODE_AFFINE,
            "pwl_segment_index": None,
            "pwl_error_at_x": None,
            "gamut_mapped_pixel_count": 0,
            "hard_clipped_channel_count": clipped_count,
        }

    variable_name = normalized["variable_name"]
    if variable_name not in primitive_inputs:
        raise ValueError(f"global real input is missing {variable_name!r}")
    shift = float(unwrap(primitive_inputs[variable_name]))
    if not math.isfinite(shift):
        raise ValueError("global real X must be finite")
    lower = normalized["effective_min"]
    upper = normalized["effective_max"]
    if shift < lower - 1e-9 or shift > upper + 1e-9:
        raise ValueError(
            f"global real X={shift} is outside [{lower}, {upper}]"
        )

    approximation, rgb, coordinates = _build_aces_like_approximation(
        primitive_inputs,
        normalized,
    )
    try:
        exact = apply_aces_like_transform(
            rgb,
            shift,
            kind=normalized["global_shift_kind"],
        )
    except AcesLikeTransformError as exc:
        raise ValueError(f"unable to materialize ACES-like transform: {exc}") from exc
    approximated_rgb = approximation.evaluate(shift)
    error_at_x = float(np.max(np.abs(exact.rgb - approximated_rgb)))
    if error_at_x > normalized["pwl_error_tolerance"] + 1e-12:
        raise ValueError(
            "ACES-like PWL error at X={} exceeds tolerance {} (observed {})".format(
                shift,
                normalized["pwl_error_tolerance"],
                error_at_x,
            )
        )
    materialized: MutableMapping[str, Any] = {}
    for name, raw_value in primitive_inputs.items():
        if name == variable_name:
            continue
        if name not in coordinates:
            materialized[name] = unwrap(raw_value)
            continue
        materialized[name] = float(approximated_rgb[coordinates[name]])
    diagnostics = exact.diagnostics
    return dict(materialized), shift, diagnostics.hard_clipped_channel_count, {
        "transform_mode": TRANSFORM_MODE_ACES_LIKE_PWL,
        "pwl_segment_index": approximation.segment_index(shift),
        "pwl_error_at_x": error_at_x,
        "gamut_mapped_pixel_count": diagnostics.gamut_mapped_pixel_count,
        "hard_clipped_channel_count": diagnostics.hard_clipped_channel_count,
        "mean_hue_drift_degrees": diagnostics.mean_hue_drift_degrees,
        "max_hue_drift_degrees": diagnostics.max_hue_drift_degrees,
        "mean_chroma_delta": diagnostics.mean_chroma_delta,
        "max_abs_chroma_delta": diagnostics.max_abs_chroma_delta,
        "mean_luminance_delta": diagnostics.mean_luminance_delta,
        "max_abs_luminance_delta": diagnostics.max_abs_luminance_delta,
    }


def _materialize_global_real_affine_arguments(
    primitive_inputs: Mapping[str, Any],
    normalized: Mapping[str, Any],
) -> Tuple[Dict[str, Any], float, int]:
    variable_name = normalized["variable_name"]
    if variable_name not in primitive_inputs:
        raise ValueError(f"global real input is missing {variable_name!r}")
    shift = float(unwrap(primitive_inputs[variable_name]))
    if not math.isfinite(shift):
        raise ValueError("global real X must be finite")
    lower = normalized["effective_min"]
    upper = normalized["effective_max"]
    if shift < lower - 1e-9 or shift > upper + 1e-9:
        raise ValueError(
            f"global real X={shift} is outside [{lower}, {upper}]"
        )
    coefficients = normalized["coefficient_by_input"]
    materialized: MutableMapping[str, Any] = {}
    clipped_count = 0
    for name, raw_value in primitive_inputs.items():
        if name == variable_name:
            continue
        if name not in coefficients:
            materialized[name] = unwrap(raw_value)
            continue
        shifted = float(unwrap(raw_value)) + coefficients[name] * shift
        if shifted < 0.0 or shifted > 1.0:
            clipped_count += 1
        if normalized["bounds_mode"] == BOUNDS_MODE_STRICT and (
            shifted < -1e-7 or shifted > 1.0 + 1e-7
        ):
            raise ValueError("global real strict-mode input is outside [0, 1]")
        materialized[name] = min(max(shifted, 0.0), 1.0)
    return dict(materialized), shift, clipped_count


def materialize_global_real_arguments(
    primitive_inputs: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], float, int]:
    normalized = validate_global_real_config(config)
    if normalized["transform_mode"] == TRANSFORM_MODE_ACES_LIKE_PWL:
        materialized, shift, clipped_count, _diagnostics = materialize_global_real_details(
            primitive_inputs, normalized
        )
        return materialized, shift, clipped_count
    return _materialize_global_real_affine_arguments(primitive_inputs, normalized)


__all__ = [
    "BOUNDS_MODE_CLIP",
    "BOUNDS_MODE_STRICT",
    "BOUNDS_MODES",
    "GLOBAL_X_INPUT_NAME",
    "GLOBAL_X_SMT_NAME",
    "build_concolic_global_real_kwargs",
    "materialize_global_real_arguments",
    "solver_variable_bounds",
    "validate_global_real_config",
]
