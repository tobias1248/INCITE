from __future__ import annotations

import math
from typing import Any, Dict, Mapping, MutableMapping, Tuple

from libct.utils import ConcolicObject, unwrap


GLOBAL_X_INPUT_NAME = "__pyct_global_x"
GLOBAL_X_SMT_NAME = f"{GLOBAL_X_INPUT_NAME}_VAR"
BOUNDS_MODE_CLIP = "clip"
BOUNDS_MODE_STRICT = "strict"
BOUNDS_MODES = (BOUNDS_MODE_CLIP, BOUNDS_MODE_STRICT)


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

    raw_coefficients = config.get("coefficient_by_input")
    raw_signs = config.get("sign_by_input")
    if raw_coefficients is not None and raw_signs is not None:
        raise ValueError("global real config cannot define both coefficient_by_input and sign_by_input")

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

    normalized = dict(config)
    normalized.pop("sign_by_input", None)
    normalized.update(
        {
            "variable_name": variable_name,
            "bounds_mode": bounds_mode,
            "effective_min": lower,
            "effective_max": upper,
            "coefficient_by_input": coefficients,
        }
    )
    return normalized

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


def materialize_global_real_arguments(
    primitive_inputs: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], float, int]:
    normalized = validate_global_real_config(config)
    variable_name = normalized["variable_name"]
    if variable_name not in primitive_inputs:
        raise ValueError(f"global real input is missing {variable_name!r}")
    shift = float(unwrap(primitive_inputs[variable_name]))
    if not math.isfinite(shift):
        raise ValueError("global real X must be finite")

    lower = normalized["effective_min"]
    upper = normalized["effective_max"]
    if shift < lower - 1e-9 or shift > upper + 1e-9:
        raise ValueError(f"global real X={shift} is outside [{lower}, {upper}]")

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
