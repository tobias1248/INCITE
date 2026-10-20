"""Ordered, independent photo controls applied afresh to the source image."""
from __future__ import annotations

import math
from dataclasses import replace
from collections.abc import Mapping

import numpy as np

from libct.aces_like import (
    AcesLikeTransformError, AcesLikeTransformResult, _diagnostics,
    apply_aces_like_transform, validate_control_range,
)

CONTROL_ORDER = ("exposure", "contrast", "highlights", "shadows", "saturation", "vibrance", "curves")
MULTI_CONTROL_VERSION = "ordered-photo-controls-v1"
MULTI_SHIFT_KIND = "aces-multi"
MULTI_TRANSFORM_MODE = "aces-like-pwl-nd"


def normalize_controls(controls):
    if (not isinstance(controls, (list, tuple)) or not 2 <= len(controls) <= 3
            or any(not isinstance(c, str) or c not in CONTROL_ORDER for c in controls)
            or len(set(controls)) != len(controls)):
        raise AcesLikeTransformError("Choose 2 or 3 unique ACES controls from {}".format(CONTROL_ORDER))
    return tuple(c for c in CONTROL_ORDER if c in controls)


def normalize_bounds(controls, bounds):
    ordered = normalize_controls(controls)
    if isinstance(bounds, Mapping):
        if set(bounds) != set(controls):
            raise AcesLikeTransformError("Bounds must name every selected control exactly once")
        pairs = bounds
    else:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != len(controls):
            raise AcesLikeTransformError("Bounds must contain one pair per control")
        pairs = dict(zip(controls, bounds))
    result = []
    for control in ordered:
        pair = pairs[control]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise AcesLikeTransformError("Each control requires lower and upper bounds")
        if any(isinstance(v, (bool, str, bytes)) for v in pair):
            raise AcesLikeTransformError("Control bounds must be numeric")
        lo, hi = map(float, pair)
        if not all(math.isfinite(v) for v in (lo, hi)) or not lo < hi or not lo <= 0 <= hi:
            raise AcesLikeTransformError("Control bounds must be finite, ordered and include zero")
        validate_control_range("aces-" + control, lo, hi)
        result.append((lo, hi))
    return tuple(result)


def parameter_name(control):
    return "__pyct_aces_" + control


def apply_aces_like_multi_transform(rgb, values, *, controls):
    ordered = normalize_controls(controls)
    if isinstance(values, Mapping):
        if set(values) != set(controls):
            raise AcesLikeTransformError("Values must name every selected control")
        params = dict(values)
    else:
        if len(values) != len(controls):
            raise AcesLikeTransformError("Values must match selected controls")
        params = dict(zip(controls, values))
    for control, value in params.items():
        if isinstance(value, (bool, str, bytes)) or not math.isfinite(float(value)):
            raise AcesLikeTransformError("Control values must be finite numbers")
        validate_control_range("aces-" + control, float(value), float(value))
    source = np.asarray(rgb, dtype=np.float64)
    result = source
    gamut_count = clipped_count = 0
    for control in ordered:
        step = apply_aces_like_transform(result, float(params[control]), kind="aces-" + control)
        result = step.rgb
        gamut_count += step.diagnostics.gamut_mapped_pixel_count
        clipped_count += step.diagnostics.hard_clipped_channel_count
    diagnostics = _diagnostics(source, result, np.zeros(source.shape[:-1], dtype=bool), 0)
    return AcesLikeTransformResult(result, replace(
        diagnostics, gamut_mapped_pixel_count=gamut_count,
        hard_clipped_channel_count=clipped_count,
    ))
