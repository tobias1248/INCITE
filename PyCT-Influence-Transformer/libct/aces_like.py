"""Concrete ACES-like OKLCh transforms for global-real experiments.

This module intentionally implements a small, deterministic engineering
approximation rather than an ACES JMh output transform.  It is a concrete
reference oracle for later symbolic work: tone is adjusted in OKLab lightness,
then out-of-gamut colours are mapped by reducing OKLCh chroma at fixed
lightness and hue.
"""

from __future__ import annotations

import bisect
import heapq
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


ACES_LIKE_SHIFT_KINDS: Tuple[str, str] = (
    "aces-brightness",
    "aces-contrast",
)
ACES_LIKE_COLOR_SPACE = "OKLCh-sRGB"
ACES_LIKE_CURVE_VERSION = "oklch-logit-v1"
ACES_LIKE_GAMUT_MAPPER = "css-color-4-local-minde-v1"
ACES_LIKE_PWL_ERROR_METRIC = "sampled-max-abs-rgb"
ACES_LIKE_PWL_VALIDATOR_VERSION = "adaptive-31-point-v1"
DEFAULT_PWL_ERROR_TOLERANCE = 1.0 / 255.0
DEFAULT_PWL_MAX_SEGMENTS = 32
ACES_LIKE_TRANSFORM_ORDERS = ("brightness-contrast", "contrast-brightness")


class AcesLikeTransformError(ValueError):
    """Raised when a concrete ACES-like transform cannot be represented safely."""


@dataclass(frozen=True)
class AcesLikeDiagnostics:
    """Colour-fidelity information for one reference-transform invocation."""

    mean_hue_drift_degrees: float
    max_hue_drift_degrees: float
    mean_chroma_delta: float
    max_abs_chroma_delta: float
    mean_luminance_delta: float
    max_abs_luminance_delta: float
    gamut_mapped_pixel_count: int
    hard_clipped_channel_count: int


@dataclass(frozen=True)
class AcesLikeTransformResult:
    """The exact sRGB result and diagnostics for a single shared ``X``."""

    rgb: np.ndarray
    diagnostics: AcesLikeDiagnostics


@dataclass(frozen=True)
class PiecewiseLinearApproximation:
    """A shared-X, RGB-output piecewise-linear table.

    ``rgb_at_knots[k]`` is the exact reference image at ``knots[k]``.  The
    symbolic layer can use ``affine_for_segment`` to build one affine RGB
    expression per channel after it chooses a single segment for shared X.
    """

    knots: np.ndarray
    rgb_at_knots: np.ndarray
    max_abs_error: float
    error_tolerance: float

    def __post_init__(self) -> None:
        knots = np.asarray(self.knots, dtype=np.float64)
        values = np.asarray(self.rgb_at_knots, dtype=np.float64)
        if knots.ndim != 1 or knots.size < 2:
            raise AcesLikeTransformError(
                "PWL knots must be a one-dimensional array of length >= 2"
            )
        if values.ndim < 2 or values.shape[0] != knots.size or values.shape[-1] != 3:
            raise AcesLikeTransformError("PWL RGB table must have shape (knots, ..., 3)")
        if not np.all(np.isfinite(knots)) or not np.all(np.isfinite(values)):
            raise AcesLikeTransformError("PWL knots and RGB table must be finite")
        if not np.all(np.diff(knots) > 0.0):
            raise AcesLikeTransformError("PWL knots must be strictly increasing")
        if not math.isfinite(float(self.max_abs_error)) or self.max_abs_error < 0.0:
            raise AcesLikeTransformError("PWL max_abs_error must be finite and non-negative")
        if not math.isfinite(float(self.error_tolerance)) or self.error_tolerance <= 0.0:
            raise AcesLikeTransformError("PWL error_tolerance must be finite and positive")
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "rgb_at_knots", values)

    @property
    def segment_count(self) -> int:
        return int(self.knots.size - 1)

    def segment_index(self, x: float) -> int:
        value = _finite_scalar(x, "PWL X")
        if value < self.knots[0] or value > self.knots[-1]:
            raise AcesLikeTransformError(
                "PWL X={} is outside [{}, {}]".format(value, self.knots[0], self.knots[-1])
            )
        return min(
            int(np.searchsorted(self.knots, value, side="right") - 1),
            self.segment_count - 1,
        )

    def affine_for_segment(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        if not 0 <= int(index) < self.segment_count:
            raise AcesLikeTransformError("PWL segment index is outside the available range")
        lower = self.knots[index]
        upper = self.knots[index + 1]
        slope = (self.rgb_at_knots[index + 1] - self.rgb_at_knots[index]) / (upper - lower)
        intercept = self.rgb_at_knots[index] - slope * lower
        return slope, intercept

    def evaluate(self, x: float) -> np.ndarray:
        index = self.segment_index(x)
        slope, intercept = self.affine_for_segment(index)
        return slope * float(x) + intercept


# Ottosson's OKLab matrices, with rows representing output coordinates.
_LINEAR_RGB_TO_LMS = np.asarray(
    (
        (0.4122214708, 0.5363325363, 0.0514459929),
        (0.2119034982, 0.6806995451, 0.1073969566),
        (0.0883024619, 0.2817188376, 0.6299787005),
    ),
    dtype=np.float64,
)
_LMS_TO_OKLAB = np.asarray(
    (
        (0.2104542553, 0.7936177850, -0.0040720468),
        (1.9779984951, -2.4285922050, 0.4505937099),
        (0.0259040371, 0.7827717662, -0.8086757660),
    ),
    dtype=np.float64,
)
_OKLAB_TO_LMS = np.asarray(
    (
        (1.0, 0.3963377774, 0.2158037573),
        (1.0, -0.1055613458, -0.0638541728),
        (1.0, -0.0894841775, -1.2914855480),
    ),
    dtype=np.float64,
)
_LMS_TO_LINEAR_RGB = np.asarray(
    (
        (4.0767416621, -3.3077115913, 0.2309699292),
        (-1.2684380046, 2.6097574011, -0.3413193965),
        (-0.0041960863, -0.7034186147, 1.7076147010),
    ),
    dtype=np.float64,
)
_LUMINANCE = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
_BRIGHTNESS_LOGIT_STRENGTH = 2.0
_CONTRAST_LOG_SLOPE = math.log(2.0)
_GAMUT_JND = 0.02
_GAMUT_SEARCH_EPSILON = 0.0001
_GAMUT_BISECTION_STEPS = 32
_PWL_VALIDATION_FRACTIONS = tuple(index / 32.0 for index in range(1, 32))


def _finite_scalar(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AcesLikeTransformError("{} must be a finite scalar".format(name)) from exc
    if not math.isfinite(result):
        raise AcesLikeTransformError("{} must be finite".format(name))
    return result


def _as_rgb(value: np.ndarray, *, normalized_srgb: bool = False) -> np.ndarray:
    rgb = np.asarray(value, dtype=np.float64)
    if rgb.ndim < 1 or rgb.shape[-1] != 3:
        raise AcesLikeTransformError("RGB input must have shape (..., 3), got {}".format(rgb.shape))
    if not np.all(np.isfinite(rgb)):
        raise AcesLikeTransformError("RGB input must contain only finite values")
    if normalized_srgb and (np.any(rgb < 0.0) or np.any(rgb > 1.0)):
        raise AcesLikeTransformError("sRGB input must be normalized to [0, 1]")
    return rgb


def _matmul_last(values: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.matmul(values, matrix.T)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Decode finite sRGB values to linear RGB using the IEC transfer curve."""

    values = _as_rgb(rgb)
    result = np.empty_like(values, dtype=np.float64)
    low = values <= 0.04045
    result[low] = values[low] / 12.92
    high_values = values[~low]
    result[~low] = ((high_values + 0.055) / 1.055) ** 2.4
    return result


def linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    """Encode finite linear RGB values with the IEC sRGB transfer curve."""

    values = _as_rgb(linear_rgb)
    result = np.empty_like(values, dtype=np.float64)
    low = values <= 0.0031308
    result[low] = 12.92 * values[low]
    high_values = values[~low]
    result[~low] = 1.055 * np.power(high_values, 1.0 / 2.4) - 0.055
    return result


def linear_rgb_to_oklab(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert finite linear sRGB values to OKLab."""

    values = _as_rgb(linear_rgb)
    lms = _matmul_last(values, _LINEAR_RGB_TO_LMS)
    return _matmul_last(np.cbrt(lms), _LMS_TO_OKLAB)


def oklab_to_linear_rgb(oklab: np.ndarray) -> np.ndarray:
    """Convert finite OKLab values to linear sRGB (which may be out of gamut)."""

    values = _as_rgb(oklab)
    lms = _matmul_last(values, _OKLAB_TO_LMS)
    return _matmul_last(lms * lms * lms, _LMS_TO_LINEAR_RGB)


def oklab_to_oklch(oklab: np.ndarray) -> np.ndarray:
    """Convert OKLab to polar OKLCh, using radians for hue."""

    values = _as_rgb(oklab)
    chroma = np.hypot(values[..., 1], values[..., 2])
    hue = np.where(chroma > 1e-15, np.arctan2(values[..., 2], values[..., 1]), 0.0)
    return np.stack((values[..., 0], chroma, hue), axis=-1)


def oklch_to_oklab(oklch: np.ndarray) -> np.ndarray:
    """Convert polar OKLCh (radian hue) to OKLab."""

    values = _as_rgb(oklch)
    if np.any(values[..., 1] < 0.0):
        raise AcesLikeTransformError("OKLCh chroma must be non-negative")
    return np.stack(
        (
            values[..., 0],
            values[..., 1] * np.cos(values[..., 2]),
            values[..., 1] * np.sin(values[..., 2]),
        ),
        axis=-1,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0.0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _logit_tone_curve(
    lightness: np.ndarray, *, offset: float = 0.0, slope: float = 1.0
) -> np.ndarray:
    """A bounded display curve with exact endpoints and a toe/shoulder."""

    result = np.empty_like(lightness, dtype=np.float64)
    low = lightness <= 0.0
    high = lightness >= 1.0
    middle = ~(low | high)
    result[low] = 0.0
    result[high] = 1.0
    if np.any(middle):
        values = lightness[middle]
        result[middle] = _sigmoid(slope * (np.log(values) - np.log1p(-values)) + offset)
    return result


def _transformed_lightness(lightness: np.ndarray, x: float, kind: str) -> np.ndarray:
    if x == 0.0:
        return np.array(lightness, dtype=np.float64, copy=True)
    if kind == "aces-brightness":
        return _logit_tone_curve(lightness, offset=x * _BRIGHTNESS_LOGIT_STRENGTH)
    if kind == "aces-contrast":
        try:
            slope = math.exp(x * _CONTRAST_LOG_SLOPE)
        except OverflowError as exc:
            raise AcesLikeTransformError("ACES-like contrast X is too large") from exc
        return _logit_tone_curve(lightness, slope=slope)
    raise AcesLikeTransformError(
        "unknown ACES-like shift kind {!r}; expected one of {}".format(
            kind, ACES_LIKE_SHIFT_KINDS
        )
    )


def _in_linear_srgb_gamut(linear_rgb: np.ndarray) -> np.ndarray:
    return np.all((linear_rgb >= 0.0) & (linear_rgb <= 1.0), axis=-1)


def _constant_lightness_hue_gamut_map(
    oklch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Map to sRGB with constant-L/h chroma reduction and local MINDE.

    A strict in-gamut bisection can over-compress colours near concave gamut
    cusps: an infinitesimal channel overshoot may jump to a much lower-chroma
    intersection. CSS Color 4 avoids that discontinuity by accepting the
    clipped colour when its deltaEOK from the candidate is below one JND.
    """

    initial_linear = oklab_to_linear_rgb(oklch_to_oklab(oklch))
    inside = _in_linear_srgb_gamut(initial_linear)
    needs_mapping = ~inside
    if not np.any(needs_mapping):
        return initial_linear, needs_mapping, 0

    initial_clipped = np.clip(initial_linear, 0.0, 1.0)
    current_oklab = oklch_to_oklab(oklch)
    clipped_oklab = linear_rgb_to_oklab(initial_clipped)
    initial_delta = np.linalg.norm(clipped_oklab - current_oklab, axis=-1)
    local_clip = needs_mapping & (initial_delta < _GAMUT_JND)

    mapped_linear = np.array(initial_linear, copy=True)
    mapped_linear[local_clip] = initial_clipped[local_clip]
    clipped_channels = np.zeros(initial_linear.shape, dtype=bool)
    clipped_channels[local_clip] = (
        (initial_linear[local_clip] < 0.0)
        | (initial_linear[local_clip] > 1.0)
    )

    searching = needs_mapping & ~local_clip
    if not np.any(searching):
        return mapped_linear, needs_mapping, int(np.count_nonzero(clipped_channels))

    chroma = oklch[..., 1]
    lower = np.zeros_like(chroma)
    upper = np.array(chroma, copy=True)
    min_in_gamut = np.ones_like(chroma, dtype=bool)
    for _ in range(_GAMUT_BISECTION_STEPS):
        active = searching & ((upper - lower) > _GAMUT_SEARCH_EPSILON)
        if not np.any(active):
            break
        middle = (lower + upper) * 0.5
        candidate = np.array(oklch, copy=True)
        candidate[..., 1] = middle
        candidate_linear = oklab_to_linear_rgb(oklch_to_oklab(candidate))
        candidate_inside = _in_linear_srgb_gamut(candidate_linear)
        advance_inside = active & min_in_gamut & candidate_inside
        lower = np.where(advance_inside, middle, lower)
        mapped_linear[advance_inside] = candidate_linear[advance_inside]
        clipped_channels[advance_inside] = False

        compare_clip = active & ~advance_inside
        if not np.any(compare_clip):
            continue
        candidate_clipped = np.clip(candidate_linear, 0.0, 1.0)
        candidate_oklab = oklch_to_oklab(candidate)
        candidate_clipped_oklab = linear_rgb_to_oklab(candidate_clipped)
        delta = np.linalg.norm(candidate_clipped_oklab - candidate_oklab, axis=-1)
        below_jnd = compare_clip & (delta < _GAMUT_JND)
        close_to_jnd = below_jnd & ((_GAMUT_JND - delta) < _GAMUT_SEARCH_EPSILON)

        mapped_linear[compare_clip] = candidate_clipped[compare_clip]
        clipped_channels[compare_clip] = (
            (candidate_linear[compare_clip] < 0.0)
            | (candidate_linear[compare_clip] > 1.0)
        )
        searching = searching & ~close_to_jnd
        lower = np.where(below_jnd & ~close_to_jnd, middle, lower)
        min_in_gamut = np.where(below_jnd, False, min_in_gamut)
        upper = np.where(compare_clip & ~below_jnd, middle, upper)

    return mapped_linear, needs_mapping, int(np.count_nonzero(clipped_channels))


def _hue_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(first - second), np.cos(first - second)))


def _diagnostics(
    before_srgb: np.ndarray,
    after_srgb: np.ndarray,
    gamut_mapped: np.ndarray,
    hard_clipped_channels: int,
) -> AcesLikeDiagnostics:
    before_lch = oklab_to_oklch(linear_rgb_to_oklab(srgb_to_linear(before_srgb)))
    after_linear = srgb_to_linear(after_srgb)
    after_lch = oklab_to_oklch(linear_rgb_to_oklab(after_linear))
    hue_delta = _hue_distance(before_lch[..., 2], after_lch[..., 2])
    chroma_delta = np.abs(after_lch[..., 1] - before_lch[..., 1])
    before_luminance = np.sum(
        srgb_to_linear(before_srgb) * _LUMINANCE, axis=-1
    )
    after_luminance = np.sum(after_linear * _LUMINANCE, axis=-1)
    luminance_delta = np.abs(after_luminance - before_luminance)
    return AcesLikeDiagnostics(
        mean_hue_drift_degrees=float(np.degrees(np.mean(hue_delta))),
        max_hue_drift_degrees=float(np.degrees(np.max(hue_delta))),
        mean_chroma_delta=float(np.mean(chroma_delta)),
        max_abs_chroma_delta=float(np.max(chroma_delta)),
        mean_luminance_delta=float(np.mean(luminance_delta)),
        max_abs_luminance_delta=float(np.max(luminance_delta)),
        gamut_mapped_pixel_count=int(np.count_nonzero(gamut_mapped)),
        hard_clipped_channel_count=int(hard_clipped_channels),
    )


def apply_aces_like_transform(
    rgb: np.ndarray, x: float, *, kind: str
) -> AcesLikeTransformResult:
    """Apply a shared ACES-like display brightness or contrast control.

    ``rgb`` must be a finite normalized sRGB array with final dimension three.
    The result is always finite normalized sRGB.  Invalid inputs and any
    internal non-finite result fail closed with :class:`AcesLikeTransformError`.
    """

    source = _as_rgb(rgb, normalized_srgb=True)
    shift = _finite_scalar(x, "ACES-like X")
    if kind not in ACES_LIKE_SHIFT_KINDS:
        raise AcesLikeTransformError(
            "unknown ACES-like shift kind {!r}; expected one of {}".format(
                kind, ACES_LIKE_SHIFT_KINDS
            )
        )
    if shift == 0.0:
        # This is a contract-level identity, not just a sufficiently-close
        # round trip through the finite-precision OKLab matrices.
        return AcesLikeTransformResult(
            rgb=np.array(source, dtype=np.float64, copy=True),
            diagnostics=AcesLikeDiagnostics(
                mean_hue_drift_degrees=0.0,
                max_hue_drift_degrees=0.0,
                mean_chroma_delta=0.0,
                max_abs_chroma_delta=0.0,
                mean_luminance_delta=0.0,
                max_abs_luminance_delta=0.0,
                gamut_mapped_pixel_count=0,
                hard_clipped_channel_count=0,
            ),
        )

    source_lch = oklab_to_oklch(linear_rgb_to_oklab(srgb_to_linear(source)))
    transformed_lch = np.array(source_lch, copy=True)
    transformed_lch[..., 0] = _transformed_lightness(source_lch[..., 0], shift, kind)
    mapped_linear, gamut_mapped, hard_clipped_channels = (
        _constant_lightness_hue_gamut_map(transformed_lch)
    )
    # The bisection result can differ from a boundary by machine epsilon.  This
    # final clamp is intentionally a numerical fallback, not gamut mapping.
    result = np.clip(linear_to_srgb(mapped_linear), 0.0, 1.0)
    if not np.all(np.isfinite(result)):
        raise AcesLikeTransformError("ACES-like transform produced non-finite RGB output")
    return AcesLikeTransformResult(
        rgb=result,
        diagnostics=_diagnostics(source, result, gamut_mapped, hard_clipped_channels),
    )


def apply_aces_like_joint_transform(
    rgb: np.ndarray, brightness: float, contrast: float,
    *, order: str = "brightness-contrast",
) -> AcesLikeTransformResult:
    """Apply both exact controls sequentially, with gamut mapping at each step.

    Parameters are absolute within this invocation. A hybrid invocation on a
    DE seed adds these transforms to the already transformed seed.
    """
    if order not in ACES_LIKE_TRANSFORM_ORDERS:
        raise AcesLikeTransformError("unknown ACES-like transform order: {!r}".format(order))
    values = {
        "brightness": _finite_scalar(brightness, "brightness"),
        "contrast": _finite_scalar(contrast, "contrast"),
    }
    source = _as_rgb(rgb, normalized_srgb=True)
    result = source
    gamut_mask_count = 0
    clipped_count = 0
    for axis in order.split("-"):
        transformed = apply_aces_like_transform(result, values[axis], kind="aces-" + axis)
        result = transformed.rgb
        # Counts describe mapping operations across both passes; the same
        # pixel/channel can be counted twice. Colour drift is source-to-final.
        gamut_mask_count += transformed.diagnostics.gamut_mapped_pixel_count
        clipped_count += transformed.diagnostics.hard_clipped_channel_count
    diagnostics = _diagnostics(source, result, np.zeros(source.shape[:-1], dtype=bool), 0)
    diagnostics = AcesLikeDiagnostics(
        **{**diagnostics.__dict__, "gamut_mapped_pixel_count": gamut_mask_count,
           "hard_clipped_channel_count": clipped_count}
    )
    return AcesLikeTransformResult(rgb=result, diagnostics=diagnostics)


def _pwl_interval_error(
    rgb: np.ndarray,
    kind: str,
    lower: float,
    upper: float,
    lower_rgb: np.ndarray,
    upper_rgb: np.ndarray,
) -> Tuple[float, float]:
    """Return the largest sampled interpolation error and its X location."""

    best_error = -1.0
    best_x = (lower + upper) * 0.5
    for fraction in _PWL_VALIDATION_FRACTIONS:
        x = lower + (upper - lower) * fraction
        exact = apply_aces_like_transform(rgb, x, kind=kind).rgb
        approximation = lower_rgb + (upper_rgb - lower_rgb) * fraction
        error = float(np.max(np.abs(exact - approximation)))
        if error > best_error:
            best_error = error
            best_x = x
    return best_error, best_x


def build_adaptive_pwl_approximation(
    rgb: np.ndarray,
    *,
    kind: str,
    x_min: float,
    x_max: float,
    max_segments: int = DEFAULT_PWL_MAX_SEGMENTS,
    error_tolerance: float = DEFAULT_PWL_ERROR_TOLERANCE,
) -> PiecewiseLinearApproximation:
    """Build a bounded shared-knot PWL approximation of the exact transform.

    The validation samples are a deterministic 31-point grid in every final
    interval. max_abs_error is therefore a sampled maximum, not a formal
    analytic supremum. Runtime materialization independently checks the exact
    error at every solver candidate and fails closed if it exceeds tolerance.
    """

    source = _as_rgb(rgb, normalized_srgb=True)
    lower = _finite_scalar(x_min, "PWL x_min")
    upper = _finite_scalar(x_max, "PWL x_max")
    tolerance = _finite_scalar(error_tolerance, "PWL error_tolerance")
    if kind not in ACES_LIKE_SHIFT_KINDS:
        raise AcesLikeTransformError("PWL kind must be an ACES-like shift kind")
    if lower >= upper:
        raise AcesLikeTransformError("PWL x_min must be less than x_max")
    if not lower <= 0.0 <= upper:
        raise AcesLikeTransformError("PWL X interval must include X=0")
    if tolerance <= 0.0:
        raise AcesLikeTransformError("PWL error_tolerance must be positive")
    if isinstance(max_segments, bool) or int(max_segments) != max_segments or max_segments < 1:
        raise AcesLikeTransformError("PWL max_segments must be an integer >= 1")
    segment_cap = int(max_segments)

    # Pinning X=0 makes the symbolic identity exact, rather than merely close.
    knots = sorted({lower, 0.0, upper})
    minimum_segments = len(knots) - 1
    if segment_cap < minimum_segments:
        raise AcesLikeTransformError(
            "PWL max_segments={} cannot represent the required {} initial segments".format(
                segment_cap, minimum_segments
            )
        )
    values = {
        knot: apply_aces_like_transform(source, knot, kind=kind).rgb for knot in knots
    }
    candidates = []
    for left, right in zip(knots[:-1], knots[1:]):
        error, split_x = _pwl_interval_error(
            source, kind, left, right, values[left], values[right]
        )
        heapq.heappush(candidates, (-error, left, right, split_x))

    max_error = -candidates[0][0]
    while max_error > tolerance:
        if len(knots) - 1 >= segment_cap:
            raise AcesLikeTransformError(
                "PWL approximation requires more than {} segments to meet "
                "max error {} (observed {})".format(
                    segment_cap, tolerance, max_error
                )
            )
        _negative_error, left, right, split_x = heapq.heappop(candidates)
        # Validation-grid candidates are interior by construction.
        values[split_x] = apply_aces_like_transform(source, split_x, kind=kind).rgb
        bisect.insort(knots, split_x)
        for child_left, child_right in ((left, split_x), (split_x, right)):
            error, child_split_x = _pwl_interval_error(
                source,
                kind,
                child_left,
                child_right,
                values[child_left],
                values[child_right],
            )
            heapq.heappush(
                candidates,
                (-error, child_left, child_right, child_split_x),
            )
        max_error = -candidates[0][0]

    ordered_knots = np.asarray(knots, dtype=np.float64)
    table = np.stack([values[knot] for knot in knots], axis=0)
    return PiecewiseLinearApproximation(
        knots=ordered_knots,
        rgb_at_knots=table,
        max_abs_error=float(max_error),
        error_tolerance=tolerance,
    )


__all__ = [
    "ACES_LIKE_COLOR_SPACE",
    "ACES_LIKE_CURVE_VERSION",
    "ACES_LIKE_GAMUT_MAPPER",
    "ACES_LIKE_PWL_ERROR_METRIC",
    "ACES_LIKE_PWL_VALIDATOR_VERSION",
    "ACES_LIKE_SHIFT_KINDS",
    "ACES_LIKE_TRANSFORM_ORDERS",
    "AcesLikeDiagnostics",
    "AcesLikeTransformError",
    "AcesLikeTransformResult",
    "DEFAULT_PWL_ERROR_TOLERANCE",
    "DEFAULT_PWL_MAX_SEGMENTS",
    "PiecewiseLinearApproximation",
    "apply_aces_like_transform",
    "apply_aces_like_joint_transform",
    "build_adaptive_pwl_approximation",
    "linear_rgb_to_oklab",
    "linear_to_srgb",
    "oklab_to_linear_rgb",
    "oklab_to_oklch",
    "oklch_to_oklab",
    "srgb_to_linear",
]
