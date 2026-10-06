"""Adaptive, conforming triangular RGB approximation of two ACES controls.

Each rectangular cell uses the bottom-left to top-right diagonal. Refinement
inserts a complete row or column, so adjacent planes share exact edge values.
Validation is sampled, not a mathematical uniform-error certificate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from libct.aces_like import (
    AcesLikeTransformError, DEFAULT_PWL_ERROR_TOLERANCE,
    apply_aces_like_joint_transform,
)

DEFAULT_PWL_MAX_TRIANGLES = 128
PWL_2D_VALIDATOR_VERSION = "adaptive-barycentric-eighth-v1"


@dataclass(frozen=True)
class PiecewisePlanarApproximation:
    b_knots: np.ndarray
    c_knots: np.ndarray
    rgb_at_knots: np.ndarray
    max_abs_error: float
    error_tolerance: float

    def __post_init__(self) -> None:
        b, c = np.asarray(self.b_knots, dtype=float), np.asarray(self.c_knots, dtype=float)
        values = np.asarray(self.rgb_at_knots, dtype=float)
        for knots in (b, c):
            if (knots.ndim != 1 or len(knots) < 2 or not np.isfinite(knots).all()
                    or not np.all(np.diff(knots) > 0)):
                raise AcesLikeTransformError("2D PWL knots must be finite and strictly increasing")
        if (values.ndim < 3 or values.shape[:2] != (len(b), len(c))
                or values.shape[-1] != 3 or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values > 1)):
            raise AcesLikeTransformError("2D PWL RGB table has invalid shape or values")
        if (not math.isfinite(self.max_abs_error) or self.max_abs_error < 0
                or not math.isfinite(self.error_tolerance) or self.error_tolerance <= 0):
            raise AcesLikeTransformError("2D PWL error metadata must be finite and valid")
        object.__setattr__(self, "b_knots", b)
        object.__setattr__(self, "c_knots", c)
        object.__setattr__(self, "rgb_at_knots", values)

    @property
    def regions(self):
        return tuple((i, j, half) for i in range(len(self.b_knots) - 1)
                     for j in range(len(self.c_knots) - 1) for half in (0, 1))

    @property
    def triangle_count(self) -> int:
        return 2 * (len(self.b_knots) - 1) * (len(self.c_knots) - 1)

    def region_index(self, brightness: float, contrast: float) -> int:
        indices = []
        for value, knots in ((brightness, self.b_knots), (contrast, self.c_knots)):
            if not math.isfinite(value) or value < knots[0] or value > knots[-1]:
                raise AcesLikeTransformError("2D PWL parameter outside configured bounds")
            indices.append(min(int(np.searchsorted(knots, value, side="right") - 1), len(knots) - 2))
        i, j = indices
        u = (brightness - self.b_knots[i]) / (self.b_knots[i + 1] - self.b_knots[i])
        v = (contrast - self.c_knots[j]) / (self.c_knots[j + 1] - self.c_knots[j])
        return 2 * (i * (len(self.c_knots) - 1) + j) + int(v > u)

    def affine_for_region(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < self.triangle_count:
            raise AcesLikeTransformError("2D PWL region index outside available range")
        i, j, half = self.regions[index]
        table = self.rgb_at_knots
        if half == 0:
            a = (table[i + 1, j] - table[i, j]) / (self.b_knots[i + 1] - self.b_knots[i])
            b = (table[i + 1, j + 1] - table[i + 1, j]) / (self.c_knots[j + 1] - self.c_knots[j])
        else:
            a = (table[i + 1, j + 1] - table[i, j + 1]) / (self.b_knots[i + 1] - self.b_knots[i])
            b = (table[i, j + 1] - table[i, j]) / (self.c_knots[j + 1] - self.c_knots[j])
        intercept = table[i, j] - a * self.b_knots[i] - b * self.c_knots[j]
        return a, b, intercept

    def evaluate(self, brightness: float, contrast: float) -> np.ndarray:
        a, b, intercept = self.affine_for_region(self.region_index(brightness, contrast))
        return a * brightness + b * contrast + intercept


def build_adaptive_pwl_2d(
    rgb: np.ndarray, *, lower: float, upper: float,
    order: str = "brightness-contrast", max_triangles: int = DEFAULT_PWL_MAX_TRIANGLES,
    error_tolerance: float = DEFAULT_PWL_ERROR_TOLERANCE,
) -> PiecewisePlanarApproximation:
    if (not math.isfinite(lower) or not math.isfinite(upper)
            or lower >= upper or not lower <= 0 <= upper):
        raise AcesLikeTransformError("2D PWL bounds must be finite, include zero, and satisfy lower < upper")
    if isinstance(max_triangles, bool) or not isinstance(max_triangles, int) or max_triangles < 2:
        raise AcesLikeTransformError("2D PWL max_triangles must be an integer >= 2")
    if not math.isfinite(error_tolerance) or error_tolerance <= 0:
        raise AcesLikeTransformError("2D PWL error_tolerance must be finite and positive")
    # Validates order, image, and the exact identity before allocating a grid.
    apply_aces_like_joint_transform(rgb, 0.0, 0.0, order=order)
    b_knots = sorted(set((float(lower), 0.0, float(upper))))
    c_knots = b_knots.copy()
    cache = {}
    cell_errors = {}

    def exact(b, c):
        key = (float(b), float(c))
        if key not in cache:
            cache[key] = apply_aces_like_joint_transform(rgb, *key, order=order).rgb
        return cache[key]

    while True:
        count = 2 * (len(b_knots) - 1) * (len(c_knots) - 1)
        if count > max_triangles:
            raise AcesLikeTransformError("2D PWL origin grid requires more than {} triangles".format(max_triangles))
        table = np.stack([np.stack([exact(b, c) for c in c_knots]) for b in b_knots])
        approximation = PiecewisePlanarApproximation(
            np.array(b_knots), np.array(c_knots), table, 0.0, error_tolerance,
        )
        worst = (0.0, 0, 0)
        for i in range(len(b_knots) - 1):
            for j in range(len(c_knots) - 1):
                lo_b, hi_b, lo_c, hi_c = b_knots[i], b_knots[i + 1], c_knots[j], c_knots[j + 1]
                key = (lo_b, hi_b, lo_c, hi_c)
                if key not in cell_errors:
                    error = 0.0
                    for half in (0, 1):
                        vertices = ((0, 0), (1, 0), (1, 1)) if half == 0 else ((0, 0), (1, 1), (0, 1))
                        weights = [(a / 8, b / 8) for a in range(9) for b in range(9 - a)]
                        weights.append((1 / 3, 1 / 3))
                        for w1, w2 in weights:
                            u = (1 - w1 - w2) * vertices[0][0] + w1 * vertices[1][0] + w2 * vertices[2][0]
                            v = (1 - w1 - w2) * vertices[0][1] + w1 * vertices[1][1] + w2 * vertices[2][1]
                            b, c = lo_b + u * (hi_b - lo_b), lo_c + v * (hi_c - lo_c)
                            error = max(error, float(np.max(np.abs(exact(b, c) - approximation.evaluate(b, c)))))
                    cell_errors[key] = error
                if cell_errors[key] > worst[0]:
                    worst = (cell_errors[key], i, j)
        error, i, j = worst
        if error <= error_tolerance:
            return PiecewisePlanarApproximation(
                approximation.b_knots, approximation.c_knots, table, error, error_tolerance,
            )
        # Insert a full row/column: avoid hanging vertices and discontinuous
        # shared edges. Choose the widest dimension that fits the hard cap.
        choices = [
            (b_knots[i + 1] - b_knots[i], "b", 2 * len(b_knots) * (len(c_knots) - 1)),
            (c_knots[j + 1] - c_knots[j], "c", 2 * (len(b_knots) - 1) * len(c_knots)),
        ]
        choices = [choice for choice in choices if choice[2] <= max_triangles]
        if not choices:
            raise AcesLikeTransformError(
                "2D PWL requires more than {} triangles to meet max error {} (observed {})".format(
                    max_triangles, error_tolerance, error,
                )
            )
        axis = max(choices, key=lambda choice: choice[0])[1]
        knots, index = (b_knots, i) if axis == "b" else (c_knots, j)
        midpoint = (knots[index] + knots[index + 1]) / 2
        if midpoint == knots[index] or midpoint == knots[index + 1]:
            raise AcesLikeTransformError("2D PWL refinement exhausted floating point precision")
        knots.insert(index + 1, midpoint)
