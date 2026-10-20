"""Conforming 2/3-D simplicial PWL with a hard region budget.

Validation uses a barycentric quarter grid and centroid in each simplex.
The measured error is sampled, not a uniform mathematical certificate.
"""
from __future__ import annotations

import itertools
import math
import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np

from libct.aces_like import AcesLikeTransformError
from libct.aces_like_multi import (
    apply_aces_like_multi_transform, normalize_bounds, normalize_controls,
)

DEFAULT_PWL_MAX_REGIONS = 512
PWL_ND_VALIDATOR_VERSION = "conforming-simplex-quarter-v1"


class PWLRegionBudgetError(AcesLikeTransformError):
    """All configured region caps were exhausted before meeting tolerance."""

    def __init__(self, message, *, budget, region_count, max_abs_error, build_stats):
        super().__init__(message)
        self.budget = budget
        self.region_count = region_count
        self.max_abs_error = max_abs_error
        self.build_stats = build_stats


class PWLBuildTimeout(AcesLikeTransformError):
    """The monotonic deadline expired while constructing or validating a mesh."""

    def __init__(self, message, *, budget, region_count, max_abs_error, build_stats):
        super().__init__(message)
        self.budget = budget
        self.region_count = region_count
        self.max_abs_error = max_abs_error
        self.build_stats = build_stats


def normalize_region_budgets(max_regions, region_budgets=None):
    if type(max_regions) is not int or max_regions < 1:
        raise AcesLikeTransformError("PWL max_regions must be a positive integer")
    if region_budgets is None:
        return (max_regions,)
    if not isinstance(region_budgets, (list, tuple)) or not region_budgets:
        raise AcesLikeTransformError("PWL region_budgets must be a non-empty increasing sequence")
    budgets = tuple(region_budgets)
    if (any(type(value) is not int or value < 1 for value in budgets)
            or any(a >= b for a, b in zip(budgets, budgets[1:]))
            or budgets[-1] != max_regions):
        raise AcesLikeTransformError(
            "PWL region_budgets must be positive increasing integers ending at max_regions")
    return budgets


def mesh_fingerprint(rgb, config):
    """Bind the recorded mesh receipt to the source and approximation settings."""
    source = np.ascontiguousarray(rgb, dtype=np.float64)
    names = ("controls", "control_bounds", "control_order", "control_version", "pwl_nd_knots",
             "pwl_region_count", "pwl_max_regions", "pwl_max_abs_error", "pwl_error_tolerance",
             "pwl_validator_version")
    metadata = json.dumps({name: config[name] for name in names}, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(repr(source.shape).encode() + source.tobytes())
    digest.update(metadata)
    return digest.hexdigest()


@dataclass(frozen=True)
class PiecewiseSimplexApproximation:
    controls: tuple
    knots: tuple
    rgb_at_knots: np.ndarray
    max_abs_error: float
    error_tolerance: float

    @property
    def regions(self):
        return tuple((cell, permutation)
                     for cell in itertools.product(*(range(len(k) - 1) for k in self.knots))
                     for permutation in itertools.permutations(range(len(self.controls))))

    @property
    def region_count(self):
        return math.factorial(len(self.controls)) * math.prod(len(k) - 1 for k in self.knots)

    def region_index(self, *params):
        if len(params) != len(self.controls):
            raise AcesLikeTransformError("PWL parameter dimension mismatch")
        cells, units = [], []
        ordinal = 0
        for value, knots in zip(params, self.knots):
            if not math.isfinite(value) or not knots[0] <= value <= knots[-1]:
                raise AcesLikeTransformError("PWL parameter outside configured bounds")
            cell = min(int(np.searchsorted(knots, value, side="right")) - 1, len(knots) - 2)
            cells.append(cell)
            ordinal = ordinal * (len(knots) - 1) + cell
            units.append((value - knots[cell]) / (knots[cell + 1] - knots[cell]))
        permutation = tuple(sorted(range(len(units)), key=lambda i: (-units[i], i)))
        permutations = tuple(itertools.permutations(range(len(units))))
        return ordinal * len(permutations) + permutations.index(permutation)

    def affine_for_region(self, index):
        if type(index) is not int or not 0 <= index < self.region_count:
            raise AcesLikeTransformError("PWL region index outside available range")
        cell, permutation = self.regions[index]
        vertex = list(cell)
        previous = self.rgb_at_knots[tuple(vertex)]
        intercept = previous.copy()
        coefficients = np.empty((len(cell),) + previous.shape)
        for axis in permutation:
            vertex[axis] += 1
            current = self.rgb_at_knots[tuple(vertex)]
            slope = (current - previous) / (self.knots[axis][cell[axis] + 1] - self.knots[axis][cell[axis]])
            coefficients[axis] = slope
            intercept = intercept - slope * self.knots[axis][cell[axis]]
            previous = current
        return coefficients, intercept

    def evaluate(self, *params):
        coefficients, intercept = self.affine_for_region(self.region_index(*params))
        return intercept + np.tensordot(np.asarray(params), coefficients, axes=1)


def build_adaptive_pwl_nd(rgb, *, controls, bounds, max_regions=DEFAULT_PWL_MAX_REGIONS,
                          error_tolerance=1 / 255, region_budgets=None, deadline=None,
                          build_stats=None):
    """Refine one mesh and raise its cap only when the current cap is exhausted.

    ``deadline`` is an absolute ``time.monotonic()`` timestamp. Checks surround
    rendering and validation; a single renderer call cannot be interrupted.
    ``build_stats`` receives diagnostics on success and failure when supplied.
    """
    canonical = normalize_controls(controls)
    bounds = normalize_bounds(controls, bounds)
    controls = canonical
    budgets = normalize_region_budgets(max_regions, region_budgets)
    if not math.isfinite(error_tolerance) or error_tolerance <= 0:
        raise AcesLikeTransformError("PWL error tolerance must be finite and positive")
    if deadline is not None and (isinstance(deadline, bool) or not math.isfinite(deadline)):
        raise AcesLikeTransformError("PWL deadline must be a finite monotonic timestamp")
    started = time.monotonic()
    stats = {} if build_stats is None else build_stats
    stats.update(budget_history=[], final_budget=budgets[0], build_wall_time=0.0)
    dimension = len(controls)
    knots = [sorted(set((lo, 0.0, hi))) for lo, hi in bounds]
    samples, errors = {}, {}
    budget_index = 0
    count = math.factorial(dimension) * math.prod(len(k) - 1 for k in knots)
    error = None

    def record(reason):
        stats["budget_history"].append(dict(budget=budgets[budget_index], region_count=count,
                                            max_abs_error=error, reason=reason))

    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            record("timeout")
            raise PWLBuildTimeout("PWL build deadline exceeded", budget=budgets[budget_index],
                                  region_count=count, max_abs_error=error, build_stats=stats)

    def extend_budget(message):
        nonlocal budget_index
        record("budget_exhausted")
        if budget_index + 1 == len(budgets):
            raise PWLRegionBudgetError(message, budget=budgets[budget_index], region_count=count,
                                       max_abs_error=error, build_stats=stats)
        budget_index += 1
        stats["final_budget"] = budgets[budget_index]
        check_deadline()

    def exact(point):
        check_deadline()
        key = tuple(round(float(v), 15) for v in point)
        if key not in samples:
            samples[key] = apply_aces_like_multi_transform(source, key, controls=controls).rgb
            check_deadline()
        return samples[key]

    try:
        check_deadline()
        source = apply_aces_like_multi_transform(rgb, (0.0,) * dimension, controls=controls).rgb
        check_deadline()
        weights = [np.asarray(w, dtype=float) / 4
                   for w in itertools.product(range(5), repeat=dimension + 1) if sum(w) == 4]
        weights.append(np.full(dimension + 1, 1 / (dimension + 1)))
        while True:
            check_deadline()
            count = math.factorial(dimension) * math.prod(len(k) - 1 for k in knots)
            while count > budgets[budget_index]:
                extend_budget("PWL origin grid exceeds {} regions".format(budgets[budget_index]))
            shape = tuple(len(k) for k in knots)
            table = np.empty(shape + source.shape)
            for vertex in itertools.product(*(range(n) for n in shape)):
                table[vertex] = exact(tuple(knots[i][j] for i, j in enumerate(vertex)))
            approximation = PiecewiseSimplexApproximation(controls, tuple(np.asarray(k) for k in knots),
                                                          table, 0.0, error_tolerance)
            worst = (0.0, (0,) * dimension)
            for region, (cell, permutation) in enumerate(approximation.regions):
                check_deadline()
                key = (tuple((knots[i][j], knots[i][j + 1]) for i, j in enumerate(cell)), permutation)
                if key not in errors:
                    vertex = list(cell)
                    vertices = [np.asarray([knots[i][j] for i, j in enumerate(vertex)])]
                    for axis in permutation:
                        vertex[axis] += 1
                        vertices.append(np.asarray([knots[i][j] for i, j in enumerate(vertex)]))
                    coefficients, intercept = approximation.affine_for_region(region)
                    region_error = 0.0
                    for weight in weights:
                        check_deadline()
                        point = np.dot(weight, np.asarray(vertices))
                        interpolated = intercept + np.tensordot(point, coefficients, axes=1)
                        region_error = max(region_error, float(np.max(np.abs(exact(point) - interpolated))))
                    errors[key] = region_error
                if errors[key] > worst[0]:
                    worst = errors[key], cell
            error, cell = worst
            check_deadline()
            if error <= error_tolerance:
                record("accepted")
                return PiecewiseSimplexApproximation(controls, approximation.knots, table, error, error_tolerance)
            while True:
                choices = []
                for axis in range(dimension):
                    after = count // (len(knots[axis]) - 1) * len(knots[axis])
                    if after <= budgets[budget_index]:
                        lo, hi = knots[axis][cell[axis]:cell[axis] + 2]
                        choices.append(((hi - lo) / (bounds[axis][1] - bounds[axis][0]), -axis, axis, lo, hi))
                if choices:
                    break
                extend_budget("PWL exceeds {} regions at sampled error {}".format(budgets[budget_index], error))
            _, _, axis, lo, hi = max(choices)
            midpoint = (lo + hi) / 2
            if midpoint in (lo, hi):
                raise AcesLikeTransformError("PWL refinement exhausted floating point precision")
            knots[axis].append(midpoint)
            knots[axis].sort()
    except (PWLRegionBudgetError, PWLBuildTimeout):
        raise
    except Exception:
        record("error")
        raise
    finally:
        stats.update(build_wall_time=time.monotonic() - started, region_count=count,
                     max_abs_error=error, sample_count=len(samples), validated_region_count=len(errors))
