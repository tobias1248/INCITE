"""Deterministic one-dimensional concrete probing for GlobalReal candidates.

The symbolic solver proves that a shared X value reaches a feasible path, but
that path does not necessarily change the model label.  This module probes the
same scalar X with concrete reference predictions before the caller spends
another expensive concolic execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple


PROBE_STRATEGY_VERSION = "candidate-centered-dyadic-v1"
DEFAULT_PROBE_INITIAL_POINTS = 17
DEFAULT_PROBE_MAX_REFINEMENTS = 8
DEFAULT_PROBE_TOLERANCE_FRACTION = 1.0 / 1024.0


@dataclass(frozen=True)
class ScalarProbeResult:
    """Outcome and diagnostics for a scalar concrete probe."""

    success: bool
    solved_x: Optional[float]
    attack_label: Any
    evaluated_x: Tuple[float, ...]
    bracket_count: int
    refinement_steps: int


def _validate_domain(candidate_x: float, lower: float, upper: float) -> None:
    values = (candidate_x, lower, upper)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("probe X values must be finite")
    if lower >= upper:
        raise ValueError("probe X bounds must satisfy lower < upper")
    if candidate_x < lower or candidate_x > upper:
        raise ValueError("candidate X must be inside probe bounds")


def _append_unique(points: List[float], value: float) -> None:
    if not any(
        math.isclose(value, point, rel_tol=0.0, abs_tol=1e-12)
        for point in points
    ):
        points.append(value)


def build_probe_points(
    candidate_x: float,
    lower: float,
    upper: float,
    *,
    initial_points: int = DEFAULT_PROBE_INITIAL_POINTS,
) -> Tuple[float, ...]:
    """Return candidate-centered dyadic points plus deterministic coverage points."""

    _validate_domain(candidate_x, lower, upper)
    if isinstance(initial_points, bool) or initial_points < 1:
        raise ValueError("initial_points must be an integer >= 1")

    budget = int(initial_points)
    width = upper - lower
    points: List[float] = []
    _append_unique(points, candidate_x)
    if budget == 1:
        return (candidate_x,)

    radius = width / 32.0
    while len(points) < budget and radius <= width:
        for value in (max(lower, candidate_x - radius), min(upper, candidate_x + radius)):
            if len(points) >= budget:
                break
            _append_unique(points, value)
        radius *= 2.0

    for value in (lower, upper):
        if len(points) >= budget:
            break
        _append_unique(points, value)

    # Fill any remaining budget with a uniform scaffold.  The dyadic points
    # remain first in evaluation order, while the returned points are sorted
    # for reliable neighboring-bracket detection.
    scaffold_count = max(budget, 2)
    for index in range(scaffold_count):
        if len(points) >= budget:
            break
        value = lower + width * index / (scaffold_count - 1)
        _append_unique(points, value)

    return tuple(sorted(points))


def _ordered_by_distance(
    points: Tuple[float, ...], candidate_x: float
) -> Tuple[float, ...]:
    return tuple(sorted(points, key=lambda point: (abs(point - candidate_x), point)))


def _is_attack(label: Any, original_label: Any) -> bool:
    return label != original_label


def probe_scalar_domain(
    candidate_x: float,
    lower: float,
    upper: float,
    *,
    original_label: Any,
    evaluate: Callable[[float], Any],
    initial_points: int = DEFAULT_PROBE_INITIAL_POINTS,
    max_refinements: int = DEFAULT_PROBE_MAX_REFINEMENTS,
    tolerance: Optional[float] = None,
) -> ScalarProbeResult:
    """Probe a scalar domain and refine the nearest label-transition bracket.

    The initial pass is deliberately non-monotonic-safe: it evaluates a
    candidate-centered dyadic scaffold and checks every neighboring pair.  A
    bracket is refined only after both labels have been observed.  This avoids
    assuming that a neural-network label is monotone in X.
    """

    _validate_domain(candidate_x, lower, upper)
    if isinstance(max_refinements, bool) or max_refinements < 0:
        raise ValueError("max_refinements must be an integer >= 0")
    width = upper - lower
    if tolerance is None:
        tolerance = width * DEFAULT_PROBE_TOLERANCE_FRACTION
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("probe tolerance must be finite and > 0")

    points = build_probe_points(
        candidate_x,
        lower,
        upper,
        initial_points=initial_points,
    )
    labels: dict[float, Any] = {}
    evaluated: List[float] = []

    for point in _ordered_by_distance(points, candidate_x):
        label = evaluate(point)
        labels[point] = label
        evaluated.append(point)
        if math.isclose(point, candidate_x, rel_tol=0.0, abs_tol=1e-12) and _is_attack(
            label, original_label
        ):
            return ScalarProbeResult(
                success=True,
                solved_x=point,
                attack_label=label,
                evaluated_x=tuple(evaluated),
                bracket_count=0,
                refinement_steps=0,
            )

    brackets = []
    sorted_points = tuple(sorted(labels))
    for left, right in zip(sorted_points, sorted_points[1:]):
        left_attack = _is_attack(labels[left], original_label)
        right_attack = _is_attack(labels[right], original_label)
        if left_attack != right_attack:
            brackets.append((left, right))

    if not brackets:
        attack_points = [
            point for point in sorted_points if _is_attack(labels[point], original_label)
        ]
        if attack_points:
            solved_x = min(
                attack_points, key=lambda point: (abs(point - candidate_x), point)
            )
            return ScalarProbeResult(
                success=True,
                solved_x=solved_x,
                attack_label=labels[solved_x],
                evaluated_x=tuple(evaluated),
                bracket_count=0,
                refinement_steps=0,
            )
        return ScalarProbeResult(
            success=False,
            solved_x=None,
            attack_label=None,
            evaluated_x=tuple(evaluated),
            bracket_count=0,
            refinement_steps=0,
        )

    brackets.sort(
        key=lambda pair: (
            min(abs(pair[0] - candidate_x), abs(pair[1] - candidate_x)),
            pair,
        )
    )
    refinement_steps = 0
    for left, right in brackets:
        left_attack = _is_attack(labels[left], original_label)
        attack_x = left if left_attack else right
        original_x = right if left_attack else left

        for _ in range(int(max_refinements)):
            if abs(attack_x - original_x) <= tolerance:
                break
            midpoint = (attack_x + original_x) * 0.5
            if midpoint in labels:
                break
            label = evaluate(midpoint)
            labels[midpoint] = label
            evaluated.append(midpoint)
            refinement_steps += 1
            if _is_attack(label, original_label):
                attack_x = midpoint
            else:
                original_x = midpoint

        return ScalarProbeResult(
            success=True,
            solved_x=attack_x,
            attack_label=labels[attack_x],
            evaluated_x=tuple(evaluated),
            bracket_count=len(brackets),
            refinement_steps=refinement_steps,
        )

    raise AssertionError("probe brackets must produce an adversarial endpoint")


__all__ = [
    "DEFAULT_PROBE_INITIAL_POINTS",
    "DEFAULT_PROBE_MAX_REFINEMENTS",
    "DEFAULT_PROBE_TOLERANCE_FRACTION",
    "PROBE_STRATEGY_VERSION",
    "ScalarProbeResult",
    "build_probe_points",
    "probe_scalar_domain",
]
