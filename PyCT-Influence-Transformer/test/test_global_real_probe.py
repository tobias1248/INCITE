from __future__ import annotations

import pytest

import libct.global_real_probe as probe_module
from libct.global_real_probe import (
    PROBE_STRATEGY_VERSION,
    build_probe_points,
    probe_scalar_domain,
)


def test_build_probe_points_is_bounded_and_candidate_centered() -> None:
    points = build_probe_points(0.067, -0.1, 0.1, initial_points=17)

    assert len(points) <= 17
    assert points == tuple(sorted(points))
    assert 0.067 in points
    assert min(points) >= -0.1
    assert max(points) <= 0.1


def test_build_probe_points_budget_one_returns_only_candidate() -> None:
    assert build_probe_points(0.0, -1.0, 1.0, initial_points=1) == (0.0,)


def test_probe_scalar_domain_refines_nearest_label_transition() -> None:
    evaluated = []

    def evaluate(x: float) -> int:
        evaluated.append(x)
        return int(x >= 0.03)

    result = probe_scalar_domain(
        0.0,
        -0.1,
        0.1,
        original_label=0,
        evaluate=evaluate,
        initial_points=17,
        max_refinements=8,
    )

    assert result.success is True
    assert result.attack_label == 1
    assert result.bracket_count >= 1
    assert result.refinement_steps > 0
    assert 0.03 <= result.solved_x <= 0.031
    assert tuple(evaluated) == result.evaluated_x


def test_probe_scalar_domain_handles_non_monotone_label_regions() -> None:
    def evaluate(x: float) -> int:
        return int(0.04 <= x <= 0.06)

    result = probe_scalar_domain(
        0.0,
        -0.1,
        0.1,
        original_label=0,
        evaluate=evaluate,
        initial_points=17,
        max_refinements=8,
    )

    assert result.success is True
    assert result.attack_label == 1
    assert 0.04 <= result.solved_x <= 0.041


def test_probe_scalar_domain_reports_no_attack_without_transition() -> None:
    result = probe_scalar_domain(
        0.0,
        -0.1,
        0.1,
        original_label=0,
        evaluate=lambda _x: 0,
        initial_points=17,
    )

    assert result.success is False
    assert result.solved_x is None
    assert result.bracket_count == 0
    assert len(result.evaluated_x) <= 17


def test_probe_strategy_version_is_persistable() -> None:
    assert PROBE_STRATEGY_VERSION == "candidate-centered-dyadic-v1"


def test_vector_probe_uses_total_budget_and_is_reproducible():
    candidate = (0.067, -0.03)
    evaluated = []

    def evaluate(point):
        evaluated.append(point)
        return 0

    arguments = dict(
        original_label=0, initial_points=17, max_refinements=8,
    )
    result = probe_module.probe_vector_domain(candidate, -0.1, 0.1, evaluate=evaluate, **arguments)
    repeated = probe_module.probe_vector_domain(candidate, -0.1, 0.1, evaluate=lambda point: 0, **arguments)

    assert result.success is False
    assert result.solved_x is None
    assert result.bracket_count == 0
    assert 1 <= len(result.evaluated_x) <= 17
    assert len(set(result.evaluated_x)) == len(result.evaluated_x)
    assert tuple(evaluated) == result.evaluated_x
    assert result.evaluated_x == repeated.evaluated_x
    assert result.evaluated_x[0] == candidate
    assert all(len(point) == 2 and all(-0.1 <= axis <= 0.1 for axis in point)
               for point in result.evaluated_x)
    assert any(point[0] != candidate[0] for point in result.evaluated_x)
    assert any(point[1] != candidate[1] for point in result.evaluated_x)
    assert isinstance(result.strategy_version, str)
    assert result.strategy_version != PROBE_STRATEGY_VERSION


def test_vector_probe_budget_one_evaluates_only_candidate():
    result = probe_module.probe_vector_domain(
        (0.0, 0.0), -0.1, 0.1, original_label=0, evaluate=lambda point: 0, initial_points=1,
    )
    assert result.evaluated_x == ((0.0, 0.0),)
    assert not result.success


def test_vector_probe_refines_observed_label_transition():
    def evaluate(point):
        return int(max(point) >= 0.03)

    result = probe_module.probe_vector_domain(
        (0.0, 0.0), -0.1, 0.1, original_label=0, evaluate=evaluate,
        initial_points=17, max_refinements=8,
    )
    assert result.success
    assert result.attack_label == 1
    assert result.bracket_count >= 1
    assert result.refinement_steps > 0
    assert evaluate(result.solved_x) == 1
    assert 0.03 <= max(result.solved_x) <= 0.031
    assert len(result.evaluated_x) <= 17 + 8


def test_vector_probe_accepts_successful_candidate_without_extra_probes():
    result = probe_module.probe_vector_domain(
        (0.03, -0.04), -0.1, 0.1, original_label=0, evaluate=lambda point: 1,
    )
    assert result.success
    assert result.solved_x == (0.03, -0.04)
    assert result.evaluated_x == ((0.03, -0.04),)


@pytest.mark.parametrize("candidate", [(0.0,), (0.0, 0.0, 0.0), (float("nan"), 0.0), (0.2, 0.0)])
def test_vector_probe_rejects_invalid_candidate(candidate):
    with pytest.raises(ValueError):
        probe_module.probe_vector_domain(candidate, -0.1, 0.1, original_label=0, evaluate=lambda point: 0)
