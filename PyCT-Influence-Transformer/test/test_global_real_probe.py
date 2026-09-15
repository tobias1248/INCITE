from __future__ import annotations

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
