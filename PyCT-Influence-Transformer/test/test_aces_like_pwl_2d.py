from __future__ import annotations

import numpy as np
import pytest

from libct.aces_like import apply_aces_like_joint_transform
from libct.aces_like_pwl_2d import build_adaptive_pwl_2d


@pytest.mark.parametrize("order", ["brightness-contrast", "contrast-brightness"])
def test_joint_pwl_preserves_origin_vertices_and_dense_accuracy(order):
    rgb = np.asarray([[[0.2, 0.5, 0.8], [0.65, 0.4, 0.3]]])
    approximation = build_adaptive_pwl_2d(
        rgb, lower=-0.1, upper=0.1, order=order,
        max_triangles=128, error_tolerance=1.0 / 255.0,
    )

    assert approximation.b_knots[0] == -0.1
    assert approximation.b_knots[-1] == 0.1
    assert approximation.c_knots[0] == -0.1
    assert approximation.c_knots[-1] == 0.1
    assert 0.0 in approximation.b_knots
    assert 0.0 in approximation.c_knots
    assert approximation.triangle_count == 2 * (len(approximation.b_knots) - 1) * (
        len(approximation.c_knots) - 1
    )
    assert approximation.triangle_count <= 128
    assert approximation.max_abs_error <= 1.0 / 255.0
    np.testing.assert_array_equal(approximation.evaluate(0.0, 0.0), rgb)
    for brightness in approximation.b_knots:
        for contrast in approximation.c_knots:
            expected = apply_aces_like_joint_transform(rgb, brightness, contrast, order=order).rgb
            np.testing.assert_allclose(approximation.evaluate(brightness, contrast), expected, atol=1e-12)

    dense_errors = [
        np.max(np.abs(approximation.evaluate(brightness, contrast) -
                      apply_aces_like_joint_transform(rgb, brightness, contrast, order=order).rgb))
        for brightness in np.linspace(-0.1, 0.1, 13)
        for contrast in np.linspace(-0.1, 0.1, 13)
    ]
    assert max(dense_errors) <= 1.0 / 255.0


def test_joint_pwl_region_affines_agree_across_diagonal_and_cell_boundaries():
    rgb = np.asarray([[[0.2, 0.5, 0.8]]])
    approximation = build_adaptive_pwl_2d(rgb, lower=-0.1, upper=0.1)
    seen_regions = set()
    for b0, b1 in zip(approximation.b_knots, approximation.b_knots[1:]):
        for c0, c1 in zip(approximation.c_knots, approximation.c_knots[1:]):
            for ub, uc in ((0.2, 0.8), (0.8, 0.2), (0.5, 0.5), (0.0, 0.3), (1.0, 0.7)):
                brightness, contrast = b0 + ub * (b1 - b0), c0 + uc * (c1 - c0)
                region = approximation.region_index(brightness, contrast)
                seen_regions.add(region)
                b_coefficient, c_coefficient, intercept = approximation.affine_for_region(region)
                np.testing.assert_allclose(
                    b_coefficient * brightness + c_coefficient * contrast + intercept,
                    approximation.evaluate(brightness, contrast), atol=1e-12,
                )
            diagonal_b, diagonal_c = (b0 + b1) / 2.0, (c0 + c1) / 2.0
            np.testing.assert_allclose(
                approximation.evaluate(diagonal_b - 1e-10, diagonal_c),
                approximation.evaluate(diagonal_b + 1e-10, diagonal_c), atol=1e-8,
            )
    assert len(seen_regions) == approximation.triangle_count


def test_joint_pwl_fails_closed_when_triangle_budget_cannot_meet_tolerance():
    with pytest.raises(ValueError):
        build_adaptive_pwl_2d(
            np.asarray([[[0.2, 0.5, 0.8]]]), lower=-0.2, upper=0.2,
            max_triangles=8, error_tolerance=1e-12,
        )


@pytest.mark.parametrize("settings", [
    {"lower": 0.1, "upper": -0.1},
    {"lower": 0.1, "upper": 0.2},
    {"lower": float("nan"), "upper": 0.1},
    {"max_triangles": True},
    {"max_triangles": 8.5},
    {"error_tolerance": 0.0},
    {"order": "invalid"},
])
def test_joint_pwl_rejects_invalid_domain_and_budget(settings):
    arguments = {"lower": -0.1, "upper": 0.1, **settings}
    with pytest.raises(ValueError):
        build_adaptive_pwl_2d(np.asarray([[[0.2, 0.5, 0.8]]]), **arguments)


@pytest.mark.parametrize("brightness,contrast", [(0.1001, 0.0), (0.0, -0.1001), (float("nan"), 0.0)])
def test_joint_pwl_does_not_extrapolate_outside_domain(brightness, contrast):
    approximation = build_adaptive_pwl_2d(np.asarray([[[0.2, 0.5, 0.8]]]), lower=-0.1, upper=0.1)
    with pytest.raises(ValueError):
        approximation.evaluate(brightness, contrast)


@pytest.mark.parametrize("lower,upper", [(0.0, 0.1), (-0.1, 0.0)])
def test_joint_pwl_preserves_origin_when_zero_is_domain_boundary(lower, upper):
    rgb = np.asarray([[[0.2, 0.5, 0.8]]])
    approximation = build_adaptive_pwl_2d(rgb, lower=lower, upper=upper)
    np.testing.assert_array_equal(approximation.evaluate(0.0, 0.0), rgb)
    assert approximation.b_knots[0] == approximation.c_knots[0] == lower
    assert approximation.b_knots[-1] == approximation.c_knots[-1] == upper
