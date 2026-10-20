from __future__ import annotations

from itertools import combinations, product
from math import factorial
from types import SimpleNamespace

import numpy as np
import pytest

import libct.aces_like_pwl_nd as pwl_nd
from libct.aces_like import AcesLikeTransformError
from libct.aces_like_multi import apply_aces_like_multi_transform


RGB = np.asarray([[[0.25, 0.45, 0.65]]])
CONTROLS = ("exposure", "contrast", "saturation")


def _synthetic_renderer(monkeypatch, *, nonlinear=False):
    def render(rgb, values, *, controls):
        params = np.asarray(values)
        if nonlinear:
            offset = 0.3 * np.sum(params ** 2)
        else:
            channel_slopes = np.asarray([[0.11, -0.05, 0.03], [-0.02, 0.07, 0.13],
                                         [0.04, 0.09, -0.06]])
            offset = params @ channel_slopes[:len(controls)]
        return SimpleNamespace(rgb=np.asarray(rgb) + offset)

    monkeypatch.setattr(pwl_nd, "apply_aces_like_multi_transform", render)
    return render


def _vertices(approximation, cell, permutation):
    vertex = list(cell)
    vertices = [tuple(approximation.knots[axis][index] for axis, index in enumerate(vertex))]
    for axis in permutation:
        vertex[axis] += 1
        vertices.append(tuple(approximation.knots[axis][index]
                              for axis, index in enumerate(vertex)))
    return vertices


def _affine_value(approximation, index, point):
    coefficients, intercept = approximation.affine_for_region(index)
    return intercept + np.tensordot(np.asarray(point), coefficients, axes=1)


@pytest.mark.parametrize("dimension", [2, 3])
def test_affine_renderer_is_exact_in_every_simplex_and_on_box_boundaries(monkeypatch, dimension):
    render = _synthetic_renderer(monkeypatch)
    controls = CONTROLS[:dimension]
    bounds = ((-0.2, 0.1), (-0.1, 0.3), (0.0, 0.2))[:dimension]
    approximation = pwl_nd.build_adaptive_pwl_nd(
        RGB, controls=controls, bounds=bounds, error_tolerance=1e-12
    )

    assert approximation.controls == controls
    assert approximation.region_count == factorial(dimension) * np.prod(
        [len(knots) - 1 for knots in approximation.knots]
    )
    assert len(approximation.regions) == approximation.region_count
    assert approximation.max_abs_error <= 1e-12
    seen = set()
    for index, (cell, permutation) in enumerate(approximation.regions):
        vertices = _vertices(approximation, cell, permutation)
        point = np.mean(vertices, axis=0)
        seen.add(approximation.region_index(*point))
        assert approximation.region_index(*point) == index
        coefficients, intercept = approximation.affine_for_region(index)
        assert coefficients.shape == (dimension,) + RGB.shape
        assert intercept.shape == RGB.shape
        np.testing.assert_allclose(_affine_value(approximation, index, point),
                                   render(RGB, point, controls=controls).rgb, atol=1e-12)
    assert seen == set(range(approximation.region_count))

    for point in product(*(np.linspace(lo, hi, 7) for lo, hi in bounds)):
        np.testing.assert_allclose(approximation.evaluate(*point),
                                   render(RGB, point, controls=controls).rgb, atol=1e-12)
    np.testing.assert_array_equal(approximation.evaluate(*((0.0,) * dimension)), RGB)


@pytest.mark.parametrize("dimension", [2, 3])
def test_adaptive_grid_meets_heldout_error_and_shares_all_simplex_faces(monkeypatch, dimension):
    render = _synthetic_renderer(monkeypatch, nonlinear=True)
    bounds = ((-0.2, 0.2),) * dimension
    tolerance = 0.003
    approximation = pwl_nd.build_adaptive_pwl_nd(
        RGB, controls=CONTROLS[:dimension], bounds=bounds,
        error_tolerance=tolerance, max_regions=512,
    )
    assert approximation.region_count > factorial(dimension) * 2 ** dimension
    assert approximation.region_count <= 512
    assert approximation.max_abs_error <= tolerance
    for knots in approximation.knots:
        assert knots[0] == -0.2 and knots[-1] == 0.2
        assert 0.0 in knots
        assert np.all(np.diff(knots) > 0)

    rng = np.random.RandomState(73)
    for point in rng.uniform(-0.2, 0.2, size=(47, dimension)):
        exact = render(RGB, point, controls=CONTROLS[:dimension]).rgb
        assert np.max(np.abs(approximation.evaluate(*point) - exact)) <= tolerance + 1e-12

    faces = {}
    for index, (cell, permutation) in enumerate(approximation.regions):
        vertices = _vertices(approximation, cell, permutation)
        for face in combinations(vertices, dimension):
            faces.setdefault(tuple(sorted(face)), []).append(index)
    shared_faces = 0
    for face, neighbors in faces.items():
        assert len(neighbors) in (1, 2)
        if len(neighbors) == 2:
            shared_faces += 1
            point = np.mean(face, axis=0)
            np.testing.assert_allclose(
                _affine_value(approximation, neighbors[0], point),
                _affine_value(approximation, neighbors[1], point), atol=1e-12,
            )
    assert shared_faces > 0


@pytest.mark.parametrize("controls,bounds", [
    (("exposure", "contrast"), ((-0.04, 0.04), (-0.04, 0.04))),
    (("exposure", "saturation", "curves"), ((-0.02, 0.02),) * 3),
    (("exposure", "contrast"), ((0.0, 0.04), (-0.04, 0.0))),
])
def test_real_renderer_preserves_origin_knots_and_heldout_accuracy(controls, bounds):
    approximation = pwl_nd.build_adaptive_pwl_nd(RGB, controls=controls, bounds=bounds)
    np.testing.assert_array_equal(approximation.evaluate(*((0.0,) * len(controls))), RGB)
    assert approximation.max_abs_error <= approximation.error_tolerance
    for point in product(*approximation.knots):
        np.testing.assert_allclose(
            approximation.evaluate(*point),
            apply_aces_like_multi_transform(RGB, point, controls=controls).rgb, atol=1e-12,
        )
    for point in np.random.RandomState(19).uniform(0.0, 1.0, (17, len(controls))):
        params = tuple(lo + fraction * (hi - lo) for fraction, (lo, hi) in zip(point, bounds))
        exact = apply_aces_like_multi_transform(RGB, params, controls=controls).rgb
        assert np.max(np.abs(approximation.evaluate(*params) - exact)) <= 1 / 255 + 1e-12


def test_builder_canonicalizes_controls_and_reorders_their_distinct_bounds(monkeypatch):
    _synthetic_renderer(monkeypatch)
    approximation = pwl_nd.build_adaptive_pwl_nd(
        RGB, controls=("saturation", "exposure"), bounds=((-0.3, 0.4), (-0.1, 0.2))
    )
    assert approximation.controls == ("exposure", "saturation")
    assert tuple((knots[0], knots[-1]) for knots in approximation.knots) == (
        (-0.1, 0.2), (-0.3, 0.4)
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_initial_origin_grid_cannot_exceed_hard_region_cap(monkeypatch, dimension):
    _synthetic_renderer(monkeypatch)
    with pytest.raises(AcesLikeTransformError):
        pwl_nd.build_adaptive_pwl_nd(
            RGB, controls=CONTROLS[:dimension], bounds=((-0.2, 0.2),) * dimension,
            max_regions=factorial(dimension) * 2 ** dimension - 1,
        )


@pytest.mark.parametrize("dimension", [2, 3])
def test_refinement_fails_closed_when_region_cap_cannot_meet_tolerance(monkeypatch, dimension):
    _synthetic_renderer(monkeypatch, nonlinear=True)
    with pytest.raises(AcesLikeTransformError):
        pwl_nd.build_adaptive_pwl_nd(
            RGB, controls=CONTROLS[:dimension], bounds=((-0.2, 0.2),) * dimension,
            max_regions=factorial(dimension) * 2 ** dimension, error_tolerance=1e-8,
        )


@pytest.mark.parametrize("settings", [
    {"max_regions": True}, {"max_regions": 8.5}, {"max_regions": 0},
    {"max_regions": float("inf")}, {"error_tolerance": 0.0},
    {"error_tolerance": float("nan")}, {"error_tolerance": float("inf")},
    {"controls": ("exposure",)}, {"bounds": ((0.1, 0.2), (-0.1, 0.1))},
])
def test_builder_rejects_invalid_configuration(settings):
    arguments = {"controls": ("exposure", "contrast"), "bounds": ((-0.1, 0.1),) * 2}
    arguments.update(settings)
    with pytest.raises(AcesLikeTransformError):
        pwl_nd.build_adaptive_pwl_nd(RGB, **arguments)


@pytest.mark.parametrize("params", [
    (-0.1001, 0.0), (0.0, 0.1001), (float("nan"), 0.0),
    (0.0, float("inf")), (0.0,), (0.0, 0.0, 0.0),
])
def test_evaluation_rejects_nonfinite_out_of_domain_and_wrong_dimension(monkeypatch, params):
    _synthetic_renderer(monkeypatch)
    approximation = pwl_nd.build_adaptive_pwl_nd(
        RGB, controls=CONTROLS[:2], bounds=((-0.1, 0.1),) * 2
    )
    with pytest.raises(AcesLikeTransformError):
        approximation.evaluate(*params)
    with pytest.raises(AcesLikeTransformError):
        approximation.region_index(*params)


@pytest.mark.parametrize("index", [-1, True, 0.5, 8])
def test_affine_region_index_must_name_an_existing_simplex(monkeypatch, index):
    _synthetic_renderer(monkeypatch)
    approximation = pwl_nd.build_adaptive_pwl_nd(
        RGB, controls=CONTROLS[:2], bounds=((-0.1, 0.1),) * 2
    )
    with pytest.raises(AcesLikeTransformError):
        approximation.affine_for_region(index)


def test_budget_ladder_reuses_samples_and_unchanged_region_validation(monkeypatch):
    from collections import Counter

    render = _synthetic_renderer(monkeypatch, nonlinear=True)
    samples, validations = Counter(), Counter()

    def counted_render(rgb, values, *, controls):
        samples[tuple(values)] += 1
        return render(rgb, values, controls=controls)

    original_affine = pwl_nd.PiecewiseSimplexApproximation.affine_for_region

    def counted_affine(self, index):
        cell, permutation = self.regions[index]
        intervals = tuple((self.knots[i][j], self.knots[i][j + 1]) for i, j in enumerate(cell))
        validations[(intervals, permutation)] += 1
        return original_affine(self, index)

    monkeypatch.setattr(pwl_nd, "apply_aces_like_multi_transform", counted_render)
    monkeypatch.setattr(pwl_nd.PiecewiseSimplexApproximation, "affine_for_region", counted_affine)
    stats = {}
    approximation = pwl_nd.build_adaptive_pwl_nd(
        RGB, controls=CONTROLS[:2], bounds=((-0.2, 0.2),) * 2,
        max_regions=128, region_budgets=[8, 16, 32, 128],
        error_tolerance=0.003, build_stats=stats,
    )
    assert [entry["budget"] for entry in stats["budget_history"]] == [8, 16, 32]
    assert [entry["reason"] for entry in stats["budget_history"]] == [
        "budget_exhausted", "budget_exhausted", "accepted",
    ]
    assert stats["final_budget"] == 32
    assert stats["region_count"] == approximation.region_count
    assert stats["max_abs_error"] == approximation.max_abs_error
    assert stats["build_wall_time"] >= 0
    # The origin render first validates the source, then enters the sample cache.
    assert samples.pop((0.0, 0.0)) == 2
    assert set(samples.values()) == {1}
    assert set(validations.values()) == {1}


def test_affine_mesh_stops_at_first_sufficient_budget(monkeypatch):
    _synthetic_renderer(monkeypatch)
    stats = {}
    approximation = pwl_nd.build_adaptive_pwl_nd(
        RGB, controls=CONTROLS[:2], bounds=((-0.2, 0.2),) * 2,
        max_regions=64, region_budgets=[4, 8, 64], build_stats=stats,
    )
    assert approximation.region_count == 8
    assert stats["final_budget"] == 8
    assert [entry["reason"] for entry in stats["budget_history"]] == ["budget_exhausted", "accepted"]
    assert stats["budget_history"][0]["max_abs_error"] is None


def test_exhausted_ladder_has_structured_error_and_complete_history(monkeypatch):
    _synthetic_renderer(monkeypatch, nonlinear=True)
    stats = {}
    with pytest.raises(pwl_nd.PWLRegionBudgetError, match="PWL exceeds 16 regions") as failure:
        pwl_nd.build_adaptive_pwl_nd(
            RGB, controls=CONTROLS[:2], bounds=((-0.2, 0.2),) * 2,
            max_regions=16, region_budgets=[8, 16], error_tolerance=1e-8, build_stats=stats,
        )
    assert failure.value.budget == 16
    assert failure.value.region_count <= 16
    assert failure.value.max_abs_error > 1e-8
    assert failure.value.build_stats is stats
    assert [entry["reason"] for entry in stats["budget_history"]] == ["budget_exhausted"] * 2
    assert stats["build_wall_time"] >= 0


def test_render_failure_never_increases_budget(monkeypatch):
    def broken_renderer(*args, **kwargs):
        raise AcesLikeTransformError("invalid RGB")

    monkeypatch.setattr(pwl_nd, "apply_aces_like_multi_transform", broken_renderer)
    stats = {}
    with pytest.raises(AcesLikeTransformError, match="invalid RGB") as failure:
        pwl_nd.build_adaptive_pwl_nd(
            RGB, controls=CONTROLS[:2], bounds=((-0.2, 0.2),) * 2,
            max_regions=32, region_budgets=[8, 32], build_stats=stats,
        )
    assert not isinstance(failure.value, pwl_nd.PWLRegionBudgetError)
    assert stats["final_budget"] == 8
    assert [entry["reason"] for entry in stats["budget_history"]] == ["error"]


@pytest.mark.parametrize("expire_before_build", [False, True])
def test_deadline_checks_before_build_and_after_render(monkeypatch, expire_before_build):
    now = [2.0 if expire_before_build else 0.0]
    monkeypatch.setattr(pwl_nd.time, "monotonic", lambda: now[0])
    render = _synthetic_renderer(monkeypatch)
    calls = []

    def slow_renderer(rgb, values, *, controls):
        calls.append(tuple(values))
        now[0] = 2.0
        return render(rgb, values, controls=controls)

    monkeypatch.setattr(pwl_nd, "apply_aces_like_multi_transform", slow_renderer)
    stats = {}
    with pytest.raises(pwl_nd.PWLBuildTimeout, match="deadline exceeded") as failure:
        pwl_nd.build_adaptive_pwl_nd(
            RGB, controls=CONTROLS[:2], bounds=((-0.2, 0.2),) * 2,
            max_regions=32, region_budgets=[8, 32], deadline=1.0, build_stats=stats,
        )
    assert len(calls) == (0 if expire_before_build else 1)
    assert failure.value.build_stats is stats
    assert stats["final_budget"] == 8
    assert [entry["reason"] for entry in stats["budget_history"]] == ["timeout"]


@pytest.mark.parametrize("budgets", [[], [8, 8, 32], [16, 8, 32], [0, 32], [True, 32], [8, 16]])
def test_invalid_budget_ladders_fail_before_render(monkeypatch, budgets):
    monkeypatch.setattr(pwl_nd, "apply_aces_like_multi_transform",
                        lambda *a, **kw: pytest.fail("invalid ladder reached renderer"))
    with pytest.raises(AcesLikeTransformError, match="region_budgets"):
        pwl_nd.build_adaptive_pwl_nd(
            RGB, controls=CONTROLS[:2], bounds=((-0.2, 0.2),) * 2,
            max_regions=32, region_budgets=budgets,
        )


def test_config_records_effective_cap_rebuilds_mesh_and_discards_deadline(monkeypatch):
    import libct.global_real as global_real

    _synthetic_renderer(monkeypatch, nonlinear=True)
    config = global_real.build_aces_like_global_real_config(RGB, {
        "global_shift_kind": "aces-multi", "controls": list(CONTROLS[:2]),
        "control_bounds": [[-0.2, 0.2]] * 2,
        "pwl_max_regions": 128, "pwl_region_budgets": [8, 16, 32, 128],
        "pwl_error_tolerance": 0.003, "pwl_build_deadline": pwl_nd.time.monotonic() + 10,
        "bounds_mode": "clip", "probe_enabled": False,
        "shap_priority_enabled": True, "shap_sensitivity_enabled": False,
    })
    assert config["pwl_max_regions"] == 32
    assert config["pwl_region_budgets"] == [8, 16, 32, 128]
    assert config["pwl_build_history"][-1]["reason"] == "accepted"
    assert config["pwl_build_wall_time"] >= 0
    assert "pwl_build_deadline" not in config
    global_real._cached_multi.cache_clear()
    inputs = {"v_0_0_" + str(i): float(v) for i, v in enumerate(RGB[0, 0])}
    inputs.update(dict.fromkeys(config["variable_names"], 0.0))
    approximation, _, _, _ = global_real._multi_for_inputs(inputs, config)
    assert [k.tolist() for k in approximation.knots] == config["pwl_nd_knots"]
    # A matching cached mesh must not allow an expired initial build deadline.
    with pytest.raises(pwl_nd.PWLBuildTimeout):
        global_real.build_aces_like_global_real_config(
            RGB, dict(config, pwl_max_regions=128, pwl_build_deadline=0.0),
        )


def test_deadline_is_checked_during_simplex_validation(monkeypatch):
    _synthetic_renderer(monkeypatch)
    now = [0.0]
    monkeypatch.setattr(pwl_nd.time, "monotonic", lambda: now[0])
    original_affine = pwl_nd.PiecewiseSimplexApproximation.affine_for_region

    def slow_affine(self, index):
        result = original_affine(self, index)
        now[0] = 2.0
        return result

    monkeypatch.setattr(pwl_nd.PiecewiseSimplexApproximation, "affine_for_region", slow_affine)
    stats = {}
    with pytest.raises(pwl_nd.PWLBuildTimeout):
        pwl_nd.build_adaptive_pwl_nd(
            RGB, controls=CONTROLS[:2], bounds=((-0.2, 0.2),) * 2,
            max_regions=8, deadline=1.0, build_stats=stats,
        )
    assert stats["sample_count"] == 9  # Knots populated; simplex validation interrupted.
    assert stats["validated_region_count"] == 0
    assert stats["budget_history"][-1]["reason"] == "timeout"
