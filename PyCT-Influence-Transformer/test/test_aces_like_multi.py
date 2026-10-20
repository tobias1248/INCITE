from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from libct.aces_like import AcesLikeTransformError, apply_aces_like_transform
from libct.aces_like_multi import (
    CONTROL_ORDER,
    apply_aces_like_multi_transform,
    normalize_bounds,
    normalize_controls,
)


RGB = np.asarray([[[0.12, 0.45, 0.85], [0.9, 0.25, 0.07]]])


def test_control_order_and_normalization_are_canonical():
    assert CONTROL_ORDER == (
        "exposure", "contrast", "highlights", "shadows", "saturation", "vibrance", "curves"
    )
    assert normalize_controls(("curves", "exposure", "saturation")) == (
        "exposure", "saturation", "curves"
    )


@pytest.mark.parametrize("controls", [
    (), ("exposure",), ("exposure", "contrast", "saturation", "curves"),
    ("exposure", "exposure"), ("exposure", "brightness"),
    ("exposure", "unknown"), "exposure,contrast",
])
def test_invalid_control_sets_fail_closed(controls):
    with pytest.raises(AcesLikeTransformError):
        normalize_controls(controls)


@pytest.mark.parametrize("controls", [
    controls for dimension in (2, 3) for controls in combinations(CONTROL_ORDER, dimension)
])
def test_all_supported_control_subsets_have_exact_zero_identity(controls):
    original = RGB.copy()
    result = apply_aces_like_multi_transform(RGB, (0.0,) * len(controls), controls=controls)
    np.testing.assert_array_equal(result.rgb, RGB)
    np.testing.assert_array_equal(RGB, original)
    assert result.rgb is not RGB
    assert all(value == 0 for value in vars(result.diagnostics).values())


@pytest.mark.parametrize("control", CONTROL_ORDER)
def test_zero_companions_reduce_exactly_to_existing_scalar_transform(control):
    companions = tuple(name for name in CONTROL_ORDER if name != control)[:2]
    controls = (companions[0], control, companions[1])
    values = (0.0, 0.08, 0.0)
    actual = apply_aces_like_multi_transform(RGB, values, controls=controls)
    expected = apply_aces_like_transform(RGB, 0.08, kind="aces-" + control)
    np.testing.assert_array_equal(actual.rgb, expected.rgb)


@pytest.mark.parametrize("controls,values", [
    (("saturation", "exposure"), (0.13, -0.08)),
    (("curves", "contrast", "exposure"), (-0.06, 0.1, 0.08)),
])
def test_values_follow_supplied_names_and_transforms_follow_canonical_order(controls, values):
    supplied = dict(zip(controls, values))
    expected = RGB.copy()
    mapped_count = clipped_count = 0
    for control in CONTROL_ORDER:
        if control not in supplied:
            continue
        transformed = apply_aces_like_transform(expected, supplied[control], kind="aces-" + control)
        expected = transformed.rgb
        mapped_count += transformed.diagnostics.gamut_mapped_pixel_count
        clipped_count += transformed.diagnostics.hard_clipped_channel_count

    actual = apply_aces_like_multi_transform(RGB, values, controls=controls)

    np.testing.assert_array_equal(actual.rgb, expected)
    assert actual.diagnostics.gamut_mapped_pixel_count == mapped_count
    assert actual.diagnostics.hard_clipped_channel_count == clipped_count


@pytest.mark.parametrize("values", [(0.0,), (0.0, 0.0, 0.0), (float("nan"), 0.0),
                                         (0.0, float("inf")), (0.0, -float("inf"))])
def test_exact_transform_rejects_wrong_dimension_and_nonfinite_values(values):
    with pytest.raises(AcesLikeTransformError):
        apply_aces_like_multi_transform(RGB, values, controls=("exposure", "contrast"))


@pytest.mark.parametrize("control,value", [
    ("highlights", -0.51), ("shadows", 0.51), ("curves", -1.0),
    ("curves", 1.0), ("saturation", -1.01), ("vibrance", -1.01),
])
def test_exact_transform_preserves_existing_control_domains(control, value):
    with pytest.raises(AcesLikeTransformError):
        apply_aces_like_multi_transform(RGB, (0.0, value), controls=("exposure", control))


@pytest.mark.parametrize("rgb", [
    np.zeros((2, 2)), np.asarray([[[float("nan"), 0.0, 0.0]]]),
    np.asarray([[[1.01, 0.0, 0.0]]]),
])
def test_exact_transform_rejects_invalid_image_even_at_zero(rgb):
    with pytest.raises(AcesLikeTransformError):
        apply_aces_like_multi_transform(rgb, (0.0, 0.0), controls=("exposure", "contrast"))


@pytest.mark.parametrize("bounds", [
    ((0.0, 0.2), (-0.1, 0.0)), ((-0.2, 0.2), (-0.1, 0.1)),
])
def test_normalize_bounds_preserves_distinct_per_control_intervals(bounds):
    assert normalize_bounds(("exposure", "contrast"), bounds) == bounds


def test_bounds_and_parameter_mappings_follow_control_names():
    controls = ("saturation", "exposure")
    bounds = {"exposure": (-0.1, 0.2), "saturation": (-0.3, 0.4)}
    assert normalize_bounds(controls, bounds) == ((-0.1, 0.2), (-0.3, 0.4))
    assert normalize_bounds(controls, ((-0.3, 0.4), (-0.1, 0.2))) == (
        (-0.1, 0.2), (-0.3, 0.4)
    )
    expected = apply_aces_like_multi_transform(RGB, (0.07, -0.04), controls=controls)
    actual = apply_aces_like_multi_transform(
        RGB, {"exposure": -0.04, "saturation": 0.07}, controls=controls
    )
    np.testing.assert_array_equal(actual.rgb, expected.rgb)


@pytest.mark.parametrize("mapping", [
    {"exposure": (-0.1, 0.1)},
    {"exposure": (-0.1, 0.1), "contrast": (-0.1, 0.1), "curves": (-0.1, 0.1)},
])
def test_bounds_mappings_reject_missing_or_extra_control_keys(mapping):
    with pytest.raises(AcesLikeTransformError):
        normalize_bounds(("exposure", "contrast"), mapping)


@pytest.mark.parametrize("bounds", [
    ((-0.2, 0.2),), ((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)),
    ((0.1, 0.2), (-0.2, 0.2)), ((-0.2, -0.1), (-0.2, 0.2)),
    ((0.0, 0.0), (-0.2, 0.2)), ((0.2, -0.2), (-0.2, 0.2)),
    ((float("nan"), 0.2), (-0.2, 0.2)),
    ((-0.2, float("inf")), (-0.2, 0.2)),
])
def test_normalize_bounds_rejects_invalid_intervals(bounds):
    with pytest.raises(AcesLikeTransformError):
        normalize_bounds(("exposure", "contrast"), bounds)


@pytest.mark.parametrize("control,bounds", [
    ("highlights", (-0.51, 0.2)), ("shadows", (-0.2, 0.51)),
    ("curves", (-1.0, 0.2)), ("curves", (-0.2, 1.0)),
    ("saturation", (-1.01, 0.2)), ("vibrance", (-1.01, 0.2)),
])
def test_normalize_bounds_validates_each_control_domain(control, bounds):
    with pytest.raises(AcesLikeTransformError):
        normalize_bounds(("exposure", control), ((-0.2, 0.2), bounds))
