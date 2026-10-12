from __future__ import annotations

import numpy as np
import pytest

import libct.aces_like as aces_like
from libct.aces_like import (
    AcesLikeTransformError,
    DEFAULT_PWL_MAX_SEGMENTS,
    apply_aces_like_transform,
    build_adaptive_pwl_approximation,
    linear_rgb_to_oklab,
    linear_to_srgb,
    oklab_to_linear_rgb,
    oklab_to_oklch,
    srgb_to_linear,
)


def _luminance(srgb: np.ndarray) -> np.ndarray:
    return np.sum(srgb_to_linear(srgb) * np.asarray((0.2126, 0.7152, 0.0722)), axis=-1)


def _hue(srgb: np.ndarray) -> np.ndarray:
    return oklab_to_oklch(linear_rgb_to_oklab(srgb_to_linear(srgb)))[..., 2]


@pytest.mark.parametrize("order", ["brightness-contrast", "contrast-brightness"])
def test_joint_transform_matches_ordered_exact_transforms(order) -> None:
    source = np.asarray([[[0.12, 0.45, 0.85], [0.9, 0.25, 0.07]]])
    original = source.copy()
    parameters = {"brightness": 0.08, "contrast": -0.06}
    expected = source
    for axis in order.split("-"):
        expected = apply_aces_like_transform(
            expected, parameters[axis], kind="aces-" + axis
        ).rgb

    result = aces_like.apply_aces_like_joint_transform(
        source, parameters["brightness"], parameters["contrast"], order=order
    )

    np.testing.assert_array_equal(result.rgb, expected)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("order", ["brightness-contrast", "contrast-brightness"])
@pytest.mark.parametrize("brightness,contrast", [(0.0, 0.0), (0.07, 0.0), (0.0, -0.05)])
def test_joint_transform_zero_axes_preserve_identity_or_single_axis(order, brightness, contrast):
    source = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float64)
    expected = source if brightness == contrast == 0.0 else apply_aces_like_transform(
        source, brightness or contrast,
        kind="aces-brightness" if brightness else "aces-contrast",
    ).rgb
    result = aces_like.apply_aces_like_joint_transform(source, brightness, contrast, order=order)
    np.testing.assert_array_equal(result.rgb, expected)


@pytest.mark.parametrize("brightness,contrast,order", [
    (float("nan"), 0.0, "brightness-contrast"),
    (0.0, float("inf"), "brightness-contrast"),
    (0.0, 0.0, "brightness-brightness"),
])
def test_joint_transform_rejects_invalid_parameters(brightness, contrast, order):
    with pytest.raises(ValueError):
        aces_like.apply_aces_like_joint_transform(
            np.asarray([[[0.2, 0.5, 0.8]]]), brightness, contrast, order=order
        )


def test_srgb_oklab_round_trips_are_accurate() -> None:
    rgb = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.1, 0.7, 0.3], [0.95, 0.2, 0.6]]],
        dtype=np.float64,
    )

    linear = srgb_to_linear(rgb)
    assert np.allclose(linear_to_srgb(linear), rgb, atol=2e-8)
    assert np.allclose(oklab_to_linear_rgb(linear_rgb_to_oklab(linear)), linear, atol=3e-7)


@pytest.mark.parametrize("kind", aces_like.ACES_LIKE_SHIFT_KINDS)
def test_zero_shift_is_exact_identity(kind: str) -> None:
    rgb = np.asarray([[[0.05, 0.2, 0.9], [0.3, 0.6, 0.1]]], dtype=np.float64)

    result = apply_aces_like_transform(rgb, 0.0, kind=kind)

    assert np.array_equal(result.rgb, rgb)
    assert result.diagnostics.gamut_mapped_pixel_count == 0
    assert result.diagnostics.hard_clipped_channel_count == 0
    assert result.diagnostics.max_hue_drift_degrees == 0.0


def test_exposure_uses_ev_in_linear_rgb_and_clips_channels() -> None:
    source = np.asarray([[[0.2, 0.5, 0.9]]])
    exposed = apply_aces_like_transform(source, 1.0, kind="aces-exposure")
    expected = np.clip(srgb_to_linear(source) * 2.0, 0.0, 1.0)
    np.testing.assert_allclose(srgb_to_linear(exposed.rgb), expected, atol=1e-12)
    assert exposed.diagnostics.hard_clipped_channel_count == 1
    brightness = apply_aces_like_transform(source, 1.0, kind="aces-brightness").rgb
    assert not np.allclose(exposed.rgb, brightness)
    darker = apply_aces_like_transform(source, -1.0, kind="aces-exposure").rgb
    np.testing.assert_allclose(srgb_to_linear(darker), srgb_to_linear(source) / 2.0)


@pytest.mark.parametrize("kind,unaffected", [
    ("aces-highlights", slice(None, 501)),
    ("aces-shadows", slice(500, None)),
])
@pytest.mark.parametrize("shift", [-0.5, -0.2, 0.2, 0.5])
def test_selective_tone_preserves_other_region_and_monotonicity(kind, unaffected, shift):
    lightness = np.linspace(0.0, 1.0, 1001)
    neutral = linear_to_srgb(np.repeat(lightness[:, None] ** 3, 3, axis=1))
    result = apply_aces_like_transform(neutral, shift, kind=kind).rgb
    measured = linear_rgb_to_oklab(srgb_to_linear(result))[..., 0]
    np.testing.assert_allclose(measured[unaffected], lightness[unaffected], atol=2e-7)
    assert np.all(np.diff(measured) >= -1e-8)
    assert measured[0] == pytest.approx(0.0)
    assert measured[-1] == pytest.approx(1.0, abs=1e-7)
    affected_index = 750 if kind == "aces-highlights" else 250
    assert np.sign(measured[affected_index] - lightness[affected_index]) == np.sign(shift)


def test_vibrance_adjusts_low_chroma_more_than_high_chroma():
    source = np.asarray([[[0.45, 0.5, 0.55], [0.2, 0.5, 0.7]]])
    lch = oklab_to_oklch(linear_rgb_to_oklab(srgb_to_linear(source)))
    outputs = {}
    for kind in ("aces-saturation", "aces-vibrance"):
        transformed = apply_aces_like_transform(source, 0.2, kind=kind)
        assert transformed.diagnostics.gamut_mapped_pixel_count == 0
        outputs[kind] = oklab_to_oklch(linear_rgb_to_oklab(srgb_to_linear(transformed.rgb)))
        np.testing.assert_allclose(outputs[kind][..., 0], lch[..., 0], atol=1e-7)
        np.testing.assert_allclose(outputs[kind][..., 2], lch[..., 2], atol=1e-6)
    saturation_gain = outputs["aces-saturation"][..., 1] / lch[..., 1]
    np.testing.assert_allclose(saturation_gain, 1.2, atol=1e-6)
    vibrance_gain = outputs["aces-vibrance"][..., 1] / lch[..., 1]
    assert 1.0 < vibrance_gain[0, 1] < vibrance_gain[0, 0] < 1.2
    desaturated = apply_aces_like_transform(source, -1.0, kind="aces-saturation").rgb
    np.testing.assert_allclose(desaturated[..., 0], desaturated[..., 1], atol=1e-6)
    np.testing.assert_allclose(desaturated[..., 1], desaturated[..., 2], atol=1e-6)


@pytest.mark.parametrize("shift", [-0.2, 0.2])
def test_curves_preserves_endpoints_and_moves_middle_anchor(shift):
    lightness = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    source = linear_to_srgb(np.repeat(lightness[:, None] ** 3, 3, axis=1))
    result = apply_aces_like_transform(source, shift, kind="aces-curves").rgb
    measured = linear_rgb_to_oklab(srgb_to_linear(result))[..., 0]
    np.testing.assert_allclose(
        measured, [0.0, 0.25 + shift / 4, 0.5 + shift / 2, 0.75 + shift / 4, 1.0], atol=2e-7
    )
    assert np.all(np.diff(measured) > 0)


@pytest.mark.parametrize("kind", aces_like.ACES_LIKE_SHIFT_KINDS)
@pytest.mark.parametrize("shift", [float("nan"), float("inf"), -float("inf")])
def test_all_controls_reject_nonfinite_shift(kind, shift):
    with pytest.raises(AcesLikeTransformError):
        apply_aces_like_transform(np.asarray([[[0.2, 0.5, 0.8]]]), shift, kind=kind)


@pytest.mark.parametrize("kind,lower,upper", [
    ("aces-saturation", -1.01, 0.2), ("aces-vibrance", -1.01, 0.2),
    ("aces-curves", -1.0, 0.2), ("aces-curves", -0.2, 1.0),
    ("aces-highlights", -0.51, 0.2), ("aces-shadows", -0.2, 0.51),
    ("aces-exposure", 0.2, -0.2), ("aces-exposure", float("nan"), 0.2),
])
def test_control_ranges_reject_invalid_curves_and_chroma(kind, lower, upper):
    with pytest.raises(AcesLikeTransformError):
        aces_like.validate_control_range(kind, lower, upper)


@pytest.mark.parametrize("kind", aces_like.ACES_LIKE_EXPERIMENT_CONTROLS)
def test_each_photo_control_has_bounded_pwl_and_zero_knot(kind):
    source = np.asarray([[[0.2, 0.5, 0.8], [0.85, 0.3, 0.2]]])
    approximation = build_adaptive_pwl_approximation(
        source, kind=kind, x_min=-0.2, x_max=0.2, max_segments=32
    )
    assert 0.0 in approximation.knots
    np.testing.assert_array_equal(approximation.evaluate(0.0), source)
    for shift in np.linspace(-0.2, 0.2, 41):
        exact = apply_aces_like_transform(source, shift, kind=kind).rgb
        assert np.max(np.abs(approximation.evaluate(shift) - exact)) <= 1.0 / 255.0 + 1e-10


def test_brightness_and_contrast_have_expected_direction() -> None:
    dark = np.full((1, 1, 3), 0.2, dtype=np.float64)
    light = np.full((1, 1, 3), 0.8, dtype=np.float64)

    assert _luminance(apply_aces_like_transform(dark, 0.4, kind="aces-brightness").rgb)[0, 0] > _luminance(dark)[0, 0]
    assert _luminance(apply_aces_like_transform(dark, -0.4, kind="aces-brightness").rgb)[0, 0] < _luminance(dark)[0, 0]
    assert _luminance(apply_aces_like_transform(dark, 0.8, kind="aces-contrast").rgb)[0, 0] < _luminance(dark)[0, 0]
    assert _luminance(apply_aces_like_transform(light, 0.8, kind="aces-contrast").rgb)[0, 0] > _luminance(light)[0, 0]


@pytest.mark.parametrize("kind", ("aces-brightness", "aces-contrast"))
def test_tone_operation_preserves_hue_when_no_gamut_mapping_is_needed(kind: str) -> None:
    rgb = np.asarray([[[0.3, 0.5, 0.4]]], dtype=np.float64)
    result = apply_aces_like_transform(rgb, 0.25, kind=kind)

    assert result.diagnostics.gamut_mapped_pixel_count == 0
    assert result.diagnostics.max_hue_drift_degrees < 1e-4
    assert abs(float(_hue(result.rgb)[0, 0] - _hue(rgb)[0, 0])) < 1e-6


def test_local_minde_accepts_a_nearby_clipped_candidate() -> None:
    source = np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float64)
    source_lch = aces_like.oklab_to_oklch(
        aces_like.linear_rgb_to_oklab(aces_like.srgb_to_linear(source))
    )
    candidate = np.array(source_lch, copy=True)
    candidate[..., 1] *= 1.01
    candidate_linear = aces_like.oklab_to_linear_rgb(
        aces_like.oklch_to_oklab(candidate)
    )
    assert not np.all(aces_like._in_linear_srgb_gamut(candidate_linear))

    mapped, gamut_mapped, hard_clipped = aces_like._constant_lightness_hue_gamut_map(
        candidate
    )

    assert bool(gamut_mapped[0, 0])
    assert hard_clipped >= 1
    assert np.allclose(mapped, np.clip(candidate_linear, 0.0, 1.0))


def test_local_minde_is_continuous_near_a_gamut_cusp() -> None:
    source = np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float64)
    source_lch = aces_like.oklab_to_oklch(
        aces_like.linear_rgb_to_oklab(aces_like.srgb_to_linear(source))
    )
    factors = np.linspace(1.0, 1.2, 401)
    mapped = []
    for factor in factors:
        candidate = np.array(source_lch, copy=True)
        candidate[..., 1] *= factor
        mapped.append(
            aces_like._constant_lightness_hue_gamut_map(candidate)[0][0, 0]
        )

    jumps = np.linalg.norm(np.diff(np.asarray(mapped), axis=0), axis=1)
    assert float(np.max(jumps)) < 0.02


def test_gamut_mapping_compresses_chroma_before_final_output_clamp() -> None:
    rgb = np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float64)
    result = apply_aces_like_transform(rgb, 0.8, kind="aces-brightness")

    assert np.all(np.isfinite(result.rgb))
    assert np.all((result.rgb >= 0.0) & (result.rgb <= 1.0))
    assert result.diagnostics.gamut_mapped_pixel_count == 1
    assert result.diagnostics.hard_clipped_channel_count >= 1
    assert result.diagnostics.max_abs_chroma_delta > 0.0


@pytest.mark.parametrize(
    "rgb,x,kind",
    (
        (np.zeros((2, 2)), 0.0, "aces-brightness"),
        (np.asarray([[[1.1, 0.0, 0.0]]]), 0.0, "aces-brightness"),
        (np.asarray([[[np.nan, 0.0, 0.0]]]), 0.0, "aces-brightness"),
        (np.asarray([[[0.0, 0.0, 0.0]]]), np.inf, "aces-brightness"),
        (np.asarray([[[0.0, 0.0, 0.0]]]), 0.0, "unknown"),
    ),
)
def test_invalid_transform_inputs_fail_closed(rgb: np.ndarray, x: float, kind: str) -> None:
    with pytest.raises(AcesLikeTransformError):
        apply_aces_like_transform(rgb, x, kind=kind)


def test_adaptive_pwl_meets_bound_and_uses_shared_knots() -> None:
    rgb = np.asarray(
        [[[0.1, 0.2, 0.7], [0.35, 0.55, 0.25]], [[0.8, 0.3, 0.15], [0.5, 0.5, 0.5]]],
        dtype=np.float64,
    )
    approximation = build_adaptive_pwl_approximation(
        rgb,
        kind="aces-brightness",
        x_min=-0.5,
        x_max=0.5,
        max_segments=DEFAULT_PWL_MAX_SEGMENTS,
    )

    assert DEFAULT_PWL_MAX_SEGMENTS == 32
    assert approximation.segment_count <= DEFAULT_PWL_MAX_SEGMENTS
    assert np.any(approximation.knots == 0.0)
    assert approximation.max_abs_error <= approximation.error_tolerance
    for x in np.linspace(-0.5, 0.5, 17):
        exact = apply_aces_like_transform(rgb, float(x), kind="aces-brightness").rgb
        assert np.max(np.abs(approximation.evaluate(float(x)) - exact)) <= 1.0 / 255.0 + 1e-10
    slope, intercept = approximation.affine_for_segment(approximation.segment_index(0.13))
    assert slope.shape == rgb.shape
    assert intercept.shape == rgb.shape


def test_pwl_validator_uses_31_points_and_covers_dense_error() -> None:
    assert len(aces_like._PWL_VALIDATION_FRACTIONS) == 31
    assert aces_like._PWL_VALIDATION_FRACTIONS == tuple(
        index / 32.0 for index in range(1, 32)
    )

    rgb = np.asarray(
        [[[0.05, 0.2, 0.85], [0.9, 0.1, 0.25]]],
        dtype=np.float64,
    )
    approximation = build_adaptive_pwl_approximation(
        rgb,
        kind="aces-contrast",
        x_min=-0.8,
        x_max=0.8,
        max_segments=DEFAULT_PWL_MAX_SEGMENTS,
    )
    dense_error = max(
        float(
            np.max(
                np.abs(
                    apply_aces_like_transform(
                        rgb, float(x), kind="aces-contrast"
                    ).rgb
                    - approximation.evaluate(float(x))
                )
            )
        )
        for x in np.linspace(-0.8, 0.8, 1001)
    )

    assert dense_error <= approximation.error_tolerance + 1e-10


@pytest.mark.parametrize(("x_min", "x_max"), ((0.0, 0.25), (-0.25, 0.0)))
def test_adaptive_pwl_supports_one_sided_intervals(x_min: float, x_max: float) -> None:
    rgb = np.asarray([[[0.25, 0.45, 0.35]]], dtype=np.float64)
    approximation = build_adaptive_pwl_approximation(
        rgb,
        kind="aces-brightness",
        x_min=x_min,
        x_max=x_max,
        max_segments=32,
    )

    assert np.array_equal(approximation.knots, np.unique(approximation.knots))
    assert approximation.knots.size >= 2
    assert np.array_equal(
        approximation.evaluate(0.0),
        apply_aces_like_transform(rgb, 0.0, kind="aces-brightness").rgb,
    )


def test_adaptive_pwl_fails_when_segment_cap_cannot_meet_error_bound() -> None:
    rgb = np.asarray([[[0.05, 0.2, 0.85], [0.9, 0.1, 0.25]]], dtype=np.float64)
    with pytest.raises(AcesLikeTransformError, match="requires more than"):
        build_adaptive_pwl_approximation(
            rgb,
            kind="aces-contrast",
            x_min=-0.8,
            x_max=0.8,
            max_segments=2,
            error_tolerance=1e-6,
        )


def test_extreme_contrast_x_fails_with_domain_error() -> None:
    rgb = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float64)
    with pytest.raises(AcesLikeTransformError, match="too large"):
        apply_aces_like_transform(rgb, 1e308, kind="aces-contrast")


def test_linear_to_srgb_handles_negative_values_without_runtime_warning() -> None:
    with np.errstate(all="raise"):
        result = linear_to_srgb(np.asarray([[[-1.0, 0.0, 1.0]]], dtype=np.float64))
    assert np.all(np.isfinite(result))
    assert result[0, 0, 0] < 0.0
