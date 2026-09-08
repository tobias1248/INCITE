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


def test_srgb_oklab_round_trips_are_accurate() -> None:
    rgb = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.1, 0.7, 0.3], [0.95, 0.2, 0.6]]],
        dtype=np.float64,
    )

    linear = srgb_to_linear(rgb)
    assert np.allclose(linear_to_srgb(linear), rgb, atol=2e-8)
    assert np.allclose(oklab_to_linear_rgb(linear_rgb_to_oklab(linear)), linear, atol=3e-7)


@pytest.mark.parametrize("kind", ("aces-brightness", "aces-contrast"))
def test_zero_shift_is_exact_identity(kind: str) -> None:
    rgb = np.asarray([[[0.05, 0.2, 0.9], [0.3, 0.6, 0.1]]], dtype=np.float64)

    result = apply_aces_like_transform(rgb, 0.0, kind=kind)

    assert np.array_equal(result.rgb, rgb)
    assert result.diagnostics.gamut_mapped_pixel_count == 0
    assert result.diagnostics.hard_clipped_channel_count == 0
    assert result.diagnostics.max_hue_drift_degrees == 0.0


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

