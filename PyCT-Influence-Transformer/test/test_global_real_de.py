from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libct.global_real_de import (
    coefficients_for_shift,
    materialize_shift_candidates,
    run_global_real_differential_evolution,
)
from libct.aces_like import apply_aces_like_transform


def test_brightness_and_contrast_coefficients_are_built_from_seed() -> None:
    seed = np.asarray(
        [[[0.0, 0.2, 0.8], [1.0, 0.6, 0.2]]],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        coefficients_for_shift(seed, "brightness"),
        np.ones_like(seed),
    )
    np.testing.assert_allclose(
        coefficients_for_shift(seed, "contrast"),
        seed - seed.mean(axis=(0, 1), keepdims=True),
    )


def test_materialize_shift_candidates_clips_and_keeps_batch_axis() -> None:
    seed = np.full((2, 2, 3), 0.95, dtype=np.float32)

    shifted = materialize_shift_candidates(seed, np.asarray([0.0, 0.1]), "brightness")

    assert shifted.shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(shifted[0], seed)
    assert np.all(shifted[1] == 1.0)


@pytest.mark.parametrize("kind", ["aces-brightness", "aces-contrast"])
def test_de_materializes_exact_aces_candidates_without_mutating_source(kind) -> None:
    source = np.asarray([[[0.1, 0.4, 0.9], [0.95, 0.2, 0.05]]], dtype=np.float32)
    original = source.copy()
    shifts = np.asarray([-0.1, 0.0, 0.08])

    images = materialize_shift_candidates(source, shifts, kind)

    assert images.shape == (3, 1, 2, 3)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(images[1], source)
    for image, shift in zip(images, shifts):
        expected = apply_aces_like_transform(source, float(shift), kind=kind).rgb
        np.testing.assert_allclose(image, expected, atol=1e-7)


@pytest.mark.parametrize("kind", ["aces-brightness", "aces-contrast"])
def test_de_aces_returns_exact_best_seed_against_source_label(kind) -> None:
    source = np.asarray([[[0.2, 0.4, 0.7]]], dtype=np.float32)
    result = run_global_real_differential_evolution(
        source, shift_kind=kind, lower=-0.1, upper=0.1,
        original_label=1,
        predict_batch=lambda images: np.tile([[0.1, 0.9]], (len(images), 1)),
        random_seed=4, maxiter=1, population_size=5,
    )

    assert result.success is False
    assert result.original_label == 1
    assert result.best_label == 1
    assert result.function_evaluations == 10
    np.testing.assert_allclose(
        result.best_image,
        apply_aces_like_transform(source, result.best_x, kind=kind).rgb,
        atol=1e-7,
    )


def test_de_runs_requested_generations_and_is_reproducible_on_failure() -> None:
    seed = np.full((2, 2, 3), 0.5, dtype=np.float32)
    calls = []

    def predict_batch(images):
        calls.append(images.copy())
        return np.tile(np.asarray([[0.8, 0.2]]), (len(images), 1))

    first = run_global_real_differential_evolution(
        seed,
        shift_kind="brightness",
        lower=-0.1,
        upper=0.1,
        original_label=0,
        predict_batch=predict_batch,
        random_seed=37,
        maxiter=3,
        population_size=5,
    )
    second = run_global_real_differential_evolution(
        seed,
        shift_kind="brightness",
        lower=-0.1,
        upper=0.1,
        original_label=0,
        predict_batch=lambda images: np.tile(
            np.asarray([[0.8, 0.2]]), (len(images), 1)
        ),
        random_seed=37,
        maxiter=3,
        population_size=5,
    )

    assert first.success is False
    assert first.iterations == 3
    assert first.function_evaluations == 20
    assert len(calls) == 4
    assert first.best_x == pytest.approx(second.best_x)
    np.testing.assert_array_equal(first.best_image, second.best_image)


def test_de_stops_when_best_candidate_changes_label() -> None:
    seed = np.full((2, 2, 3), 0.5, dtype=np.float32)
    evaluated_counts = []

    def predict_batch(images):
        evaluated_counts.append(len(images))
        class_one = images.mean(axis=(1, 2, 3)) > 0.5
        probabilities = np.zeros((len(images), 2), dtype=np.float64)
        probabilities[:, 0] = np.where(class_one, 0.1, 0.9)
        probabilities[:, 1] = 1.0 - probabilities[:, 0]
        return probabilities

    result = run_global_real_differential_evolution(
        seed,
        shift_kind="brightness",
        lower=-0.1,
        upper=0.1,
        original_label=0,
        predict_batch=predict_batch,
        random_seed=4,
        maxiter=75,
        population_size=8,
    )

    assert result.success is True
    assert result.best_label == 1
    assert result.iterations <= 75
    assert sum(evaluated_counts) == result.function_evaluations


def test_de_chooses_smallest_class_margin_for_failed_seed() -> None:
    seed = np.full((1, 1, 3), 0.5, dtype=np.float32)

    def predict_batch(images):
        positive = images.mean(axis=(1, 2, 3)) >= 0.5
        return np.asarray(
            [[0.5, 0.49, 0.01] if is_positive else [0.4, 0.35, 0.25]
             for is_positive in positive],
            dtype=np.float64,
        )

    result = run_global_real_differential_evolution(
        seed,
        shift_kind="brightness",
        lower=-0.1,
        upper=0.1,
        original_label=0,
        predict_batch=predict_batch,
        random_seed=7,
        maxiter=0,
        population_size=5,
    )

    assert result.success is False
    assert result.best_x >= 0.0
    assert result.best_score == pytest.approx(0.5)
    assert result.best_margin == pytest.approx(0.01)


def test_de_detects_label_flip_even_when_margins_tie() -> None:
    seed = np.full((1, 1, 3), 0.5, dtype=np.float32)

    def predict_batch(images):
        negative = images.mean(axis=(1, 2, 3)) < 0.5
        return np.asarray(
            [[0.5, 0.5, 0.0] if is_negative else [0.0, 0.5, 0.5]
             for is_negative in negative],
            dtype=np.float64,
        )

    result = run_global_real_differential_evolution(
        seed,
        shift_kind="brightness",
        lower=-0.1,
        upper=0.1,
        original_label=1,
        predict_batch=predict_batch,
        random_seed=7,
        maxiter=0,
        population_size=5,
    )

    assert result.success is True
    assert result.best_label == 0
    assert result.best_x < 0.0
    assert result.best_margin == pytest.approx(0.0)


@pytest.mark.parametrize("shift_kind", ["shap-sign", "aces-brightness"])
def test_de_rejects_unsupported_transform(shift_kind: str) -> None:
    with pytest.raises(ValueError, match="shift_kind"):
        coefficients_for_shift(np.full((1, 1, 3), 0.5), shift_kind)


@pytest.mark.parametrize("order", ["brightness-contrast", "contrast-brightness"])
def test_joint_de_population_budget_determinism_and_exact_best_seed(order) -> None:
    from libct.aces_like import apply_aces_like_joint_transform

    source = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float32)
    original = source.copy()
    batches = []

    def predict(images):
        batches.append(images.copy())
        return np.tile([[0.9, 0.1]], (len(images), 1))

    arguments = dict(
        shift_kind="aces-brightness", search_axes="both", transform_order=order,
        lower=-0.1, upper=0.1, original_label=0, random_seed=37,
        maxiter=2, population_size=5,
    )
    first = run_global_real_differential_evolution(source, predict_batch=predict, **arguments)
    first_batches = [batch.copy() for batch in batches]
    batches.clear()
    second = run_global_real_differential_evolution(source, predict_batch=predict, **arguments)

    assert not first.success
    assert first.iterations == 2
    assert first.function_evaluations == 15
    assert [len(batch) for batch in first_batches] == [5, 5, 5]
    assert len(first.best_params) == 2
    assert all(-0.1 <= value <= 0.1 for value in first.best_params)
    assert first.best_params == second.best_params
    for original_batch, repeated_batch in zip(first_batches, batches):
        np.testing.assert_array_equal(original_batch, repeated_batch)
    np.testing.assert_allclose(
        first.best_image,
        apply_aces_like_joint_transform(source, *first.best_params, order=order).rgb,
        atol=1e-7,
    )
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("axis", ["brightness", "contrast"])
def test_explicit_single_axis_de_preserves_legacy_search(axis) -> None:
    source = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float32)
    arguments = dict(
        shift_kind="aces-" + axis, lower=-0.1, upper=0.1,
        original_label=0, random_seed=14, maxiter=1, population_size=5,
        predict_batch=lambda images: np.tile([[0.9, 0.1]], (len(images), 1)),
    )
    legacy = run_global_real_differential_evolution(source, **arguments)
    explicit = run_global_real_differential_evolution(source, search_axes=axis, **arguments)

    assert explicit.best_x == legacy.best_x
    assert explicit.function_evaluations == legacy.function_evaluations
    np.testing.assert_array_equal(explicit.best_image, legacy.best_image)


@pytest.mark.parametrize("axis", ["brightness", "contrast"])
def test_explicit_de_axis_is_independent_of_global_shift_kind(axis) -> None:
    source = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float32)
    result = run_global_real_differential_evolution(
        source, shift_kind="aces-contrast" if axis == "brightness" else "aces-brightness",
        search_axes=axis, lower=-0.1, upper=0.1, original_label=0,
        random_seed=11, maxiter=0, population_size=5,
        predict_batch=lambda images: np.tile([[0.9, 0.1]], (len(images), 1)),
    )
    expected = apply_aces_like_transform(source, result.best_x, kind="aces-" + axis).rgb
    np.testing.assert_allclose(result.best_image, expected, atol=1e-7)


def test_joint_de_detects_non_best_index_label_flip_when_margins_tie():
    batches = []

    def predict(images):
        batches.append(images.copy())
        scores = np.tile([[0.0, 0.5, 0.5]], (len(images), 1))
        scores[2] = [0.5, 0.5, 0.0]
        return scores

    result = run_global_real_differential_evolution(
        np.asarray([[[0.2, 0.5, 0.8]]]), shift_kind="aces-brightness",
        search_axes="both", lower=-0.1, upper=0.1, original_label=1,
        predict_batch=predict, random_seed=37, maxiter=75, population_size=5,
    )
    assert result.success
    assert result.best_label == 0
    assert result.best_margin == 0.0
    assert result.iterations == 0
    assert result.function_evaluations == 5
    np.testing.assert_array_equal(result.best_image, batches[0][2])
