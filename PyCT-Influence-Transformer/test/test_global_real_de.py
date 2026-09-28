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


@pytest.mark.parametrize("shift_kind", ["shap-sign", "aces-brightness"])
def test_de_rejects_unsupported_transform(shift_kind: str) -> None:
    with pytest.raises(ValueError, match="shift_kind"):
        coefficients_for_shift(np.full((1, 1, 3), 0.5), shift_kind)
