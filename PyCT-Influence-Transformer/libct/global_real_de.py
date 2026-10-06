"""Batched differential evolution for scalar GlobalReal image transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from libct.aces_like import (
    ACES_LIKE_SHIFT_KINDS, apply_aces_like_transform, apply_aces_like_joint_transform,
)


SUPPORTED_SHIFT_KINDS = ("brightness", "contrast") + ACES_LIKE_SHIFT_KINDS
DEFAULT_DE_MAXITER = 75
DEFAULT_DE_POPULATION_SIZE = 400
DEFAULT_DE_MUTATION = (0.5, 1.0)


@dataclass(frozen=True)
class GlobalRealDEResult:
    success: bool
    original_label: int
    best_label: int
    best_x: Optional[float]
    best_score: float  # Original-class score, retained for artifact compatibility.
    best_margin: float  # Original-class score minus the strongest competing score.
    best_image: np.ndarray
    iterations: int
    function_evaluations: int
    best_params: Optional[Tuple[float, float]] = None


def coefficients_for_shift(image: np.ndarray, shift_kind: str) -> np.ndarray:
    """Build the affine coefficient image for brightness or contrast."""

    sample = np.asarray(image, dtype=np.float64)
    if sample.ndim < 1 or sample.shape[-1] != 3:
        raise ValueError("GlobalReal DE requires an RGB image with three channels")
    if not np.isfinite(sample).all() or np.any(sample < 0.0) or np.any(sample > 1.0):
        raise ValueError("GlobalReal DE seed must be finite and inside [0, 1]")
    if shift_kind == "brightness":
        return np.ones_like(sample, dtype=np.float64)
    if shift_kind == "contrast":
        means = np.mean(sample, axis=tuple(range(sample.ndim - 1)), keepdims=True)
        return sample - means
    raise ValueError(
        "GlobalReal DE shift_kind must be one of "
        + "brightness, contrast (affine coefficients only)"
    )


def materialize_shift_candidates(
    image: np.ndarray,
    x_values: np.ndarray,
    shift_kind: str,
) -> np.ndarray:
    """Apply a batch of clipped scalar shifts to one normalized image."""

    sample = np.asarray(image, dtype=np.float64)
    xs = np.asarray(x_values, dtype=np.float64).reshape(-1)
    if not np.isfinite(xs).all():
        raise ValueError("GlobalReal DE candidates must be finite")
    if shift_kind in ACES_LIKE_SHIFT_KINDS:
        # The reference transform accepts one scalar at a time; prediction
        # remains batched, and each candidate uses the exact color transform.
        if len(xs) == 0:
            apply_aces_like_transform(sample, 0.0, kind=shift_kind)
            return np.empty((0,) + sample.shape, dtype=np.float32)
        return np.stack([
            apply_aces_like_transform(sample, float(x), kind=shift_kind).rgb
            for x in xs
        ]).astype(np.float32, copy=False)
    coefficients = coefficients_for_shift(sample, shift_kind)
    shifted = sample[np.newaxis, ...] + xs.reshape((-1,) + (1,) * sample.ndim) * coefficients
    return np.clip(shifted, 0.0, 1.0).astype(np.float32, copy=False)


def run_global_real_differential_evolution(
    image: np.ndarray,
    *,
    shift_kind: str,
    lower: float,
    upper: float,
    original_label: int,
    predict_batch: Callable[[np.ndarray], np.ndarray],
    random_seed: int,
    maxiter: int = DEFAULT_DE_MAXITER,
    population_size: int = DEFAULT_DE_POPULATION_SIZE,
    mutation: Tuple[float, float] = DEFAULT_DE_MUTATION,
    search_axes: Optional[str] = None,
    transform_order: str = "brightness-contrast",
) -> GlobalRealDEResult:
    """Minimize the source-to-runner-up margin with a vectorized best1bin search.

    The mutation, Latin-hypercube initialization, batch objective, and
    success-after-generation behavior follow the reference attack runner.
    """

    sample = np.asarray(image, dtype=np.float64)
    if search_axes not in (None, "brightness", "contrast", "both"):
        raise ValueError("DE search_axes must be brightness, contrast or both")
    if search_axes == "both" and shift_kind not in ACES_LIKE_SHIFT_KINDS:
        raise ValueError("joint DE search currently requires ACES-like transforms")
    if search_axes in ("brightness", "contrast"):
        shift_kind = ("aces-" if shift_kind in ACES_LIKE_SHIFT_KINDS else "") + search_axes
    if shift_kind in ACES_LIKE_SHIFT_KINDS:
        apply_aces_like_transform(sample, 0.0, kind=shift_kind)
    else:
        coefficients_for_shift(sample, shift_kind)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("DE bounds must be finite and satisfy lower < upper")
    if not lower <= 0.0 <= upper:
        raise ValueError("DE bounds must include x=0")
    if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 0:
        raise ValueError("DE maxiter must be an integer >= 0")
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, int)
        or population_size < 5
    ):
        raise ValueError("DE population_size must be an integer >= 5")
    if (
        len(mutation) != 2
        or not all(math.isfinite(value) for value in mutation)
        or mutation[0] < 0.0
        or mutation[0] >= mutation[1]
        or mutation[1] >= 2.0
    ):
        raise ValueError("DE mutation bounds must satisfy 0 <= low < high < 2")

    if search_axes == "both":
        return _run_joint_de(
            sample, lower, upper, original_label, predict_batch, random_seed,
            maxiter, population_size, mutation, transform_order,
        )

    rng = np.random.RandomState(random_seed)
    population = lower + (
        (rng.permutation(population_size) + rng.random_sample(population_size))
        / population_size
    ) * (upper - lower)
    evaluations = 0

    def score_candidates(xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal evaluations
        images = materialize_shift_candidates(sample, xs, shift_kind)
        predictions = np.asarray(predict_batch(images), dtype=np.float64)
        if predictions.ndim != 2 or predictions.shape[0] != len(xs):
            raise ValueError(
                "DE predictor must return a (candidate_count, class_count) matrix"
            )
        if predictions.shape[1] < 2 or not np.isfinite(predictions).all():
            raise ValueError("DE predictor returned invalid class scores")
        if not 0 <= original_label < predictions.shape[1]:
            raise ValueError("original_label is outside the DE predictor output")
        evaluations += len(xs)
        source_scores = predictions[:, original_label]
        competing_scores = predictions.copy()
        competing_scores[:, original_label] = -np.inf
        margins = source_scores - np.max(competing_scores, axis=1)
        return margins, source_scores, np.argmax(predictions, axis=1)

    energies, source_scores, labels = score_candidates(population)
    best_index = int(np.argmin(energies))
    iterations = 0
    successful = np.flatnonzero(labels != original_label)
    winning_x = None
    winning_label = None
    winning_score = None
    winning_margin = None
    if len(successful):
        winner = int(successful[np.argmin(energies[successful])])
        winning_x = float(population[winner])
        winning_label = int(labels[winner])
        winning_score = float(source_scores[winner])
        winning_margin = float(energies[winner])

    for generation in range(1, maxiter + 1):
        if winning_x is not None:
            break
        scale = rng.uniform(mutation[0], mutation[1])
        trials = np.empty_like(population)
        for candidate in range(population_size):
            available = np.concatenate(
                (np.arange(candidate), np.arange(candidate + 1, population_size))
            )
            first, second = rng.choice(available, size=2, replace=False)
            mutant = population[best_index] + scale * (
                population[first] - population[second]
            )
            if mutant < lower or mutant > upper:
                mutant = rng.uniform(lower, upper)
            # In one dimension, CR=1 and the forced crossover coordinate
            # always select the mutant, matching best1bin in the reference.
            trials[candidate] = mutant

        trial_energies, trial_source_scores, trial_labels = score_candidates(trials)
        iterations = generation
        successful = np.flatnonzero(trial_labels != original_label)
        if len(successful):
            winner = int(successful[np.argmin(trial_energies[successful])])
            winning_x = float(trials[winner])
            winning_label = int(trial_labels[winner])
            winning_score = float(trial_source_scores[winner])
            winning_margin = float(trial_energies[winner])
            break
        improved = trial_energies < energies
        population[improved] = trials[improved]
        energies[improved] = trial_energies[improved]
        source_scores[improved] = trial_source_scores[improved]
        labels[improved] = trial_labels[improved]
        best_index = int(np.argmin(energies))

    success = winning_x is not None
    best_x = winning_x if success else float(population[best_index])
    best_image = materialize_shift_candidates(
        sample,
        np.asarray([best_x]),
        shift_kind,
    )[0]
    return GlobalRealDEResult(
        success=success,
        original_label=int(original_label),
        best_label=winning_label if success else int(labels[best_index]),
        best_x=best_x,
        best_score=winning_score if success else float(source_scores[best_index]),
        best_margin=winning_margin if success else float(energies[best_index]),
        best_image=best_image,
        iterations=iterations,
        function_evaluations=evaluations,
    )


def _run_joint_de(
    sample, lower, upper, original_label, predict_batch, random_seed,
    maxiter, population_size, mutation, transform_order,
) -> GlobalRealDEResult:
    apply_aces_like_joint_transform(sample, 0.0, 0.0, order=transform_order)
    rng = np.random.RandomState(random_seed)
    population = np.column_stack([
        lower + ((rng.permutation(population_size) + rng.random_sample(population_size))
                 / population_size) * (upper - lower)
        for _ in range(2)
    ])
    evaluations = 0

    def score(points):
        nonlocal evaluations
        images = np.stack([
            apply_aces_like_joint_transform(sample, *point, order=transform_order).rgb
            for point in points
        ]).astype(np.float32)
        predictions = np.asarray(predict_batch(images), dtype=np.float64)
        if (predictions.ndim != 2 or predictions.shape[0] != len(points)
                or predictions.shape[1] < 2 or not np.isfinite(predictions).all()
                or not 0 <= original_label < predictions.shape[1]):
            raise ValueError("DE predictor returned invalid class scores")
        evaluations += len(points)
        source_scores = predictions[:, original_label]
        others = predictions.copy()
        others[:, original_label] = -np.inf
        return source_scores - np.max(others, axis=1), source_scores, np.argmax(predictions, axis=1)

    energies, scores, labels = score(population)
    iterations = 0
    for generation in range(maxiter + 1):
        successful = np.flatnonzero(labels != original_label)
        if len(successful):
            best = int(successful[np.argmin(energies[successful])])
            break
        best = int(np.argmin(energies))
        if generation == maxiter:
            break
        scale = rng.uniform(*mutation)
        trials = np.empty_like(population)
        for candidate in range(population_size):
            available = np.concatenate((np.arange(candidate), np.arange(candidate + 1, population_size)))
            first, second = rng.choice(available, size=2, replace=False)
            mutant = population[best] + scale * (population[first] - population[second])
            outside = (mutant < lower) | (mutant > upper)
            mutant[outside] = rng.uniform(lower, upper, size=int(outside.sum()))
            # CR=1, as in the existing scalar schedule: both mutant coordinates
            # are selected, while the total population budget stays unchanged.
            trials[candidate] = mutant
        trial_energies, trial_scores, trial_labels = score(trials)
        iterations = generation + 1
        successful = np.flatnonzero(trial_labels != original_label)
        if len(successful):
            best = int(successful[np.argmin(trial_energies[successful])])
            population, energies, scores, labels = trials, trial_energies, trial_scores, trial_labels
            break
        improved = trial_energies < energies
        population[improved], energies[improved] = trials[improved], trial_energies[improved]
        scores[improved], labels[improved] = trial_scores[improved], trial_labels[improved]
    params = tuple(float(value) for value in population[best])
    return GlobalRealDEResult(
        success=bool(labels[best] != original_label), original_label=int(original_label),
        best_label=int(labels[best]), best_x=None, best_params=params,
        best_score=float(scores[best]), best_margin=float(energies[best]),
        best_image=apply_aces_like_joint_transform(sample, *params, order=transform_order).rgb.astype(np.float32),
        iterations=iterations, function_evaluations=evaluations,
    )


__all__ = [
    "DEFAULT_DE_MAXITER",
    "DEFAULT_DE_MUTATION",
    "DEFAULT_DE_POPULATION_SIZE",
    "GlobalRealDEResult",
    "SUPPORTED_SHIFT_KINDS",
    "coefficients_for_shift",
    "materialize_shift_candidates",
    "run_global_real_differential_evolution",
]
