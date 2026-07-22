from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from pyct.de.types import DeEncoding, DeObjective, EvaluationBatch


BatchObserver = Callable[[EvaluationBatch], None]


def _as_candidate_rows(parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim == 1:
        if values.shape[0] != 4:
            raise ValueError(f"Expected a 4-value candidate, got shape {values.shape}")
        return values.reshape(1, 4)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D DE candidate batch, got shape {values.shape}")
    if values.shape[0] == 4:
        return values.T.copy()
    if values.shape[1] == 4:
        return values.copy()
    raise ValueError(f"Expected candidate dimension 4, got shape {values.shape}")


def canonicalize_candidates(
    parameters: np.ndarray,
    image_shape: Sequence[int],
    encoding: DeEncoding,
) -> np.ndarray:
    rows = _as_candidate_rows(parameters)
    if not np.isfinite(rows).all():
        raise ValueError("DE candidates must contain only finite values")
    if len(image_shape) != 3:
        raise ValueError(f"DE one-pixel scout requires an HWC image, got {tuple(image_shape)}")
    height, width, channels = (int(value) for value in image_shape)
    if min(height, width, channels) < 1:
        raise ValueError(f"Invalid image shape: {tuple(image_shape)}")

    canonical = np.empty_like(rows, dtype=np.float64)
    if encoding == "mixed-integer":
        canonical[:, :3] = np.rint(rows[:, :3])
        canonical[:, 3] = rows[:, 3]
    elif encoding == "legacy-truncate":
        truncated = rows.astype(np.int64)
        canonical[:, :3] = truncated[:, :3]
        canonical[:, 3] = truncated[:, 3] / 255.0
    else:
        raise ValueError(f"Unsupported DE encoding: {encoding}")

    canonical[:, 0] = np.clip(canonical[:, 0], 0, height - 1)
    canonical[:, 1] = np.clip(canonical[:, 1], 0, width - 1)
    canonical[:, 2] = np.clip(canonical[:, 2], 0, channels - 1)
    canonical[:, 3] = np.clip(canonical[:, 3], 0.0, 1.0)
    return canonical


def perturb_batch(image: np.ndarray, canonical: np.ndarray) -> np.ndarray:
    source = np.asarray(image, dtype=np.float32)
    candidates = np.asarray(canonical, dtype=np.float64)
    if source.ndim != 3 or candidates.ndim != 2 or candidates.shape[1] != 4:
        raise ValueError("Expected one HWC image and an (N, 4) canonical candidate batch")
    batch = np.repeat(source[np.newaxis, ...], candidates.shape[0], axis=0)
    indices = candidates[:, :3].astype(np.int64)
    batch[np.arange(len(batch)), indices[:, 0], indices[:, 1], indices[:, 2]] = candidates[:, 3]
    return batch


def _predict_batch(model: object, batch: np.ndarray) -> np.ndarray:
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise TypeError("DE model must provide a callable predict(batch) method")
    try:
        output = predict(batch, verbose=0)
    except TypeError:
        output = predict(batch)
    probabilities = np.asarray(output, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != batch.shape[0]:
        raise ValueError(
            f"Model prediction must have shape (batch, classes), got {probabilities.shape}"
        )
    if probabilities.shape[1] < 2:
        raise ValueError("DE margin objective requires at least two output classes")
    if not np.isfinite(probabilities).all():
        raise ValueError("Model prediction contains NaN or Inf")
    return probabilities


def class_margin(probabilities: np.ndarray, original_class: int) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    if original_class < 0 or original_class >= values.shape[1]:
        raise ValueError(f"original_class {original_class} is outside model outputs")
    original = values[:, original_class]
    masked = values.copy()
    masked[:, original_class] = -np.inf
    competitor = np.max(masked, axis=1)
    return original - competitor, competitor


class VectorizedOnePixelObjective:
    def __init__(
        self,
        *,
        model: object,
        image: np.ndarray,
        original_class: int,
        encoding: DeEncoding,
        objective: DeObjective,
        observer: Optional[BatchObserver] = None,
    ) -> None:
        self.model = model
        self.image = np.asarray(image, dtype=np.float32)
        self.original_class = int(original_class)
        self.encoding = encoding
        self.objective = objective
        self.observer = observer
        self.model_evaluations = 0
        self.last_probabilities: Optional[np.ndarray] = None

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        raw = _as_candidate_rows(parameters)
        canonical = canonicalize_candidates(raw, self.image.shape, self.encoding)
        probabilities = _predict_batch(self.model, perturb_batch(self.image, canonical))
        margin, competitor = class_margin(probabilities, self.original_class)
        original_score = probabilities[:, self.original_class]
        if self.objective == "margin":
            energy = margin
        elif self.objective == "original-confidence":
            energy = original_score
        else:
            raise ValueError(f"Unsupported DE objective: {self.objective}")
        predicted = np.argmax(probabilities, axis=1).astype(np.int64)
        self.model_evaluations += int(raw.shape[0])
        self.last_probabilities = probabilities
        if self.observer is not None:
            self.observer(
                EvaluationBatch(
                    raw=raw.copy(),
                    canonical=canonical.copy(),
                    energy=np.asarray(energy, dtype=np.float64).copy(),
                    margin=margin.copy(),
                    predicted=predicted.copy(),
                    original_score=original_score.copy(),
                    competitor_score=competitor.copy(),
                )
            )
        return np.asarray(energy, dtype=np.float64)


__all__ = [
    "VectorizedOnePixelObjective",
    "canonicalize_candidates",
    "class_margin",
    "perturb_batch",
]
