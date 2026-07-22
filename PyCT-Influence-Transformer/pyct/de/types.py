from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np


DeEncoding = Literal["mixed-integer", "legacy-truncate"]
DeObjective = Literal["margin", "original-confidence"]


@dataclass(frozen=True)
class DeConfig:
    maxiter: int = 75
    population_size: int = 400
    seed: int = 2024
    encoding: DeEncoding = "mixed-integer"
    objective: DeObjective = "margin"
    case_timeout: Optional[float] = None

    def validate(self) -> None:
        if self.maxiter < 1:
            raise ValueError("maxiter must be >= 1")
        if self.population_size < 8 or self.population_size % 4 != 0:
            raise ValueError("population_size must be >= 8 and divisible by 4")
        if self.encoding not in {"mixed-integer", "legacy-truncate"}:
            raise ValueError(f"Unsupported DE encoding: {self.encoding}")
        if self.objective not in {"margin", "original-confidence"}:
            raise ValueError(f"Unsupported DE objective: {self.objective}")
        if self.case_timeout is not None and self.case_timeout <= 0:
            raise ValueError("case_timeout must be > 0 when provided")


@dataclass(frozen=True)
class EvaluationBatch:
    raw: np.ndarray
    canonical: np.ndarray
    energy: np.ndarray
    margin: np.ndarray
    predicted: np.ndarray
    original_score: np.ndarray
    competitor_score: np.ndarray


@dataclass(frozen=True)
class DeRunResult:
    config: DeConfig
    original_class: int
    clean_probabilities: np.ndarray
    best_raw: np.ndarray
    best_canonical: np.ndarray
    best_probabilities: np.ndarray
    best_margin: float
    predicted_class: int
    success: bool
    stop_reason: str
    duration_seconds: float
    scipy_nfev: int
    model_evaluations: int
    auxiliary_model_evaluations: int
    total_model_evaluations: int
    completed_generations: int
    trace_arrays: Dict[str, np.ndarray]


__all__ = [
    "DeConfig",
    "DeEncoding",
    "DeObjective",
    "DeRunResult",
    "EvaluationBatch",
]
