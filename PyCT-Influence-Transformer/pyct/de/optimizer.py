from __future__ import annotations

import time
from dataclasses import fields
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import differential_evolution

from pyct.de.objective import VectorizedOnePixelObjective, _predict_batch, perturb_batch
from pyct.de.types import DeConfig, DeRunResult, EvaluationBatch


_BATCH_FIELDS = tuple(field.name for field in fields(EvaluationBatch))


def _take_batch(batch: EvaluationBatch, order: np.ndarray) -> EvaluationBatch:
    return EvaluationBatch(**{name: getattr(batch, name)[order].copy() for name in _BATCH_FIELDS})


class PopulationTraceRecorder:
    def __init__(
        self,
        expected_population_size: int,
        generation_sink: Optional[Callable[[int, Dict[str, np.ndarray]], None]] = None,
    ) -> None:
        self.expected_population_size = int(expected_population_size)
        self.generation_sink = generation_sink
        self.initial: Optional[EvaluationBatch] = None
        self.current: Optional[EvaluationBatch] = None
        self.generations: List[Dict[str, np.ndarray]] = []

    @staticmethod
    def _promote(batch: EvaluationBatch) -> EvaluationBatch:
        best = int(np.argmin(batch.energy))
        order = np.arange(batch.energy.shape[0])
        order[[0, best]] = order[[best, 0]]
        return _take_batch(batch, order)

    def observe(self, batch: EvaluationBatch) -> None:
        if batch.energy.shape != (self.expected_population_size,):
            raise ValueError(
                "Unexpected DE evaluation batch size: "
                f"expected {self.expected_population_size}, got {batch.energy.shape}"
            )
        if self.current is None:
            self.initial = _take_batch(batch, np.arange(self.expected_population_size))
            self.current = self._promote(batch)
            if self.generation_sink is not None:
                self.generation_sink(
                    0,
                    {
                        f"initial_{name}": getattr(self.initial, name).copy()
                        for name in _BATCH_FIELDS
                    },
                )
            return

        parent = self.current
        accepted = batch.energy <= parent.energy
        self.generations.append(
            {
                **{f"parent_{name}": getattr(parent, name).copy() for name in _BATCH_FIELDS},
                **{f"trial_{name}": getattr(batch, name).copy() for name in _BATCH_FIELDS},
                "accepted": accepted.copy(),
            }
        )
        if self.generation_sink is not None:
            self.generation_sink(len(self.generations), self.generations[-1])
        selected = EvaluationBatch(
            **{
                name: np.where(
                    accepted.reshape((-1,) + (1,) * (getattr(parent, name).ndim - 1)),
                    getattr(batch, name),
                    getattr(parent, name),
                )
                for name in _BATCH_FIELDS
            }
        )
        self.current = self._promote(selected)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        if self.initial is None:
            raise RuntimeError("DE trace has no initial population")
        arrays = {f"initial_{name}": getattr(self.initial, name).copy() for name in _BATCH_FIELDS}
        for prefix in ("parent", "trial"):
            for name in _BATCH_FIELDS:
                key = f"{prefix}_{name}"
                arrays[key] = np.stack([record[key] for record in self.generations], axis=0)
        arrays["accepted"] = np.stack(
            [record["accepted"] for record in self.generations], axis=0
        )
        return arrays


def _de_bounds(image_shape: tuple[int, ...], encoding: str):
    height, width, channels = image_shape
    if encoding == "mixed-integer":
        return (
            [(0, height - 1), (0, width - 1), (0, channels - 1), (0.0, 1.0)],
            [True, True, True, False],
        )
    return (
        [(0, height - 1), (0, width - 1), (0, channels - 1), (0.0, 256.0)],
        None,
    )


def run_de_scout(
    model: object,
    image: np.ndarray,
    config: DeConfig,
    *,
    generation_sink: Optional[Callable[[int, Dict[str, np.ndarray]], None]] = None,
) -> DeRunResult:
    config.validate()
    sample = np.asarray(image, dtype=np.float32)
    if sample.ndim != 3:
        raise ValueError(f"DE one-pixel scout requires an HWC image, got {sample.shape}")
    clean_probabilities = _predict_batch(model, sample[np.newaxis, ...])[0]
    original_class = int(np.argmax(clean_probabilities))
    recorder = PopulationTraceRecorder(config.population_size, generation_sink=generation_sink)
    objective = VectorizedOnePixelObjective(
        model=model,
        image=sample,
        original_class=original_class,
        encoding=config.encoding,
        objective=config.objective,
        observer=recorder.observe,
    )
    bounds, integrality = _de_bounds(tuple(int(v) for v in sample.shape), config.encoding)
    started = time.perf_counter()
    callback_state = {"reason": None}

    def callback(_x: np.ndarray, _convergence: float) -> bool:
        if recorder.current is None:
            return False
        if int(recorder.current.predicted[0]) != original_class:
            callback_state["reason"] = "success"
            return True
        if config.case_timeout is not None and time.perf_counter() - started >= config.case_timeout:
            callback_state["reason"] = "timeout"
            return True
        return False

    result = differential_evolution(
        objective,
        bounds,
        maxiter=config.maxiter,
        popsize=config.population_size // 4,
        mutation=(0.5, 1.0),
        recombination=1.0,
        seed=config.seed,
        callback=callback,
        polish=False,
        atol=-1.0,
        updating="deferred",
        workers=1,
        integrality=integrality,
        vectorized=True,
    )
    duration = time.perf_counter() - started
    if recorder.current is None:
        raise RuntimeError("DE completed without an evaluated population")
    if not np.allclose(result.x, recorder.current.raw[0]) or not np.isclose(
        result.fun, recorder.current.energy[0]
    ):
        raise RuntimeError("DE population tracker diverged from SciPy result")

    best_probabilities = _predict_batch(
        model,
        perturb_batch(sample, recorder.current.canonical[[0]]),
    )[0]
    predicted_class = int(np.argmax(best_probabilities))
    stop_reason = callback_state["reason"]
    if stop_reason is None:
        stop_reason = "maxiter" if int(result.nit) >= config.maxiter else "converged"
    return DeRunResult(
        config=config,
        original_class=original_class,
        clean_probabilities=clean_probabilities.copy(),
        best_raw=recorder.current.raw[0].copy(),
        best_canonical=recorder.current.canonical[0].copy(),
        best_probabilities=best_probabilities.copy(),
        best_margin=float(recorder.current.margin[0]),
        predicted_class=predicted_class,
        success=predicted_class != original_class,
        stop_reason=str(stop_reason),
        duration_seconds=float(duration),
        scipy_nfev=int(result.nfev),
        model_evaluations=int(objective.model_evaluations),
        auxiliary_model_evaluations=2,
        total_model_evaluations=int(objective.model_evaluations) + 2,
        completed_generations=len(recorder.generations),
        trace_arrays=recorder.to_arrays(),
    )


__all__ = ["PopulationTraceRecorder", "run_de_scout"]
