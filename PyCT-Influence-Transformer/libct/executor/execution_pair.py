from __future__ import annotations

import logging
import inspect
import time
from typing import Any, Dict, Tuple

import numpy as np

from libct.executor.legacy import LegacyConcolicExecutor
from libct.global_real import (
    TRANSFORM_MODE_AFFINE_BC,
    TRANSFORM_MODE_ACES_LIKE_PWL,
    materialize_global_real_arguments,
)
from libct.global_real_probe import (
    DEFAULT_PROBE_INITIAL_POINTS,
    DEFAULT_PROBE_MAX_REFINEMENTS,
    DEFAULT_PROBE_TOLERANCE_FRACTION,
    probe_scalar_domain,
)
from libct.utils import unwrap


log = logging.getLogger("ct.explore")


class CandidateExecutionRunner:
    """Compatibility runner for SAT candidate execution and validation."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def clone_primitive_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return a deep-ish copy of inputs with every value unwrapped."""

        def _sanitize(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: _sanitize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_sanitize(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_sanitize(v) for v in value)
            return unwrap(value)

        return {key: _sanitize(val) for key, val in inputs.items()}

    def predict_reference(self, inputs: Dict[str, Any], *, phase: str) -> Any:
        recorder = self._recorder()
        primitive_inputs = self._engine._clone_primitive_inputs(inputs)
        started_at = time.perf_counter()
        try:
            ref_args, ref_kwargs = self._engine._complete_primitive_arguments(
                self._engine.reference_execute,
                primitive_inputs,
            )
            return self._engine.reference_execute(*ref_args, **ref_kwargs)
        except Exception as exc:
            recorder.mark_error(
                "reference_prediction_failure",
                str(exc),
                phase=phase,
            )
            raise
        finally:
            recorder.record_reference_prediction(
                time.perf_counter() - started_at,
                phase=phase,
            )

    def validate_sat_candidate(self, inputs: Dict[str, Any]) -> bool:
        recorder = self._recorder()
        global_real_config = getattr(self._engine, "global_real_config", None)
        if self._should_probe_global_real(global_real_config):
            return self._validate_global_real_candidate_with_probe(
                inputs,
                global_real_config,
            )
        if self._is_hybrid_bc(global_real_config):
            attack_label, margin = self._predict_hybrid_margin(
                inputs,
                phase="candidate_reference",
                original_label=recorder.original_label,
            )
            self._record_hybrid_margin(margin, update_current=True)
            recorder.extra_meta["hybrid_pyct_candidate_count"] = (
                recorder.extra_meta.get("hybrid_pyct_candidate_count", 0) + 1
            )
        else:
            attack_label = self._engine._predict_reference(
                inputs,
                phase="candidate_reference",
            )
        if recorder.original_label != attack_label:
            log.warning(
                "[RESULT_CHANGE] Keras original label %s differs from candidate label %s",
                recorder.original_label,
                attack_label,
            )
            recorder.find_adversarial_input(inputs, attack_label)
            return True
        return False

    @staticmethod
    def _is_hybrid_bc(config: Any) -> bool:
        return (
            isinstance(config, dict)
            and (
                config.get("transform_mode") == TRANSFORM_MODE_AFFINE_BC
                or (
                    config.get("hybrid_de_enabled")
                    and config.get("transform_mode") == TRANSFORM_MODE_ACES_LIKE_PWL
                )
            )
        )

    def _record_hybrid_margin(self, margin: float, *, update_current: bool) -> None:
        meta = self._recorder().extra_meta
        meta["hybrid_pyct_best_margin"] = min(margin, meta.get("hybrid_pyct_best_margin", margin))
        if update_current:
            self._engine.current_reference_margin = margin
            meta["hybrid_pyct_last_margin"] = margin

    def _predict_hybrid_margin(
        self,
        inputs: Dict[str, Any],
        *,
        phase: str,
        original_label: Any = None,
    ) -> Tuple[int, float]:
        recorder = self._recorder()
        started_at = time.perf_counter()
        try:
            predictor = getattr(self._engine, "reference_score_predictor", None)
            if not callable(predictor):
                raise RuntimeError("hybrid PyCT requires a Keras class-score predictor")
            if getattr(recorder, "global_real_config", None) is not self._engine.global_real_config:
                raise ValueError("hybrid recorder transform differs from engine transform")
            image = recorder._build_input_from_dict(inputs)
            if image is None:
                raise ValueError("hybrid PyCT could not materialize an image")
            predictions = np.asarray(predictor(image[np.newaxis, ...]), dtype=np.float64)
            if predictions.ndim != 2 or predictions.shape[0] != 1:
                raise ValueError("hybrid Keras predictor must return one class-score vector")
            scores = predictions[0]
            if len(scores) < 2 or not np.isfinite(scores).all():
                raise ValueError("hybrid Keras predictor returned invalid class scores")
            label = int(np.argmax(scores))
            source_label = label if original_label is None else int(original_label)
            if not 0 <= source_label < len(scores):
                raise ValueError("hybrid source label is outside Keras output")
            competing_scores = scores.copy()
            competing_scores[source_label] = -np.inf
            margin = float(scores[source_label] - np.max(competing_scores))
            return label, margin
        except Exception as exc:
            recorder.mark_error("reference_prediction_failure", str(exc), phase=phase)
            raise
        finally:
            recorder.record_reference_prediction(
                time.perf_counter() - started_at, phase=phase
            )

    @staticmethod
    def _should_probe_global_real(global_real_config: Any) -> bool:
        if not isinstance(global_real_config, dict):
            return False
        if global_real_config.get("transform_mode") != "aces-like-pwl":
            return False
        if global_real_config.get("probe_enabled", True) is False:
            return False
        return global_real_config.get("global_shift_kind") in {
            "aces-brightness",
            "aces-contrast",
        }

    def _validate_global_real_candidate_with_probe(
        self,
        inputs: Dict[str, Any],
        global_real_config: Dict[str, Any],
    ) -> bool:
        recorder = self._recorder()
        variable_name = global_real_config.get("variable_name")
        if not isinstance(variable_name, str) or variable_name not in inputs:
            raise ValueError("GlobalReal probe requires a named X input")

        candidate_x = float(unwrap(inputs[variable_name]))
        lower = float(global_real_config["effective_min"])
        upper = float(global_real_config["effective_max"])
        initial_points = int(global_real_config.get("probe_initial_points", 17))
        max_refinements = int(global_real_config.get("probe_max_refinements", 8))
        tolerance_fraction = float(
            global_real_config.get("probe_tolerance_fraction", 1.0 / 1024.0)
        )
        tolerance = (upper - lower) * tolerance_fraction
        hybrid = self._is_hybrid_bc(global_real_config)
        if hybrid:
            recorder.extra_meta["hybrid_pyct_candidate_count"] = (
                recorder.extra_meta.get("hybrid_pyct_candidate_count", 0) + 1
            )

        def evaluate(x_value: float) -> Any:
            probe_inputs = dict(inputs)
            probe_inputs[variable_name] = float(x_value)
            if hybrid:
                label, margin = self._predict_hybrid_margin(
                    probe_inputs,
                    phase="candidate_probe",
                    original_label=recorder.original_label,
                )
                # After an unsuccessful probe, concolic execution uses the
                # SAT input, so its branches must inherit that input's margin.
                self._record_hybrid_margin(margin, update_current=(x_value == candidate_x))
                recorder.extra_meta["hybrid_pyct_probe_count"] = (
                    recorder.extra_meta.get("hybrid_pyct_probe_count", 0) + 1
                )
                return label
            return self._engine._predict_reference(
                probe_inputs,
                phase="candidate_probe",
            )

        started_at = time.perf_counter()
        result = probe_scalar_domain(
            candidate_x,
            lower,
            upper,
            original_label=recorder.original_label,
            evaluate=evaluate,
            initial_points=initial_points,
            max_refinements=max_refinements,
            tolerance=tolerance,
        )
        wall_time = time.perf_counter() - started_at
        if hasattr(recorder, "record_global_real_probe"):
            recorder.record_global_real_probe(result, wall_time=wall_time)

        if result.success:
            solved_inputs = dict(inputs)
            solved_inputs[variable_name] = float(result.solved_x)
            recorder.find_adversarial_input(solved_inputs, result.attack_label)
            return True
        return False

    def run_initial_execution(
        self,
        all_args: Dict[str, Any],
        concolic_dict: Dict[str, Any],
    ) -> None:
        recorder = self._recorder()
        if self._is_hybrid_bc(getattr(self._engine, "global_real_config", None)):
            source_label = recorder.extra_meta["hybrid_de_original_label"]
            label, margin = self._predict_hybrid_margin(
                all_args,
                phase="original_reference",
                original_label=source_label,
            )
            if label != source_label:
                message = "hybrid PyCT seed label differs from DE source label"
                recorder.mark_error("hybrid_seed_prediction_mismatch", message, phase="initial_seed")
                raise ValueError(message)
            recorder.original_label = source_label
            self._engine.current_reference_margin = margin
            recorder.extra_meta["hybrid_pyct_seed_margin"] = margin
            recorder.extra_meta["hybrid_pyct_best_margin"] = margin
            recorder.extra_meta["hybrid_pyct_candidate_count"] = 0
        else:
            recorder.original_label = self._engine._predict_reference(
                all_args,
                phase="original_reference",
            )
        self._engine._one_execution(all_args, concolic_dict)

    def one_execution(self, all_args: Dict[str, Any], concolic_dict: Dict[str, Any]) -> bool:
        """Run one concolic+primitive execution pair to advance exploration."""
        execution_executor = getattr(self._engine, "_execution_executor", None)
        if execution_executor is None:
            execution_executor = LegacyConcolicExecutor(self._engine)
            self._engine._execution_executor = execution_executor
        primitive_inputs = self._engine._clone_primitive_inputs(all_args)
        # primitive input arguments "all_args" may be modified here.
        result = execution_executor.run_concolic(all_args, concolic_dict)
        # We don't measure coverage in primitive mode under the non-single coverage setting.
        if not self._engine.single_coverage:
            return True

        # Coverage is measured in primitive mode because concolic-mode constraints can become unpicklable.
        answer = execution_executor.run_primitive(primitive_inputs)

        if self._engine.Timeout not in (result, answer):
            if result != answer:
                log.warning(
                    "Result mismatch detected (input=%s result=%s answer=%s)",
                    all_args,
                    result,
                    answer,
                )
            assert result == answer
        else:
            log.warning("[SINGLE_TIMEOUT] Single execution hit timeout")

        if self._engine.file_as_total:
            s = (
                self._engine.module_lines_range - self._engine.deadcode
            ) & self._engine.coverage_accumulated_missing_lines[self._engine.target_file]
        else:
            s = (
                self._engine.function_lines_range - self._engine.deadcode
            ) & self._engine.coverage_accumulated_missing_lines[self._engine.target_file]
        log.info(
            "Not Covered Yet: %s %s",
            self._engine.target_file,
            sorted(s) if s else "{}",
        )

        return True

    def complete_primitive_arguments(self, func: Any, all_args: Dict[str, Any]) -> Tuple[list, dict]:
        prim_args = []
        prim_kwargs = {}
        for v in inspect.signature(func).parameters.values():
            if v.kind in (inspect.Parameter.VAR_POSITIONAL,):
                continue  # ignore *args
            if v.kind in (inspect.Parameter.VAR_KEYWORD,):
                # only support 1 **kwargs and no other arguments.
                assert len(inspect.signature(func).parameters.values()) == 1
                global_real_config = getattr(
                    self._engine,
                    "global_real_config",
                    None,
                )
                if global_real_config is None:
                    prim_kwargs = all_args.copy()
                else:
                    prim_kwargs, _shift, _clipped_count = (
                        materialize_global_real_arguments(
                            all_args,
                            global_real_config,
                            # Primitive search/coverage must match the PWL
                            # search model. Reference calls use exact hybrid RGB.
                            exact_transform=(
                                None if func is self._engine.reference_execute else False
                            ),
                        )
                    )
                break

            value = v.default if (t := all_args[v.name]) is self._engine.LazyLoading else t
            if v.kind is inspect.Parameter.KEYWORD_ONLY:
                prim_kwargs[v.name] = value
            else:
                prim_args.append(value)

        return prim_args, prim_kwargs

    def _recorder(self) -> Any:
        return self._engine._get_recorder()
