from __future__ import annotations

import logging
import inspect
import time
from typing import Any, Dict, Tuple

from libct.executor.legacy import LegacyConcolicExecutor
from libct.global_real import materialize_global_real_arguments
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

    def run_initial_execution(
        self,
        all_args: Dict[str, Any],
        concolic_dict: Dict[str, Any],
    ) -> None:
        recorder = self._recorder()
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
