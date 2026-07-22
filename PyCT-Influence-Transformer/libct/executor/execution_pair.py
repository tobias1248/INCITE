from __future__ import annotations

import logging
import inspect
from typing import Any, Dict, Optional, Tuple

import numpy as np

from libct.executor.legacy import LegacyConcolicExecutor
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

    def predict_validation(self, inputs: Dict[str, Any]) -> Any:
        primitive_inputs = self._engine._clone_primitive_inputs(inputs)
        val_args, val_kwargs = self._engine._complete_primitive_arguments(
            self._engine.validation_execute,
            primitive_inputs,
        )
        return self._engine.validation_execute(*val_args, **val_kwargs)

    def validate_sat_candidate(self, inputs: Dict[str, Any]) -> bool:
        recorder = self._recorder()
        attack_label = self._engine._predict_validation(inputs)
        if recorder.original_label != attack_label:
            log.warning(
                "[RESULT_CHANGE] Original result %s differs from validated candidate %s",
                recorder.original_label,
                attack_label,
            )
            recorder.find_adversarial_input(inputs, attack_label)
            return True
        return False

    def search_result_changes_label(self, inputs: Dict[str, Any], result: Any) -> bool:
        recorder = self._recorder()
        if result in (self._engine.Timeout, self._engine.Exception, self._engine.Unpicklable):
            return False
        if recorder.original_label != result:
            log.warning(
                "[RESULT_CHANGE] Original result %s differs from search candidate %s",
                recorder.original_label,
                result,
            )
            recorder.find_adversarial_input(inputs, result)
            return True
        return False

    def record_result(self, inputs: Dict[str, Any], result: Any) -> bool:
        """Retain search execution results without using them for attack validation."""
        self._engine.previous_result = result
        return True

    def candidate_execution_can_validate(self) -> bool:
        return bool(getattr(self._engine, "reuse_search_result_for_validation", False)) and not bool(
            getattr(self._engine, "single_coverage", False)
        )

    def is_valid_label_result(self, result: Any) -> bool:
        if (
            result is self._engine.Timeout
            or result is self._engine.Exception
            or result is self._engine.Unpicklable
        ):
            return False
        if result is None:
            return False
        if isinstance(result, (bool, np.bool_)):
            return False
        return isinstance(result, (int, np.integer))

    def run_initial_execution(
        self,
        all_args: Dict[str, Any],
        concolic_dict: Dict[str, Any],
    ) -> None:
        recorder = self._recorder()
        if getattr(self._engine, "trace_only", False):
            result, constraint_payload = self._engine._one_execution_deferred_constraints(
                all_args,
                concolic_dict,
            )
            recorder.original_label = (
                result if self._engine._is_valid_label_result(result) else None
            )
            log.info(
                "[TRACE-ONLY-RESULT] idx=%s result=%s label=%s",
                self._engine.idx,
                result,
                recorder.original_label,
            )
            if constraint_payload is not None:
                self._engine._apply_constraint_transfer_payload(constraint_payload)
            self._engine.in_out.append((all_args.copy(), result))
            self._engine._record_result(all_args, result)
            return
        if not self._engine._candidate_execution_can_validate():
            recorder.original_label = self._engine._predict_validation(all_args)
            self._engine.previous_result = recorder.original_label
            self._engine._one_execution(all_args, concolic_dict)
            return

        initial_args = all_args.copy()
        result, constraint_payload = self._engine._one_execution_deferred_constraints(
            all_args,
            concolic_dict,
        )
        if self._engine._is_valid_label_result(result):
            recorder.original_label = result
            log.info(
                "[INITIAL-REUSE] idx=%s reused initial search result for original label: %s",
                self._engine.idx,
                result,
            )
        else:
            recorder.original_label = self._engine._predict_validation(initial_args)
            log.warning(
                "[INITIAL-FALLBACK] idx=%s initial search result was invalid for original label: %s",
                self._engine.idx,
                result,
            )

        if constraint_payload is not None:
            self._engine._apply_constraint_transfer_payload(constraint_payload)
        self._engine.in_out.append((all_args.copy(), result))
        self._engine._record_result(all_args, result)

    def one_execution_deferred_constraints(
        self,
        all_args: Dict[str, Any],
        concolic_dict: Dict[str, Any],
    ) -> Tuple[Any, Optional[Any]]:
        envelope = self._engine._one_execution_concolic_deferred(all_args, concolic_dict)
        return self._engine._handle_child_envelope_deferred_constraints(all_args, envelope)

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
            # .copy() is important! Think why.
            self._engine.in_out.append((all_args.copy(), result))
            return self._engine._record_result(all_args, result)

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

        return self._engine._record_result(all_args, result)

    def complete_primitive_arguments(self, func: Any, all_args: Dict[str, Any]) -> Tuple[list, dict]:
        prim_args = []
        prim_kwargs = {}
        for v in inspect.signature(func).parameters.values():
            if v.kind in (inspect.Parameter.VAR_POSITIONAL,):
                continue  # ignore *args
            if v.kind in (inspect.Parameter.VAR_KEYWORD,):
                # only support 1 **kwargs and no other arguments.
                assert len(inspect.signature(func).parameters.values()) == 1
                prim_kwargs = all_args.copy()
                break

            value = v.default if (t := all_args[v.name]) is self._engine.LazyLoading else t
            if v.kind is inspect.Parameter.KEYWORD_ONLY:
                prim_kwargs[v.name] = value
            else:
                prim_args.append(value)

        return prim_args, prim_kwargs

    def _recorder(self) -> Any:
        return self._engine._get_recorder()
