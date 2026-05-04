from __future__ import annotations
import builtins
import coverage
import func_timeout
import gc
import inspect
import logging
import math
import multiprocessing
import os
import pickle
import sys
import time
import traceback
from libct.path import PathToConstraint
from libct.solver import Solver, _ensure_smtlib2_logger
from libct.position import summarize_indices, summarize_position
from libct.utils import ConcolicObject, unwrap, get_in_dict_shape
from libct.record import ConcolicTestRecorder
from libct.executor import LegacyConcolicExecutor
from libct.searcher import Searcher, create_constraint_searcher
from libct.state import ConstraintWorkItem
import cProfile
import shap
import numpy as np

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from types import ModuleType
from libct.constraint import Constraint
from explainability.shap_calculator import ShapValuesComparator
import heapq
from collections import deque


log = logging.getLogger("ct.explore")
ENABLE_COVERAGE_LOGGING = False
# The original limit is not enough in some special cases.
sys.setrecursionlimit(1000000)
module = None
execute = None
recorder = None


class ConstraintTransferError(RuntimeError):
    """Raised when child process constraints cannot be transferred safely."""


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def prepare():
    #################################################################
    # Since the source code in https://github.com/python/cpython/blob/e822e37946f27c09953bb5733acf3b07c2db690f/Modules/socketmodule.c#L6485
    # only accepts "unwrapped" input arguments, we simply do it here.
    #################################################################
    import socket
    _socket_getaddrinfo = socket.getaddrinfo

    def socket_getaddrinfo(*args, **kwargs):
        return _socket_getaddrinfo(*map(unwrap, args), **{k: unwrap(v) for (k, v) in kwargs.items()})
    socket.getaddrinfo = socket_getaddrinfo
    #####################################################################
    # The builtin len(...) function will automatically unwrap our result,
    # so we want to avoid this by doing the following line.
    #####################################################################
    builtins.len = lambda x: x.__len__()


class ExplorationEngine:
    SHAP_SCORE_EPS = 1e-12
    # indicate occurrence of Exception during execution
    class Exception(metaclass=type(
        '', (type,), {"__repr__": lambda self: '<EXCEPTION>'})): pass
    # indicate timeout after either a concolic or a primitive execution

    class Timeout(metaclass=type(
        '', (type,), {"__repr__": lambda self: '<TIMEOUT>'})): pass
    # indicate that an object is unpicklable

    class Unpicklable(metaclass=type(
        '', (type,), {"__repr__": lambda self: '<UNPICKLABLE>'})): pass
    # lazily loading default values of primitive arguments

    class LazyLoading(metaclass=type(
        '', (type,), {"__repr__": lambda self: '<DEFAULT>'})): pass

    def __init__(self, *,
                solver="cvc5",
                timeout=20,
                constraint_build_timeout=True,
                constraint_build_timeout_seconds: int = 30,
                solver_run_timeout: Optional[int] = None,
                safety=0,
                store=None,
                verbose=1,
                logfile=None,
                statsdir=None,
                smtdir=None,
                save_dir=None,
                input_name=None,
                module_: ModuleType,
                execute_: Callable,
                validation_execute_: Optional[Callable] = None,
                only_first_forward: bool,
                shap_score_alpha: Optional[float] = None,
                symbolic_path_threshold: Optional[int] = None):
        global module, execute

        module = module_
        execute = execute_
        self.validation_execute = validation_execute_ or execute_

        self.save_dir = save_dir
        self.input_name = input_name
        self.only_first_forward = only_first_forward
        self.shap_score_alpha = (
            None if shap_score_alpha is None else float(shap_score_alpha)
        )
        self.symbolic_path_threshold = None if symbolic_path_threshold is None else int(symbolic_path_threshold)
        self.symbolic_enabled = True
        self.symbolic_disabled_at_path_len = None
        self.constraint_log_enabled = _env_flag("PYCT_ENABLE_CONSTRAINT_LOG", False)
        self.verbose = verbose
        self.logfile = logfile

        self.normalize = None
        self.__init2__()
        self.statsdir = statsdir
        if self.statsdir:
            os.system(f"rm -rf '{statsdir}'")
            os.system(f"mkdir -p '{statsdir}'")
        Solver.set_basic_configurations(
            solver,
            timeout,
            safety,
            store,
            smtdir,
            constraint_build_timeout,
            constraint_build_timeout_seconds,
            solver_run_timeout,
        )
        _ensure_smtlib2_logger()

    def __init2__(self):
        global recorder
        recorder = ConcolicTestRecorder(self.save_dir, self.input_name)
        self._reset_symbolic_guard()

        # consists of the constraints that are going to be solved by the solver
        self.path = PathToConstraint()
        self.in_out: List[Tuple[Any, Any]] = []
        self.coverage_data = coverage.CoverageData()
        self.coverage_accumulated_missing_lines = {}
        self.var_to_types = {}
        self.concolic_name_list: List[str] = []  # NOTE for DNN testing
        self.concolic_flag_dict: dict[str, int] = {}  # NOTE for DNN testing
        self.previous_result = None
        self.original_args = None  # used to limit variable range
        self._execution_executor = LegacyConcolicExecutor(self)

    def _reset_symbolic_guard(self) -> None:
        self.symbolic_enabled = True
        self.symbolic_disabled_at_path_len = None

    def _maybe_disable_symbolic(self, current_height: int) -> None:
        if not self.symbolic_enabled:
            return
        threshold = self.symbolic_path_threshold
        if threshold is None:
            return
        if current_height >= threshold:
            self.symbolic_enabled = False
            if self.symbolic_disabled_at_path_len is None:
                self.symbolic_disabled_at_path_len = current_height

    def _mark_constraint_transfer_failure(self, reason: str) -> None:
        self._mark_runtime_error(
            "constraint_transfer_failure",
            reason,
            phase="transfer",
        )

    def _mark_runtime_error(
        self,
        error_type: str,
        reason: str,
        *,
        phase: Optional[str] = None,
        child_pid: Optional[int] = None,
        event_type: Optional[str] = None,
    ) -> None:
        if recorder is None:
            return
        mark_error = getattr(recorder, "mark_error", None)
        if callable(mark_error):
            mark_error(
                error_type,
                reason,
                phase=phase,
                child_pid=child_pid,
                event_type=event_type,
            )
            return
        extra_meta = getattr(recorder, "extra_meta", None)
        if extra_meta is None:
            extra_meta = {}
            recorder.extra_meta = extra_meta
        extra_meta["status"] = "error"
        extra_meta["error_type"] = error_type
        extra_meta["error_reason"] = reason
        if phase is not None:
            extra_meta["error_phase"] = phase
        if child_pid is not None:
            extra_meta["child_pid"] = child_pid
        if event_type is not None:
            extra_meta["child_event_type"] = event_type

    def _record_child_event(
        self,
        event_type: str,
        message: str,
        *,
        phase: str,
        child_pid: Optional[int],
    ) -> None:
        if recorder is not None:
            mark_event = getattr(recorder, "mark_child_event", None)
            if callable(mark_event):
                mark_event(
                    event_type,
                    message,
                    phase=phase,
                    child_pid=child_pid,
                )
            else:
                extra_meta = getattr(recorder, "extra_meta", None)
                if extra_meta is None:
                    extra_meta = {}
                    recorder.extra_meta = extra_meta
                extra_meta["child_event_type"] = event_type
                extra_meta["child_event_message"] = message
                extra_meta["child_event_phase"] = phase
                if child_pid is not None:
                    extra_meta["child_pid"] = child_pid
        log.warning(
            "[CHILD-EVENT] idx=%s pid=%s phase=%s event_type=%s input_name=%s save_dir=%s message=%s",
            self.idx,
            child_pid,
            phase,
            event_type,
            self.input_name,
            self.save_dir,
            message,
        )

    def _write_diagnostic_file(self, filename: str, contents: Optional[str]) -> None:
        if not self.save_dir or not contents:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, filename), "w", encoding="utf-8") as handle:
            handle.write(contents)

    def _build_child_shared_state(self, updated_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "updated_args": updated_args,
            "var_to_types": self.var_to_types,
            "concolic_name_list": self.concolic_name_list,
            "concolic_flag_dict": self.concolic_flag_dict,
        }

    def _build_child_ok_envelope(
        self,
        *,
        pid: int,
        updated_args: Dict[str, Any],
        result: Any,
        constraint_payload: Any,
    ) -> Dict[str, Any]:
        envelope = {
            "kind": "ok",
            "pid": pid,
            "phase": "execute",
            "result": result,
            "constraint_payload": constraint_payload,
            "message": "child execution completed successfully",
        }
        envelope.update(self._build_child_shared_state(updated_args))
        return envelope

    def _build_child_event_envelope(
        self,
        *,
        pid: int,
        updated_args: Optional[Dict[str, Any]],
        result: Any,
        event_type: str,
        message: str,
        error_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        envelope = {
            "kind": "child_event",
            "pid": pid,
            "phase": "execute",
            "result": result,
            "event_type": event_type,
            "message": message,
        }
        if error_class is not None:
            envelope["error_class"] = error_class
        envelope.update(self._build_child_shared_state(updated_args))
        return envelope

    def _build_child_error_envelope(
        self,
        *,
        pid: int,
        updated_args: Optional[Dict[str, Any]],
        error_type: str,
        phase: str,
        message: str,
        error_class: Optional[str] = None,
        traceback_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        envelope = {
            "kind": "child_error",
            "pid": pid,
            "phase": phase,
            "result": self.Exception,
            "error_type": error_type,
            "message": message,
        }
        if error_class is not None:
            envelope["error_class"] = error_class
        if traceback_text is not None:
            envelope["traceback"] = traceback_text
        envelope.update(self._build_child_shared_state(updated_args))
        return envelope

    def _validate_child_envelope(self, envelope: Any) -> Dict[str, Any]:
        if not isinstance(envelope, dict):
            raise ValueError(f"child returned non-dict envelope: {type(envelope).__name__}")
        kind = envelope.get("kind")
        if kind not in {"ok", "child_event", "child_error"}:
            raise ValueError(f"child returned unknown envelope kind: {kind!r}")
        if "pid" not in envelope:
            raise ValueError("child envelope missing pid")
        if "phase" not in envelope:
            raise ValueError("child envelope missing phase")
        if "result" not in envelope:
            raise ValueError("child envelope missing result")
        if kind == "ok" and "constraint_payload" not in envelope:
            raise ValueError("ok envelope missing constraint_payload")
        if kind == "child_event" and "event_type" not in envelope:
            raise ValueError("child_event envelope missing event_type")
        if kind == "child_error" and "error_type" not in envelope:
            raise ValueError("child_error envelope missing error_type")
        return envelope

    def _apply_child_shared_state(self, all_args: Dict[str, Any], envelope: Dict[str, Any]) -> None:
        updated_args = envelope.get("updated_args")
        if isinstance(updated_args, dict):
            all_args.clear()
            all_args.update(updated_args)
        self.var_to_types = envelope.get("var_to_types", self.var_to_types)
        self.concolic_name_list = envelope.get("concolic_name_list", self.concolic_name_list)
        self.concolic_flag_dict = envelope.get("concolic_flag_dict", self.concolic_flag_dict)

    def _raise_transport_failure(
        self,
        reason: str,
        *,
        phase: str,
        child_pid: Optional[int] = None,
        details: Optional[str] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        log.error(
            "[PARENT-RECV-ERROR] idx=%s pid=%s phase=%s error_type=constraint_transfer_failure input_name=%s save_dir=%s message=%s",
            self.idx,
            child_pid,
            phase,
            self.input_name,
            self.save_dir,
            reason,
        )
        self._mark_runtime_error(
            "constraint_transfer_failure",
            reason,
            phase=phase,
            child_pid=child_pid,
        )
        if details:
            self._write_diagnostic_file("transfer_error_traceback.txt", details)
        if exc is None:
            raise ConstraintTransferError(reason)
        raise ConstraintTransferError(reason) from exc

    def _receive_child_envelope(
        self,
        conn: Any,
        process: multiprocessing.Process,
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            if conn.poll(0.05):
                break
            if not process.is_alive():
                reason = f"child exited before sending a valid envelope (exitcode={process.exitcode})"
                self._raise_transport_failure(
                    reason,
                    phase="transport",
                    child_pid=process.pid,
                    details=reason,
                )
            if time.monotonic() >= deadline:
                reason = f"timed out waiting for child envelope after {timeout_seconds}s"
                self._raise_transport_failure(
                    reason,
                    phase="transport",
                    child_pid=process.pid,
                    details=reason,
                )

        try:
            envelope = conn.recv()
        except (EOFError, OSError, ValueError) as exc:
            reason = f"failed to receive child envelope: {exc.__class__.__name__}: {exc}"
            self._raise_transport_failure(
                reason,
                phase="transport",
                child_pid=process.pid,
                details=traceback.format_exc(),
                exc=exc,
            )

        try:
            return self._validate_child_envelope(envelope)
        except ValueError as exc:
            reason = str(exc)
            self._raise_transport_failure(
                reason,
                phase="protocol",
                child_pid=envelope.get("pid") if isinstance(envelope, dict) else process.pid,
                details=repr(envelope),
                exc=exc,
            )

    def _handle_child_envelope(self, all_args: Dict[str, Any], envelope: Dict[str, Any]) -> Any:
        self._apply_child_shared_state(all_args, envelope)
        kind = envelope["kind"]
        child_pid = envelope.get("pid")
        phase = str(envelope.get("phase", "execute"))
        message = str(envelope.get("message", ""))

        if kind == "ok":
            self._apply_constraint_transfer_payload(envelope["constraint_payload"])
            return envelope["result"]

        if kind == "child_event":
            self._record_child_event(
                str(envelope["event_type"]),
                message,
                phase=phase,
                child_pid=child_pid,
            )
            return envelope["result"]

        error_type = str(envelope["error_type"])
        log.error(
            "[CHILD-ERROR] idx=%s pid=%s phase=%s error_type=%s input_name=%s save_dir=%s message=%s",
            self.idx,
            child_pid,
            phase,
            error_type,
            self.input_name,
            self.save_dir,
            message,
        )
        traceback_text = envelope.get("traceback")
        if traceback_text:
            filename = (
                "transfer_error_traceback.txt"
                if error_type == "constraint_transfer_failure"
                else "child_error_traceback.txt"
            )
            self._write_diagnostic_file(filename, traceback_text)
        self._mark_runtime_error(
            error_type,
            message,
            phase=phase,
            child_pid=child_pid,
        )
        if error_type == "constraint_transfer_failure":
            raise ConstraintTransferError(message)
        raise RuntimeError(message)

    def _apply_constraint_transfer_payload(self, payload: Any) -> None:
        if payload is self.Unpicklable:
            reason = "child process returned an unpicklable constraint/path payload"
            log.error("Constraint transfer failed: %s", reason)
            self._mark_constraint_transfer_failure(reason)
            raise ConstraintTransferError(reason)

        symbolic_disabled_at_path_len = None
        if isinstance(payload, tuple) and len(payload) == 4:
            (
                Constraint.global_constraints,
                self.constraints_to_solve,
                self.path,
                symbolic_disabled_at_path_len,
            ) = payload
        else:
            Constraint.global_constraints, self.constraints_to_solve, self.path = payload
        if symbolic_disabled_at_path_len is not None:
            self.symbolic_disabled_at_path_len = symbolic_disabled_at_path_len

    def _execution_loop(self, max_iterations: int, all_args, concolic_dict, *, deadline: Optional[float] = None) -> bool:
        recorder.start()
        Solver.norm = self.normalize
        Solver.limit_change_range = self.limit_change_range
        recorder.original_label = self._predict_validation(all_args)
        self.previous_result = recorder.original_label

        tried_input_args = [all_args.copy()]  # .copy() is important!!
        iterations = 0
        cont = True
        timed_out = False
        log.info(
            "[ITER-START] idx=%s iteration=%s queue_size=%s",
            self.idx,
            iterations,
            len(self.constraints_to_solve),
        )

        # this execution only for generating constraints
        log.info(f"=== Iterations: {iterations} ===")
        recorder.iter_start(Solver)
        recorder.execution_start()
        self._one_execution(all_args, concolic_dict)
        recorder.execution_end()
        recorder.iter_end(Solver.stats, 0)
        recorder.gen_constraint.append(len(self.constraints_to_solve))
        recorder.first_execution_end()

        if hasattr(recorder, "save_original_input"):
            recorder.save_original_input(all_args)
        self._update_symbolic_meta()
        recorder.save_stats_dict()

        # After First execution, no constr to solve
        if len(self.constraints_to_solve) == 0:
            log.info(
                "[FIRST_NO_CONSTR] After first execution, no constraint to solve",
            )
            return timed_out

        def _check_deadline() -> bool:
            if deadline is None:
                return False
            if time.monotonic() >= deadline:
                log.warning("[TOTAL TIMEOUT] idx=%s exceeded total timeout", self.idx)
                return True
            return False

        while cont and (max_iterations == 0 or iterations < max_iterations):
            if _check_deadline():
                timed_out = True
                recorder.total_timeout()
                break
            ##############################################################
            # In each iteration, we take one constraint out of the queue
            # and try to solve for it. After that we'll obtain a model as
            # a list of arguments and continue the next iteration with it.
            log.info(f"=== Iterations: {iterations+1} ===")
            recorder.iter_start(Solver)

            recorder.solve_constr_start()
            solve_constr_num = len(self.constraints_to_solve)
            found_adversarial = False
            while len(self.constraints_to_solve) > 0:
                if _check_deadline():
                    timed_out = True
                    break
                popped = self.pop_constraint()
                if isinstance(popped, tuple) and len(popped) == 3:
                    constraint, shap_value, position = popped
                else:
                    constraint = popped
                    shap_value = None
                    position = None
                model = Solver.find_model_from_constraint(
                    self, constraint, shap_value, position, self.idx, self.original_args)
                if model is not None and not self.only_first_forward:
                    # sat
                    all_args.update(model)  # from model to argument
                    recorder.save_sat_input(all_args)
                    if all_args not in tried_input_args:
                        # sat and this input args have not used
                        # .copy() is important!!
                        tried_input_args.append(all_args.copy())
                        found_adversarial = self._validate_sat_candidate(all_args)
                        break

            recorder.solve_constr_end()
            solve_constr_num = solve_constr_num - \
                len(self.constraints_to_solve)

            if timed_out:
                recorder.total_timeout()
                break

            if found_adversarial:
                recorder.gen_constraint.append(0)
                iterations += 1
                recorder.iter_end(Solver.stats, solve_constr_num)
                self._update_symbolic_meta()
                recorder.save_stats_dict()
                break

            # solve new input and use it to execute
            if not self.only_first_forward:
                gen_constr_num = len(self.constraints_to_solve)
                recorder.execution_start()
                cont = self._one_execution(all_args, concolic_dict)
                recorder.execution_end()
                gen_constr_num = len(
                    self.constraints_to_solve) - gen_constr_num
                recorder.gen_constraint.append(gen_constr_num)

            iterations += 1
            recorder.iter_end(Solver.stats, solve_constr_num)
            self._update_symbolic_meta()
            recorder.save_stats_dict()
            ##############################################################
            log.info(
                "[ITER-END] idx=%s iteration=%s sat=%s unsat=%s queue_size=%s",
                self.idx,
                iterations,
                Solver.stats["sat_number"],
                Solver.stats["unsat_number"],
                len(self.constraints_to_solve),
            )

            if len(self.constraints_to_solve) == 0:
                recorder.no_ctr_to_solve()
                log.info("[SOLVED_ALL_CONSTR] No constraints remain to solve")
                break
            if _check_deadline():
                timed_out = True
                recorder.total_timeout()
                break

        return timed_out

    def _update_symbolic_meta(self) -> None:
        if recorder is None:
            return
        extra_meta = getattr(recorder, "extra_meta", None)
        if extra_meta is None:
            extra_meta = {}
            recorder.extra_meta = extra_meta
        threshold = getattr(self, "symbolic_path_threshold", None)
        disabled_at = getattr(self, "symbolic_disabled_at_path_len", None)
        extra_meta["symbolic_path_threshold"] = threshold
        extra_meta["symbolic_disabled_at_path_len"] = disabled_at
        extra_meta["symbolic_disabled"] = disabled_at is not None

    def explore(
            self, modpath, all_args={}, /, *, root='.', funcname=None,
            max_iterations=0, single_timeout=15, total_timeout=900,
            deadcode=set(), include_exception=False, lib=None, single_coverage=True,
            file_as_total=False, concolic_dict={}, solve_order_stack=False,
            norm=False, limit_change_range=None,
            model_path=None, input_for_shap=None, background_dataset_for_shap=None,idx=None, shap_value_pre_calculated = False,
            collect_constraints_with: Literal['stack', 'queue', 'priority_queue'] = 'priority_queue',
            popped_log_attack_mode: str = "unknown"):

        self.model_path = model_path
        self.modpath = modpath
        self.input_for_shap = input_for_shap
        self.background_dataset_for_shap = background_dataset_for_shap
        self.idx = idx
        self.funcname = funcname
        self.single_timeout = single_timeout
        self.total_timeout = total_timeout
        self.include_exception = include_exception
        self.deadcode = deadcode
        self.lib = lib
        self.file_as_total = file_as_total
        self.normalize = norm
        self.solve_order_stack = solve_order_stack
        self.limit_change_range = limit_change_range
        self.shap_value_pre_calculated = shap_value_pre_calculated
        self.popped_log_attack_mode = popped_log_attack_mode
        self.constraints_collection_type: Literal['stack',
                                                'queue', 'priority_queue'] = collect_constraints_with
        if self.constraints_collection_type == 'priority_queue':
            self.comparator = ShapValuesComparator(
                model_path= self.model_path ,
                background_dataset = self.background_dataset_for_shap,
                input = np.expand_dims(self.input_for_shap, axis=0),
                idx = self.idx,
                shap_value_pre_calculated = self.shap_value_pre_calculated)
            self.compare = self.comparator.compare
        else:
            self.comparator = None
        self.constraints_to_solve = create_constraint_searcher(self.constraints_collection_type)

        if self.funcname is None:
            self.funcname = self.modpath.split('.')[-1]

        self.__init2__()
        if hasattr(self, "extra_meta") and isinstance(self.extra_meta, dict):
            recorder.extra_meta.update(self.extra_meta)

        recorder.input_shape = get_in_dict_shape(all_args)
        self.original_args = all_args.copy()

        self.root = os.path.abspath(root)
        self.target_file = os.path.join(
            self.root, self.modpath.replace('./', ''))

        self.single_coverage = bool(single_coverage and ENABLE_COVERAGE_LOGGING)
        if ENABLE_COVERAGE_LOGGING:
            if self.single_coverage:
                self.coverage = coverage.Coverage(
                    data_file=None, include=[self.target_file]
                )
            else:
                self.coverage = coverage.Coverage(
                    data_file=None,
                    source=[self.root],
                    omit=["**/__pycache__/**", "**/.venv/**"],
                )
        else:
            self.coverage = None
        if self.lib:
            sys.path.insert(0, os.path.abspath(self.lib))
        sys.path.insert(0, self.root)  # ; sys.path.insert(0, file_dir)
        # file_dir = os.path.abspath(os.path.dirname(os.path.join(self.root, self.modpath.replace('.', '/') + '.py')))
        file_dir = os.path.abspath(os.path.dirname(os.path.join(
            self.root, self.modpath.replace('./', '') + '.py')))
        now_dir = os.getcwd()
        os.chdir(file_dir)
        self.can_use_concolic_wrapper = self._can_use_concolic_wrapper(
            self.root, self.modpath)

        deadline = time.monotonic() + self.total_timeout if self.total_timeout else None
        timed_out = False
        interrupted = False
        try:
            timed_out = self._execution_loop(
                max_iterations,
                all_args,
                concolic_dict,
                deadline=deadline,
            )
        except KeyboardInterrupt:
            interrupted = True
            log.warning("Exploration interrupted by user (idx=%s)", self.idx)
            raise
        except BaseException:
            interrupted = True
            log.exception("Exploration loop terminated unexpectedly")
        finally:
            if timed_out:
                log.info('[TOTAL TIMEOUT]: Total Timeout happened')
            self._update_symbolic_meta()
            recorder.end(
                constraint_complexity=Solver.ctr_size,
                completed=not (timed_out or interrupted),
            )

        # After finishing self._execution_loop, we can get total iteration from recorder
        iteration = recorder.total_iter

        os.chdir(now_dir)
        del sys.path[0]
        if self.lib:
            del sys.path[0]
        if self.statsdir:
            with open(self.statsdir + '/inputs.pkl', 'wb') as f:
                # store only inputs
                pickle.dump([e[0] for e in self.in_out], f)
            if self.single_coverage:
                with open(self.statsdir + '/missing_lines.txt', 'w') as f:
                    if self.file_as_total:
                        f.write(str(sorted(self.module_lines_range &
                                self.coverage_accumulated_missing_lines[self.target_file])) + '\n')
                        f.write(str(sorted(self.module_lines_range)) + '\n')
                    else:
                        f.write(str(sorted(self.function_lines_range &
                                self.coverage_accumulated_missing_lines[self.target_file])) + '\n')
                        f.write(str(sorted(self.function_lines_range)) + '\n')
            with open(self.statsdir + '/smt.csv', 'w') as f:
                f.write(',number,time\n')
                f.write(
                    f'sat,{Solver.stats["sat_number"]},{Solver.stats["sat_time"]}\n')
                f.write(
                    f'unsat,{Solver.stats["unsat_number"]},{Solver.stats["unsat_time"]}\n')
                f.write(
                    f'otherwise,{Solver.stats["otherwise_number"]},{Solver.stats["otherwise_time"]}\n')

        return iteration, recorder

    def _clone_primitive_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
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

    def _predict_validation(self, inputs: Dict[str, Any]) -> Any:
        primitive_inputs = self._clone_primitive_inputs(inputs)
        val_args, val_kwargs = self._complete_primitive_arguments(
            self.validation_execute,
            primitive_inputs,
        )
        return self.validation_execute(*val_args, **val_kwargs)

    def _validate_sat_candidate(self, inputs: Dict[str, Any]) -> bool:
        attack_label = self._predict_validation(inputs)
        if recorder.original_label != attack_label:
            log.warning(
                "[RESULT_CHANGE] Original result %s differs from validated candidate %s",
                recorder.original_label,
                attack_label,
            )
            recorder.find_adversarial_input(inputs, attack_label)
            return True
        return False

    def _record_result(self, inputs: Dict[str, Any], result: Any) -> bool:
        """Retain search execution results without using them for attack validation."""
        self.previous_result = result
        return True

    def _one_execution(self, all_args, concolic_dict):
        """Run one concolic+primitive execution pair to advance exploration."""
        execution_executor = getattr(self, "_execution_executor", None)
        if execution_executor is None:
            execution_executor = LegacyConcolicExecutor(self)
            self._execution_executor = execution_executor
        primitive_inputs = self._clone_primitive_inputs(all_args)
        # primitive input arguments "all_args" may be modified here.
        result = execution_executor.run_concolic(all_args, concolic_dict)
        # We don't measure coverage in the primitive mode under the non-single coverage setting.
        if not self.single_coverage:
            # .copy() is important! Think why.
            self.in_out.append((all_args.copy(), result))
            return self._record_result(all_args, result)
        # we must measure the coverage in the primitive mode since self.constraints_to_solve would become unpicklable if measured in the concolic mode
        answer = execution_executor.run_primitive(primitive_inputs)

        if self.Timeout not in (result, answer):
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

        # Note only in the self.single_coverage mode does the program go here.
        if self.file_as_total:
            s = (self.module_lines_range -
                self.deadcode) & self.coverage_accumulated_missing_lines[self.target_file]
        else:
            s = (self.function_lines_range -
                self.deadcode) & self.coverage_accumulated_missing_lines[self.target_file]
        log.info(
            f"Not Covered Yet: {self.target_file} {sorted(s) if s else '{}'}")

        return self._record_result(all_args, result)
        # return s # continue iteration only if the target file / function coverage is not full yet.

    def _one_execution_concolic(self, all_args: dict, concolic_dict: dict):
        r2, s2 = multiprocessing.Pipe()

        def child_process():
            # very important to prevent the later primitive mode from using concolic objects imported here...
            sys.dont_write_bytecode = True
            child_pid = os.getpid()
            updated_args = None
            envelope = None
            try:
                prepare()
                self.path.__init__()
                self._reset_symbolic_guard()
                if self.can_use_concolic_wrapper:
                    import libct.wrapper
                else:
                    import libct

                ccc_args, ccc_kwargs = self._get_concolic_arguments(
                    execute, all_args, concolic_dict)
                updated_args = dict(all_args)
                result = self.Exception
                try:
                    result = libct.utils.unwrap(
                        func_timeout.func_timeout(
                            self.single_timeout,
                            execute,
                            args=ccc_args,
                            kwargs=ccc_kwargs,
                        )
                    )
                    log.info(f"Return: {result}")
                    envelope = self._build_child_ok_envelope(
                        pid=child_pid,
                        updated_args=updated_args,
                        result=result,
                        constraint_payload=(
                            Constraint.global_constraints,
                            self.constraints_to_solve,
                            self.path,
                            self.symbolic_disabled_at_path_len,
                        ),
                    )
                except func_timeout.FunctionTimedOut:
                    result = self.Timeout
                    message = (
                        f"Timeout (soft) for: {all_args} >> ./pyct.py -r '{self.root}' "
                        f"'{self.modpath}' -s {self.funcname} {{}} --lib '{self.lib}' "
                        "--include_exception"
                    )
                    log.error(message)
                    if self.statsdir:
                        with open(self.statsdir + '/exception.txt', 'a') as f:
                            f.write(message + "\n")
                    envelope = self._build_child_event_envelope(
                        pid=child_pid,
                        updated_args=updated_args,
                        result=result,
                        event_type="soft_timeout",
                        message=message,
                    )
                except Exception as e:
                    message = (
                        f"Exception for: {all_args} >> ./pyct '{self.root}' "
                        f"'{self.modpath}' -s {self.funcname} {{}} -m 20 --lib "
                        f"'{self.lib}' --include_exception"
                    )
                    log.exception(message)
                    if self.statsdir:
                        with open(self.statsdir + '/exception.txt', 'a') as f:
                            f.write(message + "\n")
                            f.write(f"{e}\n")
                    envelope = self._build_child_event_envelope(
                        pid=child_pid,
                        updated_args=updated_args,
                        result=self.Exception,
                        event_type="target_exception",
                        message=str(e) or message,
                        error_class=e.__class__.__name__,
                    )
            except Exception as exc:
                traceback_text = traceback.format_exc()
                envelope = self._build_child_error_envelope(
                    pid=child_pid,
                    updated_args=updated_args,
                    error_type="child_unexpected_error",
                    phase="execute",
                    message=str(exc) or exc.__class__.__name__,
                    error_class=exc.__class__.__name__,
                    traceback_text=traceback_text,
                )

            try:
                s2.send(envelope)
            except Exception as exc:
                fallback = self._build_child_error_envelope(
                    pid=child_pid,
                    updated_args=updated_args,
                    error_type="constraint_transfer_failure",
                    phase="transfer",
                    message=(
                        "failed to send child envelope to parent: "
                        f"{exc.__class__.__name__}: {exc}"
                    ),
                    error_class=exc.__class__.__name__,
                    traceback_text=traceback.format_exc(),
                )
                try:
                    s2.send(fallback)
                except Exception:
                    pass

        process = multiprocessing.Process(target=child_process)
        process.start()

        try:
            envelope = self._receive_child_envelope(r2, process, self.single_timeout + 5)
            result = self._handle_child_envelope(all_args, envelope)
        finally:
            r2.close()
            s2.close()
            if process.is_alive():
                process.kill()
            process.join(timeout=0.1)
        return result

    def _one_execution_primitive(self, primitive_inputs):
        """Execute the target without symbolic wrappers to collect coverage."""
        r1, s1 = multiprocessing.Pipe()
        r2, s2 = multiprocessing.Pipe()
        r0, s0 = multiprocessing.Pipe()

        def child_process():
            sys.dont_write_bytecode = True  # same reason mentioned in the concolic mode
            self.coverage.start()
            # module = get_module_from_rootdir_and_modpath(self.root, self.modpath)
            # execute = get_function_from_module_and_funcname(module, self.funcname)
            # Note inspect.getsourcelines(module)[1] always returns 0, which is not the fact.
            s1.send(set(self.coverage.analysis(self.target_file)[1]) & set(
                range(1, 1+len(inspect.getsourcelines(module)[0]))))
            s1.send(set(self.coverage.analysis(self.target_file)[1]) & set(range(inspect.getsourcelines(
                execute)[1], inspect.getsourcelines(execute)[1] + len(inspect.getsourcelines(execute)[0]))))
            pri_args, pri_kwargs = self._complete_primitive_arguments(
                execute, primitive_inputs)
            answer = self.Exception
            try:
                answer = func_timeout.func_timeout(
                    self.single_timeout, execute, args=pri_args, kwargs=pri_kwargs)
            except func_timeout.FunctionTimedOut:
                answer = self.Timeout
            except:
                pass
            self.coverage.stop()
            self.coverage_data.update(self.coverage.get_data())
            for file in self.coverage_data.measured_files():  # "file" is absolute here.
                _, _, missing_lines, _ = self.coverage.analysis(file)
                if file not in self.coverage_accumulated_missing_lines:
                    self.coverage_accumulated_missing_lines[file] = set(
                        missing_lines)
                else:
                    self.coverage_accumulated_missing_lines[file] &= set(
                        missing_lines)
            ###################################### Communication Section ######################################
            # just a notification to the parent process that we're going to send data
            s0.send(0)
            try:
                s1.send(answer)
            except:
                answer = self.Unpicklable
                s1.send(answer)
            if self.include_exception or (answer is not self.Exception):
                s2.send(
                    (self.coverage_data, self.coverage_accumulated_missing_lines))
            else:
                s2.send(self.Exception)
        process = multiprocessing.Process(target=child_process)
        process.start()
        self.module_lines_range = r1.recv()
        self.function_lines_range = r1.recv()
        if not r0.poll(self.single_timeout + 5):
            answer = self.Timeout
        else:
            answer = r1.recv()
            if (t := r2.recv()) is not self.Exception:
                (self.coverage_data, self.coverage_accumulated_missing_lines) = t
        if self.target_file not in self.coverage_accumulated_missing_lines:
            self.coverage_accumulated_missing_lines[self.target_file] = self.module_lines_range
        self.in_out.append((primitive_inputs.copy(), answer))
        r1.close()
        s1.close()
        r2.close()
        s2.close()
        r0.close()
        s0.close()
        if process.is_alive():
            process.kill()
        return answer

    @classmethod
    def _complete_primitive_arguments(cls, func, all_args):
        prim_args = []
        prim_kwargs = {}
        for v in inspect.signature(func).parameters.values():
            if v.kind in (inspect.Parameter.VAR_POSITIONAL, ):
                continue  # ignore *args
            elif v.kind in (inspect.Parameter.VAR_KEYWORD, ):
                # only support 1 **kwargs and no other arguments.
                assert len(inspect.signature(func).parameters.values()) == 1
                prim_kwargs = all_args.copy()
                break
            else:
                value = v.default if (
                    t := all_args[v.name]) is cls.LazyLoading else t
                if v.kind is inspect.Parameter.KEYWORD_ONLY:
                    prim_kwargs[v.name] = value
                # v.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                else:
                    prim_args.append(value)

        return prim_args, prim_kwargs

    def _get_concolic_arguments(self, func, prim_args: dict[str, any], concolic_dict: dict):
        ccc_args = []
        ccc_kwargs = {}
        self.concolic_name_list = []

        for v in inspect.signature(func).parameters.values():
            if v.kind in (inspect.Parameter.VAR_POSITIONAL, ):
                # do not support *args currently
                prim_args.pop(v.name, None)
                continue

            elif v.kind in (inspect.Parameter.VAR_KEYWORD, ):
                # only support 1 **kwargs and no other arguments.
                assert len(inspect.signature(func).parameters.values()) == 1

                for name, value in prim_args.items():
                    ccc_obj_name: str = name + '_VAR'  # '_VAR' is used to avoid name collision
                    self.concolic_flag_dict[ccc_obj_name] = 0
                    if type(value) in (bool, float, int, str) and concolic_dict.get(name, 0):
                        value = ConcolicObject(value, ccc_obj_name, self)
                        self.concolic_name_list.append(ccc_obj_name)
                        self.concolic_flag_dict[ccc_obj_name] = 1

                    ccc_kwargs[name] = value

                break
            else:
                if v.name in prim_args:
                    value = prim_args[v.name]
                else:
                    has_value = False
                    if (t := v.annotation) is not inspect._empty:
                        try:
                            value = t()
                            # may raise TypeError: Cannot instantiate ...
                            has_value = True
                        except:
                            pass
                    if not has_value:
                        if (t := v.default) is not inspect._empty:
                            # default values may also be wrapped
                            value = unwrap(t)
                        else:
                            value = ''
                    prim_args[v.name] = value if type(value) in (
                        bool, float, int, str) else self.LazyLoading

                self.concolic_flag_dict[v.name+'_VAR'] = 0
                if type(value) in (bool, float, int, str) and concolic_dict.get(v.name, 1):
                    #print(v.name + " set to ConcolicObj")
                    # '_VAR' is used to avoid name collision
                    value = ConcolicObject(value, v.name + '_VAR', self)
                    self.concolic_name_list.append(v.name + '_VAR')
                    self.concolic_flag_dict[v.name+'_VAR'] = 1

                if v.kind is inspect.Parameter.KEYWORD_ONLY:
                    ccc_kwargs[v.name] = value
                # v.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                else:
                    ccc_args.append(value)

        if not self.var_to_types:  # remain unchanged once determined
            for (k, v) in prim_args.items():
                k += '_VAR'  # '_VAR' is used to avoid name collision
                if type(v) is bool:
                    self.var_to_types[k] = 'Bool'
                elif type(v) is float:
                    self.var_to_types[k] = 'Real'
                elif type(v) is int:
                    self.var_to_types[k] = 'Int'
                elif type(v) is str:
                    self.var_to_types[k] = 'String'
                else:
                    pass  # for some default values that cannot be concolic-ized
        log.info(
            "[WRAP] idx=%s concolic=%s primitive=%s queue_type=%s",
            self.idx,
            len(self.concolic_name_list),
            len(prim_args),
            self.constraints_collection_type,
        )

        return ccc_args, ccc_kwargs

    def _can_use_concolic_wrapper(self, root, modpath):
        r, s = multiprocessing.Pipe()
        if os.fork() == 0:  # child process
            try:
                import libct.wrapper
                # module = get_module_from_rootdir_and_modpath(root, modpath)
                s.send(1)
            except:
                s.send(0)
            os._exit(os.EX_OK)
        os.wait()
        ans = r.recv()
        r.close()
        s.close()
        return ans

    def coverage_statistics(self):
        total_lines = 0
        executed_lines = 0
        missing_lines = {}
        for file in self.coverage_data.measured_files():
            executable_lines = set(self.coverage.analysis(file)[1])
            if file == self.target_file and not self.file_as_total:
                executable_lines &= self.function_lines_range
            m_lines = self.coverage_accumulated_missing_lines[file]
            if file == self.target_file and not self.file_as_total:
                m_lines &= self.function_lines_range
            total_lines += len(set(executable_lines))
            # Do not use "len(set(self.coverage_data.lines(file)))" here!!!
            executed_lines += len(set(executable_lines)) - len(m_lines)
            if m_lines:
                missing_lines[file] = m_lines
            # print(file, executed_lines, total_lines)
        if self.statsdir:
            with open(self.statsdir + '/coverage.txt', 'w') as f:
                f.write("{}/{} ({:.2%})\n".format(executed_lines, total_lines,
                        (executed_lines/total_lines) if total_lines > 0 else 0))
        return total_lines, executed_lines, missing_lines

    def print_coverage(self):
        total_lines, executed_lines, missing_lines = self.coverage_statistics()
        ratio = (executed_lines/total_lines) if total_lines > 0 else 0
        log.info(
            "Line coverage %s/%s (%.2f%%)",
            executed_lines,
            total_lines,
            ratio * 100,
        )
        if missing_lines and self.single_coverage:
            for file, lines in missing_lines.items():
                log.debug("Missing lines for %s: %s", file, sorted(lines))
    
    def push_constraint(self, constraint: Constraint, position):
        shap_value = 0.0
        if position is not None and hasattr(self, "comparator") and self.comparator is not None:
            layer_number, indices = position
            shap_value = self.comparator.get_shap_influence(layer_number, indices)
        layer_number = None
        index_summary = "None"
        if isinstance(position, tuple) and len(position) == 2:
            layer_number = position[0]
            index_summary = summarize_indices(position[1])
        path_len = getattr(constraint, "height", None)
        if self.constraints_collection_type == 'priority_queue':
            score, _assert_num = self._compute_priority_score(shap_value, constraint)
            item = ConstraintWorkItem.from_constraint(
                constraint,
                position=position,
                shap_value=shap_value,
                score=score,
            )
            self._push_work_item(item)
            if recorder is not None:
                current_size = len(self.constraints_to_solve)
                recorder.queue_last = current_size
                if current_size > getattr(recorder, "queue_max", 0):
                    recorder.queue_max = current_size
            if self.constraint_log_enabled:
                log.info(
                    "[PUSH] idx=%s layer=%s position=%s shap=%.3e path_len=%s queue_size=%s",
                    self.idx,
                    layer_number,
                    index_summary,
                    abs(shap_value),
                    path_len,
                    len(self.constraints_to_solve),
                )
        else:
            item = ConstraintWorkItem.from_constraint(
                constraint,
                position=position,
                shap_value=shap_value,
            )
            self._push_work_item(item)
            if recorder is not None:
                current_size = len(self.constraints_to_solve)
                recorder.queue_last = current_size
                if current_size > getattr(recorder, "queue_max", 0):
                    recorder.queue_max = current_size
            if self.constraint_log_enabled:
                log.info(
                    "[PUSH] idx=%s queue=%s position=%s shap=%.3e path_len=%s total=%s",
                    self.idx,
                    self.constraints_collection_type,
                    summarize_position(position),
                    abs(shap_value),
                    path_len,
                    len(self.constraints_to_solve),
                )

    def _compute_priority_score(self, shap_value: float, constraint: Constraint) -> tuple[float, int]:
        if self.shap_score_alpha is None:
            raise ValueError(
                "shap_score_alpha is required when collect_constraints_with='priority_queue'; pass via --score-alpha"
            )
        path_len = int(getattr(constraint, "height", 0) or 0)
        alpha = self.shap_score_alpha
        score = (1 - alpha) * math.log10(abs(shap_value) + self.SHAP_SCORE_EPS)
        score -= alpha * math.log10(path_len + 1)
        return score, path_len

    def _push_work_item(self, item: ConstraintWorkItem) -> None:
        worklist = self.constraints_to_solve
        if isinstance(worklist, Searcher):
            worklist.push(item)
        elif self.constraints_collection_type == 'priority_queue':
            heapq.heappush(
                worklist,
                (-item.score, item.constraint.id, item.position, item.constraint, item.shap_value),
            )
        else:
            worklist.append(item.constraint)

    def _log_pop_event(
        self,
        *,
        queue_mode: str,
        remaining: int,
        layer: Any = None,
        indices: Any = None,
        shap_value: float | None = None,
        path_len: int | None = None,
    ) -> None:
        if not getattr(self, "constraint_log_enabled", False):
            return
        attack_mode = getattr(self, "popped_log_attack_mode", "unknown")
        sample_idx = getattr(self, "idx", "unknown")
        if queue_mode == "priority":
            log.info(
                "[POP] idx=%s attack=%s queue=%s layer=%s position=%s shap=%s path_len=%s remaining=%d",
                sample_idx,
                attack_mode,
                queue_mode,
                layer,
                summarize_indices(indices),
                shap_value,
                path_len,
                remaining,
            )
        else:
            log.info(
                "[POP] idx=%s attack=%s queue=%s path_len=%s remaining=%d",
                sample_idx,
                attack_mode,
                queue_mode,
                path_len,
                remaining,
            )

    def pop_constraint(self) -> Constraint:
        if isinstance(self.constraints_to_solve, Searcher):
            item = self.constraints_to_solve.pop()
            constraint = item.constraint
            if self.constraints_collection_type == 'priority_queue':
                position = item.position
                layer_number = None
                indices = None
                if isinstance(position, tuple) and len(position) == 2:
                    layer_number, indices = position
                self._log_pop_event(
                    queue_mode="priority",
                    remaining=len(self.constraints_to_solve),
                    layer=layer_number,
                    indices=indices,
                    shap_value=f"{abs(item.shap_value):.3e}",
                    path_len=getattr(constraint, "height", None),
                )
                log.debug(
                    "Popped constraint from queue (position=%s shap_value=%s constraint_id=%s)",
                    summarize_position(position),
                    item.shap_value,
                    constraint.id,
                )
                return constraint, item.shap_value, position
            queue_mode = "stack" if self.constraints_collection_type == 'stack' else "queue"
            self._log_pop_event(
                queue_mode=queue_mode,
                remaining=len(self.constraints_to_solve),
                path_len=getattr(constraint, "height", None),
            )
            return constraint
        if self.constraints_collection_type =='stack':
            constraint = self.constraints_to_solve.pop()
            self._log_pop_event(
                queue_mode="stack",
                remaining=len(self.constraints_to_solve),
                path_len=getattr(constraint, "height", None),
            )
            return constraint
        elif self.constraints_collection_type == 'queue':
            constraint = self.constraints_to_solve.popleft()
            self._log_pop_event(
                queue_mode="queue",
                remaining=len(self.constraints_to_solve),
                path_len=getattr(constraint, "height", None),
            )
            return constraint
        elif self.constraints_collection_type == 'priority_queue':
            score, constraint_id, position, constraint, shap_value = heapq.heappop(self.constraints_to_solve)
            layer_number, indices = position
            self._log_pop_event(
                queue_mode="priority",
                remaining=len(self.constraints_to_solve),
                layer=layer_number,
                indices=indices,
                shap_value=f"{abs(shap_value):.3e}",
                path_len=getattr(constraint, "height", None),
            )
            log.debug(
                "Popped constraint from queue (position=%s shap_value=%s constraint_id=%s)",
                summarize_position(position),
                shap_value,
                constraint_id,
            )
            return constraint, shap_value, position


def clear_global_context():
    """Reset module level state so the GC can reclaim large objects between runs."""
    global module, execute, recorder
    module = None
    execute = None
    recorder = None
    Constraint.global_constraints.clear()
    PathToConstraint.root_constraint = None
    gc.collect()
