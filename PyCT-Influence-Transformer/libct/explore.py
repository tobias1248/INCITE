from __future__ import annotations
import coverage
import gc
import logging
import multiprocessing
import os
import pickle
import sys
import time
from libct.path import PathToConstraint
from libct.solver import Solver, _ensure_smtlib2_logger
from libct.utils import get_in_dict_shape
from libct.record import ConcolicTestRecorder
from libct.executor import CandidateExecutionRunner, ConcolicArgumentBuilder
from libct.executor.child_protocol import ChildProtocol, ConstraintTransferError
from libct.executor.concolic import ConcolicExecutionRunner, prepare_child_environment
from libct.executor.primitive import PrimitiveExecutionRunner
from libct.searcher import ConstraintScheduler, create_constraint_searcher
import cProfile
import shap
import numpy as np

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from types import ModuleType
from libct.constraint import Constraint
from explainability.shap_calculator import ShapValuesComparator


log = logging.getLogger("ct.explore")
ENABLE_COVERAGE_LOGGING = False
# The original limit is not enough in some special cases.
sys.setrecursionlimit(1000000)
module = None
execute = None
recorder = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def prepare():
    prepare_child_environment()


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
                trace_only: bool = False,
                shap_score_alpha: Optional[float] = None,
                symbolic_path_threshold: Optional[int] = None,
                reuse_search_result_for_validation: bool = False):
        global module, execute

        module = module_
        execute = execute_
        self.validation_execute = validation_execute_ or execute_

        self.save_dir = save_dir
        self.input_name = input_name
        self.only_first_forward = only_first_forward
        self.trace_only = bool(trace_only)
        self.branch_trace_enabled = False
        self.branch_model_sha256 = ""
        self.shap_score_alpha = (
            None if shap_score_alpha is None else float(shap_score_alpha)
        )
        self.symbolic_path_threshold = None if symbolic_path_threshold is None else int(symbolic_path_threshold)
        self.reuse_search_result_for_validation = bool(reuse_search_result_for_validation)
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
        self._candidate_execution_runner = CandidateExecutionRunner(self)
        self._concolic_argument_builder = ConcolicArgumentBuilder(self)
        self._child_protocol = ChildProtocol(self)
        self._concolic_runner = ConcolicExecutionRunner(self)
        self._primitive_runner = PrimitiveExecutionRunner(self)
        self._constraint_scheduler = ConstraintScheduler(self)

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

    def _get_recorder(self):
        return recorder

    def _get_execute(self):
        return execute

    def _get_module(self):
        return module

    def _get_child_protocol(self) -> ChildProtocol:
        protocol = getattr(self, "_child_protocol", None)
        if protocol is None:
            protocol = ChildProtocol(self)
            self._child_protocol = protocol
        return protocol

    def _get_concolic_runner(self) -> ConcolicExecutionRunner:
        runner = getattr(self, "_concolic_runner", None)
        if runner is None:
            runner = ConcolicExecutionRunner(self)
            self._concolic_runner = runner
        return runner

    def _get_primitive_runner(self) -> PrimitiveExecutionRunner:
        runner = getattr(self, "_primitive_runner", None)
        if runner is None:
            runner = PrimitiveExecutionRunner(self)
            self._primitive_runner = runner
        return runner

    def _get_candidate_execution_runner(self) -> CandidateExecutionRunner:
        runner = getattr(self, "_candidate_execution_runner", None)
        if runner is None:
            runner = CandidateExecutionRunner(self)
            self._candidate_execution_runner = runner
        return runner

    def _get_concolic_argument_builder(self) -> ConcolicArgumentBuilder:
        builder = getattr(self, "_concolic_argument_builder", None)
        if builder is None:
            builder = ConcolicArgumentBuilder(self)
            self._concolic_argument_builder = builder
        return builder

    def _get_constraint_scheduler(self) -> ConstraintScheduler:
        scheduler = getattr(self, "_constraint_scheduler", None)
        if scheduler is None:
            scheduler = ConstraintScheduler(self)
            self._constraint_scheduler = scheduler
        return scheduler

    def _mark_constraint_transfer_failure(self, reason: str) -> None:
        self._get_child_protocol().mark_constraint_transfer_failure(reason)

    def _mark_runtime_error(
        self,
        error_type: str,
        reason: str,
        *,
        phase: Optional[str] = None,
        child_pid: Optional[int] = None,
        event_type: Optional[str] = None,
    ) -> None:
        self._get_child_protocol().mark_runtime_error(
            error_type,
            reason,
            phase=phase,
            child_pid=child_pid,
            event_type=event_type,
        )

    def _record_child_event(
        self,
        event_type: str,
        message: str,
        *,
        phase: str,
        child_pid: Optional[int],
    ) -> None:
        self._get_child_protocol().record_child_event(
            event_type,
            message,
            phase=phase,
            child_pid=child_pid,
        )

    def _write_diagnostic_file(self, filename: str, contents: Optional[str]) -> None:
        self._get_child_protocol().write_diagnostic_file(filename, contents)

    def _build_child_shared_state(self, updated_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._get_child_protocol().build_child_shared_state(updated_args)

    def _build_child_ok_envelope(
        self,
        *,
        pid: int,
        updated_args: Dict[str, Any],
        result: Any,
        constraint_payload: Any,
    ) -> Dict[str, Any]:
        return self._get_child_protocol().build_child_ok_envelope(
            pid=pid,
            updated_args=updated_args,
            result=result,
            constraint_payload=constraint_payload,
        )

    def _build_child_event_envelope(
        self,
        *,
        pid: int,
        updated_args: Optional[Dict[str, Any]],
        result: Any,
        event_type: str,
        message: str,
        error_class: Optional[str] = None,
        branch_trace: Optional[Any] = None,
    ) -> Dict[str, Any]:
        return self._get_child_protocol().build_child_event_envelope(
            pid=pid,
            updated_args=updated_args,
            result=result,
            event_type=event_type,
            message=message,
            error_class=error_class,
            branch_trace=branch_trace,
        )

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
        return self._get_child_protocol().build_child_error_envelope(
            pid=pid,
            updated_args=updated_args,
            error_type=error_type,
            phase=phase,
            message=message,
            error_class=error_class,
            traceback_text=traceback_text,
        )

    def _validate_child_envelope(self, envelope: Any) -> Dict[str, Any]:
        return self._get_child_protocol().validate_child_envelope(envelope)

    def _apply_child_shared_state(self, all_args: Dict[str, Any], envelope: Dict[str, Any]) -> None:
        self._get_child_protocol().apply_child_shared_state(all_args, envelope)

    def _raise_transport_failure(
        self,
        reason: str,
        *,
        phase: str,
        child_pid: Optional[int] = None,
        details: Optional[str] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        self._get_child_protocol().raise_transport_failure(
            reason,
            phase=phase,
            child_pid=child_pid,
            details=details,
            exc=exc,
        )

    def _receive_child_envelope(
        self,
        conn: Any,
        process: multiprocessing.Process,
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        return self._get_child_protocol().receive_child_envelope(conn, process, timeout_seconds)

    def _handle_child_envelope(self, all_args: Dict[str, Any], envelope: Dict[str, Any]) -> Any:
        return self._get_child_protocol().handle_child_envelope(all_args, envelope)

    def _handle_child_envelope_deferred_constraints(
        self,
        all_args: Dict[str, Any],
        envelope: Dict[str, Any],
    ) -> Tuple[Any, Optional[Any]]:
        return self._get_child_protocol().handle_child_envelope_deferred_constraints(all_args, envelope)

    def _apply_constraint_transfer_payload(self, payload: Any) -> None:
        self._get_child_protocol().apply_constraint_transfer_payload(payload)

    def _execution_loop(self, max_iterations: int, all_args, concolic_dict, *, deadline: Optional[float] = None) -> bool:
        recorder.start()
        Solver.norm = self.normalize
        Solver.limit_change_range = self.limit_change_range
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
        self._run_initial_execution(all_args, concolic_dict)
        recorder.execution_end()
        recorder.iter_end(Solver.stats, 0)
        recorder.gen_constraint.append(len(self.constraints_to_solve))
        recorder.first_execution_end()

        if hasattr(recorder, "save_original_input"):
            recorder.save_original_input(all_args)
        self._update_symbolic_meta()
        recorder.save_stats_dict()

        if getattr(self, "trace_only", False):
            log.info(
                "[TRACE-ONLY] idx=%s branches=%s constraints=%s",
                self.idx,
                len(getattr(self.path, "branch_trace", ())),
                len(self.constraints_to_solve),
            )
            return timed_out

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
            executed_sat_candidate = False
            sat_candidate_for_execution = False
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
                        if self._candidate_execution_can_validate():
                            sat_candidate_for_execution = True
                        else:
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

            if sat_candidate_for_execution:
                gen_constr_num = len(self.constraints_to_solve)
                recorder.execution_start()
                result, constraint_payload = self._one_execution_deferred_constraints(all_args, concolic_dict)
                recorder.execution_end()
                executed_sat_candidate = True
                found_adversarial = self._search_result_changes_label(all_args, result)
                if found_adversarial:
                    recorder.gen_constraint.append(0)
                    iterations += 1
                    recorder.iter_end(Solver.stats, solve_constr_num)
                    self._update_symbolic_meta()
                    recorder.save_stats_dict()
                    break
                if constraint_payload is not None:
                    self._apply_constraint_transfer_payload(constraint_payload)
                self.in_out.append((all_args.copy(), result))
                self._record_result(all_args, result)
                gen_constr_num = len(self.constraints_to_solve) - gen_constr_num
                recorder.gen_constraint.append(gen_constr_num)

            # solve new input and use it to execute
            if not self.only_first_forward and not executed_sat_candidate:
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
        except BaseException as exc:
            interrupted = True
            log.exception("Exploration loop terminated unexpectedly")
            if recorder.extra_meta.get("status") != "error":
                recorder.mark_error(
                    "exploration_failure",
                    str(exc),
                    phase="execution_loop",
                )
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
                f.write(
                    f'invalid_model,{Solver.stats.get("invalid_model_number", 0)},0\n')

        return iteration, recorder

    def _clone_primitive_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._get_candidate_execution_runner().clone_primitive_inputs(inputs)

    def _predict_validation(self, inputs: Dict[str, Any]) -> Any:
        return self._get_candidate_execution_runner().predict_validation(inputs)

    def _validate_sat_candidate(self, inputs: Dict[str, Any]) -> bool:
        return self._get_candidate_execution_runner().validate_sat_candidate(inputs)

    def _search_result_changes_label(self, inputs: Dict[str, Any], result: Any) -> bool:
        return self._get_candidate_execution_runner().search_result_changes_label(inputs, result)

    def _record_result(self, inputs: Dict[str, Any], result: Any) -> bool:
        return self._get_candidate_execution_runner().record_result(inputs, result)

    def _candidate_execution_can_validate(self) -> bool:
        return self._get_candidate_execution_runner().candidate_execution_can_validate()

    def _is_valid_label_result(self, result: Any) -> bool:
        return self._get_candidate_execution_runner().is_valid_label_result(result)

    def _run_initial_execution(self, all_args: Dict[str, Any], concolic_dict: Dict[str, Any]) -> None:
        self._get_candidate_execution_runner().run_initial_execution(all_args, concolic_dict)

    def _one_execution_deferred_constraints(
        self,
        all_args: Dict[str, Any],
        concolic_dict: Dict[str, Any],
    ) -> Tuple[Any, Optional[Any]]:
        return self._get_candidate_execution_runner().one_execution_deferred_constraints(
            all_args,
            concolic_dict,
        )

    def _one_execution(self, all_args, concolic_dict):
        return self._get_candidate_execution_runner().one_execution(all_args, concolic_dict)

    def _one_execution_concolic(self, all_args: dict, concolic_dict: dict):
        return self._get_concolic_runner().run(all_args, concolic_dict)

    def _one_execution_concolic_deferred(self, all_args: dict, concolic_dict: dict):
        return self._get_concolic_runner().run_deferred(all_args, concolic_dict)

    def _one_execution_primitive(self, primitive_inputs):
        return self._get_primitive_runner().run(primitive_inputs)

    def _complete_primitive_arguments(self, func, all_args):
        return self._get_candidate_execution_runner().complete_primitive_arguments(func, all_args)

    def _get_concolic_arguments(self, func, prim_args: dict[str, any], concolic_dict: dict):
        return self._get_concolic_argument_builder().build(func, prim_args, concolic_dict)

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
        self._get_constraint_scheduler().push_constraint(constraint, position)

    def _compute_priority_score(self, shap_value: float, constraint: Constraint) -> tuple[float, int]:
        return self._get_constraint_scheduler()._compute_priority_score(shap_value, constraint)

    def _push_work_item(self, item) -> None:
        self._get_constraint_scheduler()._push_work_item(item)

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
        self._get_constraint_scheduler()._log_pop_event(
            queue_mode=queue_mode,
            remaining=remaining,
            layer=layer,
            indices=indices,
            shap_value=shap_value,
            path_len=path_len,
        )

    def pop_constraint(self) -> Constraint:
        return self._get_constraint_scheduler().pop_constraint()


def clear_global_context():
    """Reset module level state so the GC can reclaim large objects between runs."""
    global module, execute, recorder
    module = None
    execute = None
    recorder = None
    Constraint.global_constraints.clear()
    PathToConstraint.root_constraint = None
    gc.collect()
