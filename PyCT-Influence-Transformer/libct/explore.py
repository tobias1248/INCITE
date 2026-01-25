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
from libct.path import PathToConstraint
from libct.solver import Solver, _ensure_smtlib2_logger
from libct.utils import ConcolicObject, unwrap, get_in_dict_shape
from libct.record import ConcolicTestRecorder
import cProfile
import shap
import numpy as np

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from types import ModuleType
from libct.constraint import Constraint
from libct.shapInfl import ShapValuesComparator
import heapq
from collections import deque


log = logging.getLogger("ct.explore")
ENABLE_COVERAGE_LOGGING = False
# The original limit is not enough in some special cases.
sys.setrecursionlimit(1000000)
module = None
execute = None
recorder = None


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
                only_first_forward: bool,
                shap_score_alpha: Optional[float] = None):
        global module, execute

        module = module_
        execute = execute_

        self.save_dir = save_dir
        self.input_name = input_name
        self.only_first_forward = only_first_forward
        if shap_score_alpha is None:
            raise ValueError("shap_score_alpha is required; pass via --score-alpha")
        self.shap_score_alpha = float(shap_score_alpha)
        self.verbose = verbose
        self.logfile = logfile

        self.normalize = None
        self.__init2__()
        self.statsdir = statsdir
        if self.statsdir:
            os.system(f"rm -rf '{statsdir}'")
            os.system(f"mkdir -p '{statsdir}'")
        Solver.set_basic_configurations(solver, timeout, safety, store, smtdir, constraint_build_timeout, solver_run_timeout)
        _ensure_smtlib2_logger()

    def __init2__(self):
        global recorder
        recorder = ConcolicTestRecorder(self.save_dir, self.input_name)

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
        self._one_execution(all_args, concolic_dict)
        recorder.execution_end()
        recorder.iter_end(Solver.stats, 0)
        recorder.gen_constraint.append(len(self.constraints_to_solve))
        recorder.first_execution_end()

        # first self.previous_result is the original label

        recorder.original_label = self.previous_result
        recorder.save_original_input(all_args)
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
                        break

            recorder.solve_constr_end()
            solve_constr_num = solve_constr_num - \
                len(self.constraints_to_solve)

            if timed_out:
                recorder.total_timeout()
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
            self.constraints_to_solve = []
        else:
            self.constraints_to_solve = deque()

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

    def _record_result(self, inputs: Dict[str, Any], result: Any) -> bool:
        """Update cached prediction and persist adversarial example if it changed."""
        if result in (self.Timeout, self.Exception, self.Unpicklable):
            self.previous_result = result
            return True
        if self.previous_result is not None and self.previous_result != result:
            log.warning(
                "[RESULT_CHANGE] Previous result %s differs from current %s",
                self.previous_result,
                result,
            )
            recorder.find_adversarial_input(inputs, result)
            return False

        self.previous_result = result
        return True

    def _one_execution(self, all_args, concolic_dict):
        """Run one concolic+primitive execution pair to advance exploration."""
        primitive_inputs = self._clone_primitive_inputs(all_args)
        # primitive input arguments "all_args" may be modified here.
        result = self._one_execution_concolic(all_args, concolic_dict)
        # We don't measure coverage in the primitive mode under the non-single coverage setting.
        if not self.single_coverage:
            # .copy() is important! Think why.
            self.in_out.append((all_args.copy(), result))
            return self._record_result(all_args, result)
        # we must measure the coverage in the primitive mode since self.constraints_to_solve would become unpicklable if measured in the concolic mode
        answer = self._one_execution_primitive(primitive_inputs)

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
        r1, s1 = multiprocessing.Pipe()
        r2, s2 = multiprocessing.Pipe()
        r3, s3 = multiprocessing.Pipe()
        r0, s0 = multiprocessing.Pipe()

        def child_process():
            # very important to prevent the later primitive mode from using concolic objects imported here...
            sys.dont_write_bytecode = True
            prepare()
            self.path.__init__()
            # log.info("Inputs: " + str(all_args))
            if self.can_use_concolic_wrapper:
                import libct.wrapper
            else:
                import libct

            # module = get_module_from_rootdir_and_modpath(self.root, self.modpath)
            # execute = get_function_from_module_and_funcname(module, self.funcname)

            # primitive input arguments "all_args" may be modified here.
            ccc_args, ccc_kwargs = self._get_concolic_arguments(
                execute, all_args, concolic_dict)

            s1.send((all_args, self.var_to_types,
                    self.concolic_name_list, self.concolic_flag_dict))
            result = self.Exception
            try:
                
                result = libct.utils.unwrap(func_timeout.func_timeout(
                    self.single_timeout, execute, args=ccc_args, kwargs=ccc_kwargs))
        
                log.info(f"Return: {result}")
            except func_timeout.FunctionTimedOut:
                result = self.Timeout
                # ; traceback.print_exc()
                log.error(
                    f"Timeout (soft) for: {all_args} >> ./pyct.py -r '{self.root}' '{self.modpath}' -s {self.funcname} {{}} --lib '{self.lib}' --include_exception")
                if self.statsdir:
                    with open(self.statsdir + '/exception.txt', 'a') as f:
                        f.write(
                            f"Timeout (soft) for: {all_args} >> ./pyct.py -r '{self.root}' '{self.modpath}' -s {self.funcname} {{}} --lib '{self.lib}' --include_exception\n")
            except Exception as e:
                log.exception(
                    f"Exception for: {all_args} >> ./pyct '{self.root}' '{self.modpath}' -s {self.funcname} {{}} -m 20 --lib '{self.lib}' --include_exception",
                )
                if self.statsdir:
                    with open(self.statsdir + '/exception.txt', 'a') as f:
                        f.write(
                            f"Exception for: {all_args} >> ./pyct '{self.root}' '{self.modpath}' -s {self.funcname} {{}} -m 20 --lib '{self.lib}' --include_exception\n")
                        f.write(f"{e}\n")
            ###################################### Communication Section ######################################
            # just a notification to the parent process that we're going to send data
            s0.send(0)
            try:
                s2.send(result)
                
                
            except:
                s2.send(self.Unpicklable)

            try:
                s3.send((Constraint.global_constraints,
                        self.constraints_to_solve, self.path))
            except Exception:
                log.exception(
                    "Failed to send constraints back to parent process due to unpicklable objects",
                )
                # may fail if they contain some unpicklable objects
                s3.send(self.Unpicklable)

        process = multiprocessing.Process(target=child_process)
        process.start()
        (all_args2, self.var_to_types, self.concolic_name_list, self.concolic_flag_dict) =\
            r1.recv()
        r1.close()
        s1.close()
        all_args.clear()
        all_args.update(all_args2)  # update the parameter directly

        if not r0.poll(self.single_timeout + 5):
            result = self.Timeout
            
            log.error(
                f"Timeout (hard) for: {all_args} >> ./pyct.py -r '{self.root}' '{self.modpath}' -s {self.funcname} {{}} --lib '{self.lib}' --include_exception")
            if self.statsdir:
                with open(self.statsdir + '/exception.txt', 'a') as f:
                    f.write(
                        f"Timeout (hard) for: {all_args} >> ./pyct.py -r '{self.root}' '{self.modpath}' -s {self.funcname} {{}} --lib '{self.lib}' --include_exception\n")
        else:
            result = r2.recv()

            if (t := r3.recv()) is not self.Unpicklable:
                Constraint.global_constraints, self.constraints_to_solve, self.path = t
            else:
                log.warning(
                    "Constraints payload contains unpicklable objects; skipping constraint transfer",
                )
                self.constraints_to_solve = deque()

        r2.close()
        s2.close()
        r3.close()
        s3.close()
        r0.close()
        s0.close()
        if process.is_alive():
            process.kill()
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
                    if type(value) in (bool, float, int, str) and concolic_dict[name]:
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
        path_len = getattr(constraint, "height", None)
        if self.constraints_collection_type == 'priority_queue':
            score, _assert_num = self._compute_priority_score(shap_value, constraint)
            heapq.heappush(
                self.constraints_to_solve,
                (-score, constraint.id, position, constraint, abs(shap_value)),
            )
            if recorder is not None:
                current_size = len(self.constraints_to_solve)
                recorder.queue_last = current_size
                if current_size > recorder.queue_max:
                    recorder.queue_max = current_size
            log.info(
                "[PUSH] idx=%s layer=%s position=%s shap=%.3e path_len=%s queue_size=%s",
                self.idx,
                position[0],
                position[1],
                abs(shap_value),
                path_len,
                len(self.constraints_to_solve),
            )
        else:
            self.constraints_to_solve.append(constraint)
            if recorder is not None:
                current_size = len(self.constraints_to_solve)
                recorder.queue_last = current_size
                if current_size > recorder.queue_max:
                    recorder.queue_max = current_size
            log.info(
                "[PUSH] idx=%s queue=%s position=%s shap=%.3e path_len=%s total=%s",
                self.idx,
                self.constraints_collection_type,
                position,
                abs(shap_value),
                path_len,
                len(self.constraints_to_solve),
            )

    def _compute_priority_score(self, shap_value: float, constraint: Constraint) -> tuple[float, int]:
        path_len = int(getattr(constraint, "height", 0) or 0)
        alpha = self.shap_score_alpha
        score = (1 - alpha) * math.log10(abs(shap_value) + self.SHAP_SCORE_EPS)
        score -= alpha * math.log10(path_len + 1)
        return score, path_len

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
        attack_mode = getattr(self, "popped_log_attack_mode", "unknown")
        sample_idx = getattr(self, "idx", "unknown")
        if queue_mode == "priority":
            log.info(
                "[POP] idx=%s attack=%s queue=%s layer=%s position=%s shap=%s path_len=%s remaining=%d",
                sample_idx,
                attack_mode,
                queue_mode,
                layer,
                indices,
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
                position,
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
