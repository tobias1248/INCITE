from __future__ import annotations

import func_timeout
import inspect
import multiprocessing
import sys
from typing import Any, Dict


class PrimitiveExecutionRunner:
    """Subprocess runner for one primitive target execution and coverage pass."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def run(self, primitive_inputs: Dict[str, Any]) -> Any:
        """Execute the target without symbolic wrappers to collect coverage."""
        r1, s1 = multiprocessing.Pipe()
        r2, s2 = multiprocessing.Pipe()
        r0, s0 = multiprocessing.Pipe()

        process = multiprocessing.Process(target=self._child_process, args=(s1, s2, s0, primitive_inputs))
        process.start()
        self._engine.module_lines_range = r1.recv()
        self._engine.function_lines_range = r1.recv()
        if not r0.poll(self._engine.single_timeout + 5):
            answer = self._engine.Timeout
        else:
            answer = r1.recv()
            if (payload := r2.recv()) is not self._engine.Exception:
                (
                    self._engine.coverage_data,
                    self._engine.coverage_accumulated_missing_lines,
                ) = payload
        if self._engine.target_file not in self._engine.coverage_accumulated_missing_lines:
            self._engine.coverage_accumulated_missing_lines[
                self._engine.target_file
            ] = self._engine.module_lines_range
        r1.close()
        s1.close()
        r2.close()
        s2.close()
        r0.close()
        s0.close()
        if process.is_alive():
            process.kill()
        return answer

    def _child_process(
        self,
        range_conn: Any,
        payload_conn: Any,
        ready_conn: Any,
        primitive_inputs: Dict[str, Any],
    ) -> None:
        sys.dont_write_bytecode = True  # same reason mentioned in the concolic mode
        coverage = self._engine.coverage
        target_file = self._engine.target_file
        module = self._engine._get_module()
        execute = self._engine._get_execute()

        coverage.start()
        # Note inspect.getsourcelines(module)[1] always returns 0, which is not the fact.
        range_conn.send(
            set(coverage.analysis(target_file)[1])
            & set(range(1, 1 + len(inspect.getsourcelines(module)[0])))
        )
        range_conn.send(
            set(coverage.analysis(target_file)[1])
            & set(
                range(
                    inspect.getsourcelines(execute)[1],
                    inspect.getsourcelines(execute)[1] + len(inspect.getsourcelines(execute)[0]),
                )
            )
        )
        pri_args, pri_kwargs = self._engine._complete_primitive_arguments(execute, primitive_inputs)
        answer = self._engine.Exception
        try:
            answer = func_timeout.func_timeout(
                self._engine.single_timeout,
                execute,
                args=pri_args,
                kwargs=pri_kwargs,
            )
        except func_timeout.FunctionTimedOut:
            answer = self._engine.Timeout
        except Exception:
            pass
        coverage.stop()
        self._engine.coverage_data.update(coverage.get_data())
        for file in self._engine.coverage_data.measured_files():  # "file" is absolute here.
            _, _, missing_lines, _ = coverage.analysis(file)
            if file not in self._engine.coverage_accumulated_missing_lines:
                self._engine.coverage_accumulated_missing_lines[file] = set(missing_lines)
            else:
                self._engine.coverage_accumulated_missing_lines[file] &= set(missing_lines)
        ###################################### Communication Section ######################################
        # just a notification to the parent process that we're going to send data
        ready_conn.send(0)
        try:
            range_conn.send(answer)
        except Exception:
            answer = self._engine.Unpicklable
            range_conn.send(answer)
        if self._engine.include_exception or (answer is not self._engine.Exception):
            payload_conn.send(
                (
                    self._engine.coverage_data,
                    self._engine.coverage_accumulated_missing_lines,
                )
            )
        else:
            payload_conn.send(self._engine.Exception)
