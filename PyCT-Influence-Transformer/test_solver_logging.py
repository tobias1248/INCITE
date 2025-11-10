#!/usr/bin/env python3
"""Lightweight tests for libct.solver logging integration."""

from __future__ import annotations

import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import func_timeout as _func_timeout  # type: ignore
except ModuleNotFoundError:
    class _StubFuncTimeoutModule:
        class exceptions:
            class FunctionTimedOut(TimeoutError):
                ...

        @staticmethod
        def func_timeout(_timeout, func, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    import sys

    sys.modules["func_timeout"] = _StubFuncTimeoutModule()
else:
    del _func_timeout

import libct.solver as solver


class SolverLoggingTests(unittest.TestCase):
    """Exercise solver logging without invoking a real SMT solver."""

    def setUp(self) -> None:
        solver.Solver.stats = {
            "sat_number": 0,
            "sat_time": 0,
            "unsat_number": 0,
            "unsat_time": 0,
            "otherwise_number": 0,
            "otherwise_time": 0,
        }
        solver.Solver.smtdir = None
        solver.Solver.store = None
        solver.Solver.iter = 1
        solver.Solver.iter_count = 1
        solver.Solver.cnt = 1
        solver.Solver.cmd = ["cvc5", "--tlimit=1000"]
        solver.Solver.limit_change_range = None
        solver.Solver.norm = None
        solver.Solver.ctr_size = {
            "type": [],
            "time": [],
            "byte": [],
            "assert_num": [],
            "assert_len": [],
        }

    def test_find_model_from_constraint_logs_status(self) -> None:
        engine = SimpleNamespace(
            concolic_flag_dict={"pixel": 1},
            shap_value_pre_calculated=True,
            model_path="model/mock.h5",
            var_to_types={"x_VAR": "Real"},
            popped_log_attack_mode="priority_queue",
            concolic_name_list=["x_VAR"],
        )
        fake_subprocess = SimpleNamespace(stdout=b"sat\n((x_VAR 1))\n")

        with mock.patch("libct.solver.func_timeout.func_timeout", return_value="(check-sat)\n"):
            with mock.patch("libct.solver.subprocess.run", return_value=fake_subprocess):
                with mock.patch.object(solver.Solver, "_resolve_constraint_log_path", return_value=Path("inline.log")):
                    with mock.patch.object(solver.Solver, "_append_constraint_log"):
                        with self.assertLogs("ct.solver", level=logging.INFO) as captured:
                            result = solver.Solver.find_model_from_constraint(
                                engine,
                                constraint=object(),
                                shap_value=0.42,
                                position=(1, 2, 3),
                                idx=1,
                                ori_args={"x": 0.0},
                            )

        self.assertEqual(result, {"x": 1.0})
        self.assertTrue(
            any("SMT solver status" in message for message in captured.output),
            msg="Expected solver status log entry to be emitted.",
        )

    def test_expr_validation_returns_engine_on_sat(self) -> None:
        sentinel_engine = object()
        fake_completed = SimpleNamespace(stdout=b"sat\n")

        with mock.patch("libct.solver.Concolic.find_engine_in_expr", return_value=sentinel_engine):
            with mock.patch("libct.solver.Predicate.get_formula_shallow", return_value="x"):
                with mock.patch("libct.solver.py2smt", return_value="1"):
                    with mock.patch("libct.solver.subprocess.run", return_value=fake_completed):
                        solver.Solver.safety = 1
                        result = solver.Solver._expr_has_engines_and_equals_value("expr", 1)

        self.assertIs(result, sentinel_engine)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
