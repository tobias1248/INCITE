#!/usr/bin/env python3
"""Unit tests covering logging behavior inside libct.explore."""

from __future__ import annotations

import heapq
import sys
import unittest
from types import ModuleType, SimpleNamespace


try:
    import coverage as _coverage  # type: ignore
except ModuleNotFoundError:
    class _StubCoverageData:
        def update(self, _data=None) -> None: ...
        def measured_files(self):
            return []

    class _StubCoverage:
        def __init__(self, *args, **kwargs) -> None:
            self._data = _StubCoverageData()

        def start(self) -> None: ...
        def stop(self) -> None: ...
        def analysis(self, *_args, **_kwargs):
            return (None, [], [], None)

        def get_data(self):
            return self._data

    sys.modules["coverage"] = SimpleNamespace(
        Coverage=_StubCoverage,
        CoverageData=_StubCoverageData,
    )
else:
    del _coverage

try:
    import shap as _shap  # type: ignore
except ModuleNotFoundError:
    sys.modules["shap"] = SimpleNamespace()
else:
    del _shap

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

    sys.modules["func_timeout"] = _StubFuncTimeoutModule()
else:
    del _func_timeout

try:
    import cv2 as _cv2  # type: ignore
except ModuleNotFoundError:
    class _StubCV2:
        @staticmethod
        def imwrite(_path, _img):
            return True

    sys.modules["cv2"] = _StubCV2()
else:
    del _cv2

if "explainability.shap_calculator" not in sys.modules:
    shap_module = ModuleType("explainability.shap_calculator")

    class _StubComparator:
        def __init__(self, *args, **kwargs) -> None:
            ...

        def compare(self, *_args, **_kwargs):
            return 0

        def get_shap_influence(self, *_args, **_kwargs):
            return 0.0

    shap_module.ShapValuesComparator = _StubComparator  # type: ignore[attr-defined]
    sys.modules["explainability.shap_calculator"] = shap_module

import libct.explore as explore


class _DummyRecorder:
    """Minimal recorder stub satisfying the interfaces used in tests."""

    def __init__(self) -> None:
        self.gen_constraint = []
        self.total_iter = 0
        self.original_label = None

    def start(self) -> None: ...
    def iter_start(self, _solver) -> None: ...
    def execution_start(self) -> None: ...
    def execution_end(self) -> None: ...
    def iter_end(self, _stats, _value) -> None: ...
    def solve_constr_start(self) -> None: ...
    def solve_constr_end(self) -> None: ...
    def first_execution_end(self) -> None: ...
    def save_stats_dict(self) -> None: ...
    def no_ctr_to_solve(self) -> None: ...
    def total_timeout(self) -> None: ...
    def end(self, constraint_complexity=None) -> None: ...
    def find_adversarial_input(self, *_args, **_kwargs) -> None: ...
    def record_reference_prediction(self, *_args, **_kwargs) -> None: ...


class ExploreLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        explore.recorder = _DummyRecorder()
        explore.Solver.stats = {
            "sat_number": 0,
            "sat_time": 0,
            "unsat_number": 0,
            "unsat_time": 0,
            "otherwise_number": 0,
            "otherwise_time": 0,
        }

    def _make_engine(self) -> explore.ExplorationEngine:
        engine = explore.ExplorationEngine.__new__(explore.ExplorationEngine)
        engine.reference_execute = lambda **_data: 0
        engine.normalize = None
        engine.limit_change_range = None
        engine.constraints_to_solve = []
        engine.idx = 0
        engine.original_args = {}
        engine.only_first_forward = True
        engine.single_coverage = True
        engine.in_out = []
        engine.path = None
        engine.deadcode = set()
        engine.module_lines_range = set()
        engine.function_lines_range = set()
        engine.coverage_accumulated_missing_lines = {"test.py": set()}
        engine.target_file = "test.py"
        engine.file_as_total = False
        engine.can_use_concolic_wrapper = False
        engine.previous_result = None
        engine.original_args = {}
        return engine

    def test_execution_loop_logs_when_no_constraints(self) -> None:
        engine = self._make_engine()
        engine._one_execution = lambda *args, **kwargs: True  # type: ignore[attr-defined]

        with self.assertLogs("ct.explore", level="INFO") as captured:
            result = engine._execution_loop(0, {}, {})

        self.assertEqual(result, 0)
        self.assertTrue(
            any("FIRST_NO_CONSTR" in entry for entry in captured.output),
            msg="Expected FIRST_NO_CONSTR log to be emitted",
        )

    def test_one_execution_logs_timeout_warning(self) -> None:
        engine = self._make_engine()
        engine.Timeout = explore.ExplorationEngine.Timeout
        engine._one_execution_concolic = lambda *args, **kwargs: engine.Timeout  # type: ignore[attr-defined]
        engine._one_execution_primitive = lambda *args, **kwargs: "ok"  # type: ignore[attr-defined]

        with self.assertLogs("ct.explore", level="WARNING") as captured:
            cont = engine._one_execution({}, {})

        self.assertTrue(cont)
        self.assertTrue(
            any("SINGLE_TIMEOUT" in entry for entry in captured.output),
            msg="Expected SINGLE_TIMEOUT warning log",
        )

    def test_pop_constraint_logs_position(self) -> None:
        engine = self._make_engine()
        engine.constraints_collection_type = "priority_queue"
        engine.constraints_to_solve = []
        constraint = object()
        heapq.heappush(engine.constraints_to_solve, (-0.5, 42, ("layer", 1), constraint, 0.5))

        with self.assertLogs("ct.explore", level="DEBUG") as captured:
            popped_constraint, shap_value, position = engine.pop_constraint()

        self.assertIs(popped_constraint, constraint)
        self.assertEqual(position, ("layer", 1))
        self.assertTrue(
            any("Popped constraint" in entry for entry in captured.output),
            msg="Expected log about popped constraint position",
        )

    def test_push_and_pop_priority_constraint_uses_modular_searcher(self) -> None:
        from libct.constraint import Constraint
        from libct.searcher import create_constraint_searcher

        Constraint.global_constraints.clear()
        engine = self._make_engine()
        engine.constraints_collection_type = "priority_queue"
        engine.constraints_to_solve = create_constraint_searcher("priority_queue")
        engine.comparator = None
        engine.shap_score_alpha = 0.5
        engine.constraint_log_enabled = False
        constraint = Constraint(None, None, height=2)

        engine.push_constraint(constraint, ("layer", (1, 2)))
        popped_constraint, shap_value, position = engine.pop_constraint()

        self.assertIs(popped_constraint, constraint)
        self.assertEqual(shap_value, 0.0)
        self.assertEqual(position, ("layer", (1, 2)))
        self.assertEqual(len(engine.constraints_to_solve), 0)


if __name__ == "__main__":
    unittest.main()
