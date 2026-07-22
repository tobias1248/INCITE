from __future__ import annotations

import shutil
import signal
import subprocess
import threading
from types import SimpleNamespace

import pytest

import libct.predicate as predicate_module
from libct.predicate import Predicate
from libct.solver import FormulaBuildTimedOut, Solver, _formula_build_deadline


def _repeated_expression():
    return [
        "+",
        ["*", "x_VAR", "12345678901234567890"],
        ["*", "y_VAR", "98765432109876543210"],
    ]


def test_predicate_uses_let_for_profitable_common_subexpression() -> None:
    shared = _repeated_expression()
    predicate = Predicate([">", ["+", shared, shared], "0"], True)

    raw = predicate.get_formula()
    compact, stats = predicate.get_formula_with_sharing()

    assert "(let ((_pyct_cse_0 " in compact
    assert compact.count("_pyct_cse_0") == 3
    assert len(compact) < len(raw)
    assert stats == {
        "binding_count": 1,
        "bytes_before": len(raw),
        "bytes_after": len(compact),
    }


def test_predicate_keeps_original_formula_when_sharing_is_not_profitable() -> None:
    predicate = Predicate([">", "x_VAR", "0"], False)

    compact, stats = predicate.get_formula_with_sharing()

    assert compact == predicate.get_formula()
    assert "(let " not in compact
    assert stats["binding_count"] == 0
    assert stats["bytes_before"] == stats["bytes_after"] == len(compact)


def test_predicate_counts_deep_shared_dag_without_expanding_occurrences() -> None:
    shared = _repeated_expression()
    for _ in range(80):
        shared = ["+", shared, shared]

    compact, stats = Predicate([">", shared, "0"], True).get_formula_with_sharing()

    assert stats["binding_count"] > 0
    assert "fallback_reason" not in stats
    assert len(compact) < 50_000


def test_predicate_falls_back_to_raw_when_cse_budget_is_exceeded(monkeypatch) -> None:
    monkeypatch.setattr(
        predicate_module._CommonSubexpressionSerializer,
        "_MAX_UNIQUE_NODES",
        2,
    )
    predicate = Predicate(
        [">", ["+", ["*", "x_VAR", "2"], ["*", "y_VAR", "3"]], "0"],
        True,
    )

    compact, stats = predicate.get_formula_with_sharing()

    assert compact == predicate.get_formula()
    assert stats["binding_count"] == 0
    assert "unique-node budget exceeded" in stats["fallback_reason"]


def test_formula_build_deadline_does_not_leave_a_worker_thread() -> None:
    thread_count = threading.active_count()

    with pytest.raises(FormulaBuildTimedOut):
        with _formula_build_deadline(0.02):
            while True:
                pass

    assert threading.active_count() == thread_count
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


@pytest.mark.skipif(shutil.which("cvc5") is None, reason="cvc5 is not installed")
def test_shared_formula_has_same_solver_result_as_original() -> None:
    shared = _repeated_expression()
    predicate = Predicate([">", ["+", shared, shared], "0"], True)
    compact, _stats = predicate.get_formula_with_sharing()
    prefix = "\n".join(
        [
            "(set-logic ALL)",
            "(declare-const x_VAR Real)",
            "(declare-const y_VAR Real)",
        ]
    )

    def solve(assertion):
        completed = subprocess.run(
            ["cvc5", "--lang", "smt", "--quiet"],
            input=f"{prefix}\n{assertion}\n(check-sat)\n".encode(),
            capture_output=True,
            check=True,
        )
        return completed.stdout.decode().splitlines()[0]

    assert solve(compact) == solve(predicate.get_formula()) == "sat"


@pytest.mark.skipif(shutil.which("cvc5") is None, reason="cvc5 is not installed")
def test_nested_let_bindings_have_valid_dependency_scope() -> None:
    inner = _repeated_expression()
    outer = ["+", inner, inner]
    predicate = Predicate([">", ["+", outer, outer], "0"], True)

    compact, stats = predicate.get_formula_with_sharing()
    formulas = "\n".join(
        [
            "(set-logic ALL)",
            "(declare-const x_VAR Real)",
            "(declare-const y_VAR Real)",
            compact,
            "(check-sat)",
        ]
    )
    completed = subprocess.run(
        ["cvc5", "--lang", "smt", "--quiet"],
        input=formulas.encode(),
        capture_output=True,
        check=True,
    )

    assert stats["binding_count"] >= 2
    assert completed.stdout.decode().splitlines()[0] == "sat"


def test_solver_builder_reports_query_compression_stats() -> None:
    shared = _repeated_expression()
    predicate = Predicate([">", ["+", shared, shared], "0"], True)
    constraint = SimpleNamespace(get_all_asserts=lambda: [predicate])
    engine = SimpleNamespace(
        concolic_name_list=["x_VAR", "y_VAR"],
        var_to_types={"x_VAR": "Real", "y_VAR": "Real"},
    )
    previous_norm = Solver.norm
    previous_limit = Solver.limit_change_range
    previous_sharing = Solver.formula_sharing_mode
    Solver.norm = False
    Solver.limit_change_range = None
    Solver.formula_sharing_mode = "let_cse"
    try:
        formulas = Solver._build_formulas_from_constraint(engine, constraint, {})
    finally:
        Solver.norm = previous_norm
        Solver.limit_change_range = previous_limit
        Solver.formula_sharing_mode = previous_sharing

    stats = Solver._last_formula_sharing_stats
    assert "(let ((_pyct_cse_0 " in formulas
    assert stats["cse_binding_count"] == 1
    assert stats["cse_assertion_count"] == 1
    assert stats["cse_fallback_count"] == 0
    assert stats["formula_sharing_mode"] == "let_cse"
    assert stats["query_bytes_after_cse"] < stats["query_bytes_before_cse"]
