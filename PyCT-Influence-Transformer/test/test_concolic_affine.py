"""Exact compact runtime forms preserve the legacy float/SMT contract."""

from fractions import Fraction
import gc
import operator
import pickle
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest

from libct.concolic import Concolic
from libct.concolic.affine import AffineForm, query_statistics
from libct.concolic.float import ConcolicFloat
from libct.predicate import Predicate
from libct.solver import Solver
from libct.utils import ConcolicObject, unwrap


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    monkeypatch.setattr(Solver, "safety", 0, raising=False)
    monkeypatch.setenv("PYCT_SYMBOLIC_REPRESENTATION", "compact-affine")


def _engine():
    branches = []
    engine = SimpleNamespace(symbolic_enabled=True, branches=branches)
    engine.path = SimpleNamespace(add_branch=lambda branch: branches.append(branch))
    return engine


def _bits(value):
    return struct.pack("!d", float(unwrap(value)))


def _equivalent(left, right, names=("x_VAR",)):
    executable = shutil.which("cvc5")
    if executable is None:
        pytest.skip("cvc5 is required for exact runtime SMT equivalence")
    query = "\n".join([
        "(set-logic ALL)",
        *("(declare-const {} Real)".format(name) for name in names),
        "(assert (distinct {} {}))".format(
            Predicate.get_formula_deep(left), Predicate.get_formula_deep(right)),
        "(check-sat)",
    ])
    result = subprocess.run([executable, "--lang=smt2", "--quiet"], input=query,
                            text=True, capture_output=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "unsat", result.stdout + result.stderr


OPERATIONS = [operator.add, lambda x, y: y + x, operator.mul,
              lambda x, y: y * x, operator.truediv]


@pytest.mark.parametrize("seed", [-0.0, 0.0, -0.125, 0.1, 1e-300, float("inf"), float("nan")])
@pytest.mark.parametrize("scalar", [-0.0, 0.0, -1.5, 0, 1, 2 ** 54 + 1, False, True])
@pytest.mark.parametrize("operation", OPERATIONS, ids=["add", "radd", "mul", "rmul", "div"])
def test_concrete_bits_shortcuts_and_exceptions_match_legacy(monkeypatch, seed, scalar, operation):
    results = []
    for mode in ("legacy", "compact-affine"):
        monkeypatch.setenv("PYCT_SYMBOLIC_REPRESENTATION", mode)
        source = ConcolicFloat(seed, "x_VAR", _engine())
        try:
            result = operation(source, scalar)
        except Exception as exc:
            results.append((type(exc), str(exc)))
        else:
            results.append((_bits(result), type(result), result is source,
                            getattr(result, "engine", None) is source.engine))
    assert results[0] == results[1]


@pytest.mark.parametrize("operation", OPERATIONS + [
    lambda x, y: x.__add__(y), lambda x, y: x.__radd__(y),
    lambda x, y: x.__mul__(y), lambda x, y: x.__rmul__(y),
    lambda x, y: x.__truediv__(y),
])
def test_huge_integer_errors_and_direct_method_results_match_legacy(monkeypatch, operation):
    results = []
    for mode in ("legacy", "compact-affine"):
        monkeypatch.setenv("PYCT_SYMBOLIC_REPRESENTATION", mode)
        x = ConcolicFloat(0.2, "x_VAR", _engine())
        try:
            value = operation(x, 10 ** 400)
        except Exception as exc:
            results.append((type(exc), str(exc)))
        else:
            results.append((type(value), value is NotImplemented))
    assert results[0] == results[1]


@pytest.mark.parametrize("method", ["__radd__", "__rmul__", "__rtruediv__"])
def test_custom_operand_overloads_are_executed_once(monkeypatch, method):
    runs = []
    for mode in ("legacy", "compact-affine"):
        monkeypatch.setenv("PYCT_SYMBOLIC_REPRESENTATION", mode)
        calls = []

        class Operand:
            pass

        def calculate(self, other):
            calls.append(other)
            return 0.625

        setattr(Operand, method, calculate)
        x = ConcolicFloat(0.2, "x_VAR", _engine())
        operation = {"__radd__": operator.add, "__rmul__": operator.mul,
                     "__rtruediv__": operator.truediv}[method]
        result = operation(x, Operand())
        runs.append((_bits(result), type(result), result.engine is x.engine, calls))
    assert runs[0] == runs[1]
    assert runs[0][-1] == [0.2]


@pytest.mark.parametrize("seed", [-0.0, 0.12345678901234567, 2.0])
@pytest.mark.parametrize("chain", [
    lambda x: ((x * 0.12345678901234567 + 0.0000000000000006) / 3) * -2,
    lambda x: x * 0.1 + x * -0.1 + 0.12345678901234567,
    lambda x: (x + 1e-17) * 0.3333333333333333,
    lambda x: ((x + -0.75) / (2 ** 54 + 1)) * 7,
    lambda x: x * x + 0.25,
    lambda x: x / (x + 1.5),
])
def test_exact_smt_equivalence_including_rounding_cancellation_and_fallback(monkeypatch, seed, chain):
    monkeypatch.setenv("PYCT_SYMBOLIC_REPRESENTATION", "legacy")
    legacy = chain(ConcolicFloat(seed, "x_VAR", _engine()))
    monkeypatch.setenv("PYCT_SYMBOLIC_REPRESENTATION", "compact-affine")
    compact = chain(ConcolicFloat(seed, "x_VAR", _engine()))
    assert _bits(compact) == _bits(legacy)
    _equivalent(legacy, compact)


def test_coefficients_follow_legacy_decimal_spelling_not_binary_float():
    result = ConcolicFloat(0.2, "x_VAR", _engine()) * 0.1 + 0.12345678901234567
    assert result.__dict__["_affine_form"] == AffineForm(
        "x_VAR", Fraction(1, 10), Fraction("0.123456789012346"))


def test_compact_chain_defers_expression_and_releases_operand_history():
    engine = _engine()
    value = ConcolicFloat(0.2, "x_VAR", engine)
    references = [weakref.ref(value)]
    for _ in range(1000):
        value = value * 0.5 + 0.25
        references.append(weakref.ref(value))
    gc.collect()
    assert all(reference() is None for reference in references[:-1])
    assert "expr" not in value.__dict__
    assert "_affine_expression" not in value.__dict__
    expression = value.expr
    assert value.expr is expression
    assert not any(isinstance(part, Concolic) for part in expression)
    assert value.engine is engine


@pytest.mark.parametrize("kind,reason", [
    ("nonlinear", "nonlinear_multiplication"),
    ("denominator", "symbolic_denominator"),
    ("variables", "multiple_variables"),
    ("ite", "unsupported_expression"),
    ("budget", "rational_digit_budget"),
])
def test_unsupported_forms_keep_original_ast(monkeypatch, kind, reason):
    engine = _engine()
    x = ConcolicFloat(0.2, "x_VAR", engine)
    y = ConcolicFloat(0.3, "y_VAR", engine)
    if kind == "nonlinear":
        result = x * x
    elif kind == "denominator":
        result = x / (x + 1)
    elif kind == "variables":
        result = x + y
    elif kind == "ite":
        expression = ["ite", [">", x, "0"], x, "0"]
        result = ConcolicFloat(0.2, expression, engine)
        assert result.expr is expression
    else:
        monkeypatch.setattr("libct.concolic.affine.MAX_RATIONAL_DIGITS", 2)
        result = x * 0.123456789
    assert "_affine_form" not in result.__dict__
    assert result.__dict__["_compact_fallback_reason"] == reason
    assert isinstance(result.expr, list)
    assert any(part is x for part in result.expr)


def test_mixed_engines_keep_ast_and_first_owner():
    first, second = _engine(), _engine()
    x = ConcolicFloat(0.2, "x_VAR", first)
    y = ConcolicFloat(0.3, "y_VAR", second)
    result = x + y
    assert result.engine is first
    assert result.expr[1] is x and result.expr[2] is y
    assert result.__dict__["_compact_fallback_reason"] == "mixed_engines"


@pytest.mark.parametrize("other", [2, True])
def test_symbolic_nonfloat_operands_preserve_expression_and_values(monkeypatch, other):
    results = []
    for mode in ("legacy", "compact-affine"):
        monkeypatch.setenv("PYCT_SYMBOLIC_REPRESENTATION", mode)
        engine = _engine()
        x = ConcolicFloat(0.2, "x_VAR", engine)
        y = ConcolicObject(other, "other_VAR", engine)
        results.append((x + y, x * y, x / y))
    for legacy, compact in zip(*results):
        assert _bits(legacy) == _bits(compact)
        assert Predicate.get_formula_deep(legacy) == Predicate.get_formula_deep(compact)


def test_symbolic_division_guard_is_evaluated_once_and_matches_legacy(monkeypatch):
    traces = []
    guard_calls = []

    def observed_guard(value, other):
        guard_calls.append((float(value), other))
        return float.__ne__(value, other)

    monkeypatch.setattr(ConcolicFloat, "__ne__", observed_guard)
    for mode in ("legacy", "compact-affine"):
        monkeypatch.setenv("PYCT_SYMBOLIC_REPRESENTATION", mode)
        engine = _engine()
        x = ConcolicFloat(0.2, "x_VAR", engine)
        denominator = x * 0 + 2
        result = x / denominator
        traces.append((engine.branches, result))
    assert guard_calls == [(2.0, 0), (2.0, 0)]
    assert traces[0][0] == traces[1][0] == []
    _equivalent(traces[0][1], traces[1][1])


@pytest.mark.parametrize("seed,divisor", [(0.2, 0.5), (1e308, 0.001), (0.2, 0.0)])
def test_division_finite_overflow_and_zero_evaluate_guard_once(monkeypatch, seed, divisor):
    runs = []
    calls = []

    def observed_guard(value, other):
        calls.append((float(value), other))
        return float.__ne__(value, other)

    monkeypatch.setattr(ConcolicFloat, "__ne__", observed_guard)
    for mode in ("legacy", "compact-affine"):
        monkeypatch.setenv("PYCT_SYMBOLIC_REPRESENTATION", mode)
        engine = _engine()
        x = ConcolicFloat(seed, "x_VAR", engine)
        denominator = ConcolicFloat(divisor, str(divisor), engine)
        try:
            result = x / denominator
        except Exception as exc:
            runs.append((type(exc), str(exc)))
        else:
            runs.append((_bits(result), type(result)))
    assert runs[0] == runs[1]
    assert calls == [(divisor, 0), (divisor, 0)]


@pytest.mark.parametrize("coefficient,compact", [(99, True), (999, False)])
def test_rational_digit_budget_boundary_is_exact(monkeypatch, coefficient, compact):
    monkeypatch.setattr("libct.concolic.affine.MAX_RATIONAL_DIGITS", 3)
    x = ConcolicFloat(0.2, "x_VAR", _engine())
    result = x * coefficient
    assert ("_affine_form" in result.__dict__) is compact
    if not compact:
        assert result.__dict__["_compact_fallback_reason"] == "rational_digit_budget"
        assert result.expr[1] is x


def test_python_integer_string_limit_falls_back_before_lazy_serialization():
    from libct.smt_shared import serialize_shared_assertions

    if not hasattr(sys, "set_int_max_str_digits"):
        pytest.skip("Python does not expose an integer-to-string conversion limit")
    previous_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        value = ConcolicFloat(0.2, "x_VAR", _engine())
        fallback_reasons = []
        for _ in range(650):
            value = value * 0.1
            reason = value.__dict__.get("_compact_fallback_reason")
            if reason:
                fallback_reasons.append(reason)
        assert fallback_reasons and fallback_reasons[0] == "rational_digit_budget"
        assert "_affine_form" not in value.__dict__
        assert isinstance(value.expr, list)
        expression = [">=", value, "0"]
        raw = Predicate(expression, True).get_formula()
        shared = serialize_shared_assertions([expression], {"x_VAR": "Real"}).formula
        assert "x_VAR" in raw and "x_VAR" in shared
        assert raw.startswith("(assert ") and shared.startswith("(assert ")
    finally:
        sys.set_int_max_str_digits(previous_limit)


def test_safety_mode_bypasses_compaction_and_keeps_validation(monkeypatch):
    calls = []
    monkeypatch.setattr(Solver, "safety", 1)
    monkeypatch.setattr(Solver, "cmd", ["fake-cvc5"], raising=False)

    def fake_run(*args, **kwargs):
        calls.append(kwargs["input"])
        return SimpleNamespace(stdout=b"sat\n")

    monkeypatch.setattr("libct.solver.subprocess.run", fake_run)
    x = ConcolicFloat(0.2, "x_VAR", _engine())
    result = x * 0.75 + 0.1
    validated = ConcolicFloat(float(result), result.expr)
    assert len(calls) == 1
    assert all(b"(check-sat)" in formula for formula in calls)
    assert "_affine_form" not in result.__dict__
    assert result.__dict__["_compact_fallback_reason"] == "safety_validation"
    assert validated.engine is x.engine


def test_disabled_engine_returns_primitive_and_no_new_branches():
    engine = _engine()
    x = ConcolicFloat(0.2, "x_VAR", engine)
    engine.symbolic_enabled = False
    assert type(x * 2 + 0.1) is float
    assert bool(x > 0)
    assert engine.branches == []


def test_pickle_and_legacy_state_dicts_preserve_expr_without_owner():
    x = ConcolicFloat(0.2, "x_VAR", _engine()) * 0.75 + 0.1
    predicate = x > 0.2
    restored = pickle.loads(pickle.dumps(predicate))
    assert restored.engine is None
    assert Predicate.get_formula_deep(restored) == Predicate.get_formula_deep(predicate)
    assert _bits(restored.expr[1]) == _bits(x)
    legacy = float.__new__(ConcolicFloat, -0.0)
    legacy.__setstate__({"engine": None, "value": "0.000000000000000",
                         "expr": ["+", "x_VAR", "1"], "formula": None})
    assert legacy.expr == ["+", "x_VAR", "1"]
    assert _bits(legacy) == _bits(-0.0)
    legacy.expr = "2.0"
    assert legacy.expr == "2.0"


def test_statistics_only_count_retained_query_values_and_handle_shared_roots():
    engine = _engine()
    discarded = ConcolicFloat(0.2, "unused_VAR", engine) * 2
    x = ConcolicFloat(0.2, "x_VAR", engine) * 0.75
    fallback = x * x
    stats = query_statistics([[">", fallback, "0"], ["<", x, "1"]])
    assert stats["compact_nodes"] == 1
    assert stats["fallback_reasons"] == {"nonlinear_multiplication": 1}
    assert not stats["truncated"]
    assert discarded.__dict__["_affine_form"].variable == "unused_VAR"
