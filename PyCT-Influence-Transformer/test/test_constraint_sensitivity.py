"""Offline checks for ranking predicates using their dependence on Global X."""
from types import SimpleNamespace

import pytest

from libct.concolic.float import ConcolicFloat
from libct.searcher.constraint_sensitivity import SensitivityEstimator


def estimator(**kwargs):
    options = dict(variable="X", x=0.0, lower=-0.2, upper=0.2)
    options.update(kwargs)
    return SensitivityEstimator(**options)


@pytest.mark.parametrize("relation", [">", ">=", "<", "<=", "="])
def test_linear_margin_has_expected_sensitivity_weight(relation):
    # available displacement .2 * slope 2 / margin .8 = .5
    weight = estimator().estimate([relation, ["+", ["*", "2", "X"], "0.8"], "0"])
    assert weight == pytest.approx(0.5, abs=1e-7)


def test_two_symbolic_sides_use_difference_and_constant_is_insensitive():
    score = estimator().estimate([">", ["+", ["*", "3", "X"], "0.8"], "X"])
    assert score == pytest.approx(0.5, abs=1e-7)
    assert estimator().estimate([">", "3", "2"]) == 0.0


def test_weight_is_capped_and_near_zero_margin_is_stable():
    assert estimator().estimate([">", "X", "0"]) == 1.0
    assert estimator().estimate([">", "X", "0.001"]) == 1.0


def test_bound_uses_one_sided_difference():
    score = estimator(x=0.2).estimate([">", ["+", "X", "0.8"], "0"])
    assert score == pytest.approx(0.4, abs=1e-7)


def test_piecewise_kink_uses_larger_one_sided_slope():
    piecewise = ["ite", [">", "X", "0"], ["*", "4", "X"], "X"]
    assert estimator().estimate([">", ["+", piecewise, "2"], "0"]) == pytest.approx(0.4, abs=1e-7)


def test_ite_does_not_evaluate_unused_invalid_branch():
    safe = ["ite", [">", "X", "1"], ["/", "1", "0"], ["+", "X", "1"]]
    assert estimator().estimate([">", safe, "0"]) == pytest.approx(0.2, abs=1e-7)


def test_real_concolic_expression_is_used_without_arithmetic_dispatch(monkeypatch):
    value = ConcolicFloat(0.0, "X", SimpleNamespace(symbolic_enabled=True))
    monkeypatch.setattr(ConcolicFloat, "_bin_op", lambda *args: pytest.fail("symbolic arithmetic executed"))
    assert estimator().estimate([">", ["+", value, "0.8"], "0"]) == pytest.approx(0.25, abs=1e-7)
    assert value.expr == "X"


@pytest.mark.parametrize("expr", [
    [">", "other_variable", "0"],
    [">", ["unrecognised", "X"], "0"],
    [">", ["/", "X", "0"], "0"],
    [">", "nan", "0"],
    [">", "inf", "0"],
])
def test_unsupported_or_nonfinite_expressions_report_fallback(expr):
    instance = estimator()
    assert instance.estimate(expr) is None
    assert instance.stats["attempted"] == 1
    assert instance.stats["fallback"] == 1
    assert instance.stats["fallback_reasons"]


def test_deep_expression_is_iterative_and_memo_is_shared():
    expr = "X"
    for _ in range(2500):
        expr = ["+", expr, "0.001"]
    instance = estimator()
    assert instance.estimate([">", expr, "0"]) == pytest.approx(0.08, abs=1e-7)
    previous_nodes = instance.stats["nodes"]
    assert instance.estimate([">", expr, "1"]) == pytest.approx(0.2 / 1.5, abs=1e-7)
    assert instance.stats["nodes"] - previous_nodes < 20


def test_cycle_node_and_wall_guards_fall_back(monkeypatch):
    cycle = ["+", "X", "0"]
    cycle[2] = cycle
    assert estimator().estimate([">", cycle, "0"]) is None
    assert estimator(max_nodes=1).estimate([">", ["+", "X", "1"], "0"]) is None
    instance = estimator(wall_budget=0.01)
    ticks = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr("libct.searcher.constraint_sensitivity.time.perf_counter", lambda: next(ticks))
    assert instance.estimate([">", "X", "0"]) is None
    assert instance.stats["fallback_reasons"] == {"time_budget": 1}


def test_independent_forward_estimators_do_not_reuse_other_x_values():
    expr = [">", ["+", ["*", "X", "X"], "0.1"], "0"]
    assert estimator(x=0.1).estimate(expr) > estimator(x=0.0).estimate(expr)
