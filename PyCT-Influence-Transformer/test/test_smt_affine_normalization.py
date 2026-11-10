from types import SimpleNamespace

import pytest

import libct.predicate as predicate_module
from libct.predicate import Predicate
from libct.expression_graph import MAX_ESTIMATED_BYTES
from libct.smt_affine import AffineBudgets, AffineContext, normalize_affine_comparison
from libct.solver import Solver


class _Constraint:
    def __init__(self, assertions):
        self._assertions = assertions

    def get_all_asserts(self):
        return self._assertions


def test_exact_affine_combines_terms_without_float_rounding():
    predicate = Predicate(
        [
            "<",
            [
                "+",
                ["*", "0.500000000000000", "x_VAR"],
                ["*", "0.250000000000000", "x_VAR"],
                "0.125000000000000",
            ],
            "1.000000000000000",
        ],
        True,
    )

    formula, stats = predicate.get_formula_with_exact_affine({"x_VAR": "Real"})

    assert formula == "(assert (< (+ (* (/ 3 4) x_VAR) (- (/ 7 8))) 0))"
    assert stats["applied"] is True
    assert stats["assertion_input_bytes"] == len(
        predicate.get_formula().encode("utf-8")
    )
    assert stats["assertion_output_bytes"] < stats["assertion_input_bytes"]


def test_exact_affine_preserves_false_predicate_wrapper():
    predicate = Predicate(
        ["<=", ["+", "x_VAR", "x_VAR", "x_VAR"], "1.000000000000000"],
        False,
    )

    formula, stats = predicate.get_formula_with_exact_affine({"x_VAR": "Real"})

    assert formula == "(assert (not (<= (+ (* 3 x_VAR) (- 1)) 0)))"
    assert stats["applied"] is True
    assert stats["assertion_input_bytes"] == len(
        predicate.get_formula().encode("utf-8")
    )


def test_exact_affine_parses_runtime_smt_negative_constant_strings():
    predicate = Predicate(
        [
            ">",
            [
                "+",
                ["*", "0.500000000000000", "x_VAR"],
                "(- 0.125000000000000)",
            ],
            "(- 0.500000000000000)",
        ],
        True,
    )

    formula, stats = predicate.get_formula_with_exact_affine({"x_VAR": "Real"})

    assert formula == "(assert (> (+ (* (/ 1 2) x_VAR) (/ 3 8)) 0))"
    assert stats["applied"] is True


def test_exact_affine_caches_serialization_for_immutable_predicate(monkeypatch):
    predicate = Predicate(
        ["<", ["+", "x_VAR", "x_VAR", "x_VAR"], "1.000000000000000"],
        True,
    )
    calls = 0
    original = predicate_module.normalize_affine_comparison

    def counting_normalizer(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        predicate_module,
        "normalize_affine_comparison",
        counting_normalizer,
    )

    first_formula, first_stats = predicate.get_formula_with_exact_affine(
        {"x_VAR": "Real"}
    )
    second_formula, second_stats = predicate.get_formula_with_exact_affine(
        {"x_VAR": "Real"}
    )

    assert first_formula == second_formula
    assert calls == 1
    assert first_stats["cache_hit"] is False
    assert second_stats["cache_hit"] is True


def test_exact_affine_does_not_materialize_raw_formula(monkeypatch):
    predicate = Predicate(
        ["<", ["+", "x_VAR", "x_VAR", "x_VAR"], "1.000000000000000"],
        True,
    )

    def fail_if_raw_formula_is_materialized(_expr):
        pytest.fail("successful affine normalization expanded the raw formula")

    monkeypatch.setattr(
        predicate,
        "get_formula_deep",
        fail_if_raw_formula_is_materialized,
    )

    formula, stats = predicate.get_formula_with_exact_affine({"x_VAR": "Real"})

    assert formula == "(assert (< (+ (* 3 x_VAR) (- 1)) 0))"
    assert stats["applied"] is True


@pytest.mark.parametrize(
    ("expr", "variable_types", "reason"),
    [
        (
            ["<", ["*", "x_VAR", "y_VAR"], "1.000000000000000"],
            {"x_VAR": "Real", "y_VAR": "Real"},
            "nonlinear_multiplication",
        ),
        (
            ["<", ["+", "x_VAR", "1"], "2"],
            {"x_VAR": "Int"},
            "non_real_variable",
        ),
        (
            ["and", ["<", "x_VAR", "1"], [">", "x_VAR", "0"]],
            {"x_VAR": "Real"},
            "unsupported_predicate",
        ),
    ],
)
def test_exact_affine_falls_back_for_unsupported_expressions(
    expr,
    variable_types,
    reason,
):
    predicate = Predicate(expr, True)

    formula, stats = predicate.get_formula_with_exact_affine(variable_types)

    assert formula == predicate.get_formula()
    assert stats["applied"] is False
    assert stats["fallback_reason"] == reason
    assert stats["assertion_output_bytes"] == stats["assertion_input_bytes"]


def test_exact_affine_falls_back_when_node_budget_is_exceeded():
    expr = ["<", ["+", "x_VAR", "1"], "2"]
    raw_body = Predicate.get_formula_deep(expr)

    normalization = normalize_affine_comparison(
        expr,
        {"x_VAR": "Real"},
        raw_body=raw_body,
        budgets=AffineBudgets(max_nodes=2),
    )

    assert normalization.applied is False
    assert normalization.fallback_reason == "node_budget"


def test_exact_affine_falls_back_when_rational_digit_budget_is_exceeded():
    expr = ["<", ["+", "x_VAR", "123456789"], "2"]
    raw_body = Predicate.get_formula_deep(expr)

    normalization = normalize_affine_comparison(
        expr,
        {"x_VAR": "Real"},
        raw_body=raw_body,
        budgets=AffineBudgets(max_rational_digits=5),
    )

    assert normalization.applied is False
    assert normalization.fallback_reason == "rational_digit_budget"


def test_solver_raw_mode_keeps_existing_formula(monkeypatch):
    monkeypatch.setenv("PYCT_SMT_EXPERIMENT_MODE", "raw")
    engine = SimpleNamespace(
        concolic_name_list=["x_VAR"],
        var_to_types={"x_VAR": "Real"},
    )
    predicate = Predicate(["<", ["+", "x_VAR", "x_VAR"], "1"], True)

    formula = Solver._build_formulas_from_constraint(
        engine,
        _Constraint([predicate]),
        {"x": 0.5},
    )

    assert predicate.get_formula() in formula
    assert Solver._last_smt_transform_stats == {
        "mode": "raw",
        "path_mode": "full",
        "original_assertion_count": 1,
        "retained_assertion_count": 1,
        "assertion_count": 1,
        "applied_count": 0,
        "fallback_count": 0,
        "cache_hit_count": 0,
        "fallback_reasons": {},
        "fallback_examples": {},
        "input_bytes": len(predicate.get_formula().encode("utf-8")),
        "output_bytes": len(predicate.get_formula().encode("utf-8")),
        "normalization_time_s": 0.0,
        "node_count": 0,
        "term_count": 0,
        "max_variable_count": 0,
        "max_rational_digits": 0,
    }


def test_solver_exact_affine_mode_is_default_and_records_transform_stats(monkeypatch):
    monkeypatch.delenv("PYCT_SMT_EXPERIMENT_MODE", raising=False)
    engine = SimpleNamespace(
        concolic_name_list=["x_VAR"],
        var_to_types={"x_VAR": "Real"},
    )
    predicate = Predicate(
        ["<", ["+", "x_VAR", "x_VAR", "x_VAR"], "1.000000000000000"],
        True,
    )

    formula = Solver._build_formulas_from_constraint(
        engine,
        _Constraint([predicate]),
        {"x": 0.5},
    )
    stats = Solver._last_smt_transform_stats

    assert "(assert (< (+ (* 3 x_VAR) (- 1)) 0))" in formula
    assert stats["mode"] == "exact_affine"
    assert stats["applied_count"] == 1
    assert stats["fallback_count"] == 0
    assert stats["output_bytes"] < stats["input_bytes"]
    assert stats["normalization_time_s"] >= 0


def test_solver_rejects_unknown_smt_experiment_mode(monkeypatch):
    monkeypatch.setenv("PYCT_SMT_EXPERIMENT_MODE", "typo")
    engine = SimpleNamespace(concolic_name_list=[], var_to_types={})

    with pytest.raises(ValueError, match="PYCT_SMT_EXPERIMENT_MODE"):
        Solver._build_formulas_from_constraint(
            engine,
            _Constraint([]),
            {},
        )


def test_affine_context_reuses_shared_nodes_across_predicates():
    types = {"x_VAR": "Real"}
    shared = ["+", "x_VAR", "x_VAR", "x_VAR"]
    context = AffineContext(types)
    first = normalize_affine_comparison(["<", shared, "1.000000000000000"], types, context=context)
    second = normalize_affine_comparison([">", shared, "0.000000000000000"], types, context=context)
    fresh = normalize_affine_comparison([">", shared, "0.000000000000000"], types)

    assert first.applied and second.applied
    assert second.formula_body == fresh.formula_body
    assert second.subexpression_cache_hits > 0
    assert second.node_count < fresh.node_count
    assert context.memo[id(shared)][0] is shared


@pytest.mark.parametrize("change", ["types", "budgets"])
def test_affine_context_rejects_changed_query_types_or_budgets(change):
    types = {"x_VAR": "Real"}
    context = AffineContext(types)
    budgets = AffineBudgets()
    if change == "types":
        types = {"x_VAR": "Int"}
    else:
        budgets = AffineBudgets(max_nodes=10)
    with pytest.raises(ValueError, match="AffineContext"):
        normalize_affine_comparison(["<", "x_VAR", "1"], types, budgets=budgets, context=context)


def test_shared_predicate_cache_does_not_reuse_real_normalization_for_integer_types():
    predicate = Predicate(["<", ["+", "x_VAR", "x_VAR", "x_VAR"], "1.000000000000000"], False)
    real_types = {"x_VAR": "Real"}
    real_expression, real_stats = predicate.get_expression_with_exact_affine(real_types, AffineContext(real_types))
    integer_types = {"x_VAR": "Int"}
    integer_expression, integer_stats = predicate.get_expression_with_exact_affine(integer_types, AffineContext(integer_types))

    assert real_stats["applied"]
    assert real_expression[0] == "not"
    assert not integer_stats["cache_hit"]
    assert not integer_stats["applied"]
    assert integer_stats["fallback_reason"] == "non_real_variable"
    assert integer_expression[1] is predicate.expr


def test_deep_shared_affine_normalization_does_not_recurse_or_expand():
    expression = "x_VAR"
    for _ in range(1200):
        expression = ["+", expression, expression]
    normalization = normalize_affine_comparison(["<", expression, "1.000000000000000"], {"x_VAR": "Real"})

    assert normalization.applied
    assert normalization.node_count < 1210
    assert normalization.subexpression_cache_hits >= 1200
    assert normalization.output_bytes < 500
    assert normalization.input_bytes == MAX_ESTIMATED_BYTES
    assert normalization.input_bytes_capped
