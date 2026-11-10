import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from libct.concolic import Concolic
from libct.expression_graph import MAX_ESTIMATED_BYTES
from libct.predicate import Predicate
from libct.smt_shared import serialize_shared_assertions
from libct.solver import Solver


class _Constraint:
    def __init__(self, predicates):
        self.predicates = predicates

    def get_all_asserts(self):
        return self.predicates


def _cvc5(smt):
    executable = shutil.which("cvc5")
    if executable is None:
        pytest.skip("cvc5 is required for shared SMT semantics regression")
    result = subprocess.run(
        [executable, "--lang=smt2", "--quiet"], input=smt, text=True,
        capture_output=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def _assert_equivalent(expressions, types):
    shared = serialize_shared_assertions(expressions, types)
    raw_body = "(and {})".format(" ".join(Predicate.get_formula_deep(expr) for expr in expressions))
    shared_body = shared.formula[len("(assert "):-1]
    declarations = "\n".join("(declare-const {} {})".format(name, sort) for name, sort in types.items())
    assert _cvc5("\n".join([
        "(set-logic ALL)", declarations,
        "(assert (xor {} {}))".format(raw_body, shared_body), "(check-sat)",
    ])) == "unsat"
    return shared


@pytest.fixture
def shared_solver(monkeypatch):
    monkeypatch.setenv("PYCT_SMT_SERIALIZATION", "shared")
    monkeypatch.setenv("PYCT_SMT_PATH_MODE", "full")
    monkeypatch.setattr(Solver, "norm", False)
    monkeypatch.setattr(Solver, "limit_change_range", None)
    monkeypatch.setattr(Solver, "_last_smt_transform_stats", None)
    return SimpleNamespace(concolic_name_list=["x_VAR"], var_to_types={"x_VAR": "Real"})


def test_shared_dependencies_use_nested_lets_with_parallel_siblings():
    first = ["+", "x_VAR", "1"]
    sibling = ["-", "x_VAR", "2"]
    dependent = ["+", first, first, sibling, sibling]
    expressions = [["<", dependent, "10"], [">=", dependent, first]]
    result = _assert_equivalent(expressions, {"x_VAR": "Real"})
    assert result.stats["shared_nodes"] == 3
    assert result.formula.count("(let (") == 2
    assert result.stats["output_bytes"] == len(result.formula.encode("utf-8"))
    assert result.stats["input_bytes"] == sum(len(Predicate(expr, True).get_formula().encode("utf-8")) for expr in expressions)


def test_concolic_wrappers_and_repeated_roots_preserve_scope():
    wrapper = Concolic()
    wrapper.expr = ["+", "x_VAR", "(- (/ 1 3))"]
    first = [">", wrapper, "(- 1)"]
    second = ["<", wrapper, "(/ 7 3)"]
    result = _assert_equivalent([first, second, first], {"x_VAR": "Real"})
    assert result.stats["shared_nodes"] >= 2
    assert "(- (/ 1 3))" in result.formula


@pytest.mark.parametrize("opaque", ["__pyct_shared_0", "|__pyct_shared_0|"])
def test_generated_names_do_not_capture_symbols_in_opaque_smt_atoms(opaque):
    shared = ["+", "x_VAR", "1"]
    expressions = [["=", shared, "(+ {} (- (/ 1 3)))".format(opaque)], [">", shared, "0"]]
    result = _assert_equivalent(expressions, {"x_VAR": "Real", opaque: "Real"})
    assert "((__pyct_shared_0 " not in result.formula


def test_declared_names_are_reserved_even_when_not_used_in_expressions():
    shared = ["+", "x_VAR", "1"]
    result = _assert_equivalent([["=", shared, shared]], {"x_VAR": "Real", "__pyct_shared_0": "Real"})
    assert "((__pyct_shared_0 " not in result.formula
    assert "__pyct_shared_1" in result.formula


def test_empty_shared_query_has_no_assertion():
    result = serialize_shared_assertions([], {})
    assert result.formula == ""
    assert result.stats == {"unique_nodes": 0, "shared_nodes": 0, "input_bytes": 0, "input_bytes_capped": False, "output_bytes": 0}


def test_deep_exponential_dag_serializes_in_bounded_space():
    expression = "x_VAR"
    for _ in range(1500):
        expression = ["+", expression, expression]
    result = serialize_shared_assertions([["=", expression, "0"]], {"x_VAR": "Real"})
    assert result.stats["shared_nodes"] == 1499
    assert len(result.formula) < 200_000
    assert result.stats["input_bytes"] == MAX_ESTIMATED_BYTES
    assert result.stats["input_bytes_capped"]


@pytest.mark.parametrize("normalization", ["raw", "exact_affine"])
@pytest.mark.parametrize("kind", ["clip", "pwl", "nonlinear"])
def test_shared_solver_fallback_never_expands_raw_formula(monkeypatch, shared_solver, normalization, kind):
    monkeypatch.setenv("PYCT_SMT_EXPERIMENT_MODE", normalization)
    repeated = ["+", "x_VAR", "(- (/ 1 3))"]
    expressions = {
        "clip": ["ite", ["<", repeated, "0.0"], "0.0", ["ite", [">", repeated, "1.0"], "1.0", repeated]],
        "pwl": ["ite", ["<", "x_VAR", "0"], ["*", "2", repeated], ["+", repeated, repeated]],
        "nonlinear": ["*", repeated, repeated],
    }
    predicates = [Predicate([">", expressions[kind], "1.0"], False), Predicate(["<", expressions[kind], "10"], True)]

    def forbid_raw(*args, **kwargs):
        pytest.fail("shared query expanded a raw formula")

    monkeypatch.setattr(Predicate, "get_formula_deep", forbid_raw)
    formula = Solver._build_formulas_from_constraint(shared_solver, _Constraint(predicates), {})
    assert "(let (" in formula
    assert "(not (> " in formula
    assert Solver._last_smt_transform_stats["serialization"] == "shared"
    if normalization == "exact_affine":
        assert Solver._last_smt_transform_stats["fallback_count"] == 2
    assert _cvc5(formula.split("(get-value", 1)[0]) == "sat"


def test_solver_reuses_subexpressions_and_predicate_metadata_per_query(monkeypatch, shared_solver):
    monkeypatch.setenv("PYCT_SMT_EXPERIMENT_MODE", "exact_affine")
    shared = ["+", "x_VAR", "x_VAR", "x_VAR"]
    predicates = [Predicate([">", shared, "0.000000000000000"], True), Predicate(["<", shared, "1.000000000000000"], True)]
    constraint = _Constraint(predicates)
    first = Solver._build_formulas_from_constraint(shared_solver, constraint, {})
    first_stats = dict(Solver._last_smt_transform_stats)
    second = Solver._build_formulas_from_constraint(shared_solver, constraint, {})
    second_stats = Solver._last_smt_transform_stats
    assert first == second
    assert first_stats["subexpression_cache_hits"] > 0
    assert first_stats["cache_hit_count"] == 0
    assert second_stats["cache_hit_count"] == 2
    assert second_stats["node_count"] == 0
    assert second_stats["subexpression_cache_hits"] == 0
    assert _cvc5(second.split("(get-value", 1)[0]) == "sat"


def test_solver_deep_fallback_uses_shared_graph_without_expansion(monkeypatch, shared_solver):
    monkeypatch.setenv("PYCT_SMT_EXPERIMENT_MODE", "exact_affine")
    expression = "x_VAR"
    for _ in range(1500):
        expression = ["+", expression, expression]
    clipped = ["ite", [">", expression, "1.0"], "1.0", expression]
    predicate = Predicate(["<=", clipped, "1.0"], True)

    def forbid_raw(*args, **kwargs):
        pytest.fail("deep fallback expanded its raw expression")

    monkeypatch.setattr(Predicate, "get_formula_deep", forbid_raw)
    formula = Solver._build_formulas_from_constraint(shared_solver, _Constraint([predicate]), {})
    assert len(formula) < 200_000
    assert Solver._last_smt_transform_stats["fallback_reasons"] == {"unsupported_operator": 1}
    assert Solver._last_smt_transform_stats["input_bytes"] == MAX_ESTIMATED_BYTES
    assert Solver._last_smt_transform_stats["input_bytes_capped"]


def test_huge_raw_expanded_size_diagnostics_remain_json_serializable(monkeypatch, shared_solver):
    monkeypatch.setenv("PYCT_SMT_EXPERIMENT_MODE", "raw")
    expression = "x_VAR"
    for _ in range(15_000):
        expression = ["+", expression, expression]
    predicate = Predicate(["=", expression, "0"], True)
    formula = Solver._build_formulas_from_constraint(shared_solver, _Constraint([predicate]), {})
    stats = Solver._last_smt_transform_stats

    assert len(formula) < 2_000_000
    assert stats["input_bytes"] == MAX_ESTIMATED_BYTES
    assert stats["input_bytes_capped"]
    assert stats["shared_serialization"]["input_bytes_capped"]
    assert json.loads(json.dumps(stats))["input_bytes"] == MAX_ESTIMATED_BYTES


@pytest.mark.parametrize("difference,expected", [(0, "sat"), (1, "unsat")])
def test_cvc5_evaluates_deep_dependent_shared_bindings(difference, expected):
    expression = "x_VAR"
    for _ in range(75):
        expression = ["+", expression, expression]
    shared = serialize_shared_assertions(
        [["=", "x_VAR", "1"], ["=", expression, str(2 ** 75 + difference)]],
        {"x_VAR": "Real"},
    )
    assert _cvc5("\n".join([
        "(set-logic QF_LRA)", "(declare-const x_VAR Real)", shared.formula, "(check-sat)",
    ])) == expected


def test_constraint_logs_separate_raw_and_shared_outputs(monkeypatch, tmp_path):
    def output_subdir(dirname, *parts):
        return tmp_path.joinpath(dirname, *parts)

    monkeypatch.setattr("libct.solver.get_repo_output_subdir", output_subdir)
    engine = SimpleNamespace(
        model_path="models/demo.h5", popped_log_attack_mode="shap_solver60s",
        concolic_flag_dict={"pixel": 1},
    )
    monkeypatch.delenv("PYCT_SMT_SERIALIZATION", raising=False)
    raw = Solver._resolve_constraint_log_path(engine, 7)
    monkeypatch.setenv("PYCT_SMT_SERIALIZATION", "shared")
    shared = Solver._resolve_constraint_log_path(engine, 7)

    assert raw == tmp_path / "popped_constraint_position" / "demo" / "shap_solver60s_1" / "constraint_7.txt"
    assert shared == tmp_path / "popped_constraint_position" / "demo" / "shap_solver60s_smtshared_1" / "constraint_7.txt"
