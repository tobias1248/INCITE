from __future__ import annotations

from types import SimpleNamespace
import shutil
import subprocess

import pytest

from libct.constraint import Constraint
from libct.predicate import Predicate
from libct.solver import Solver


@pytest.fixture
def path_and_engine(monkeypatch):
    monkeypatch.setattr(Constraint, "global_constraints", [])
    monkeypatch.setattr(Solver, "norm", True)
    monkeypatch.setattr(Solver, "limit_change_range", None)
    monkeypatch.setattr(Solver, "_last_smt_transform_stats", None)
    root = Constraint(None, None)
    first = Predicate([">", "x_VAR", "0"], True)
    last = Predicate([">", "x_VAR", "1"], False)
    path = root.add_child(first).add_child(last)
    engine = SimpleNamespace(
        concolic_name_list=["x_VAR", "pixel_VAR"],
        var_to_types={"x_VAR": "Real", "pixel_VAR": "Real"},
        solver_variable_bounds={"x_VAR": (-2.0, 2.0)},
    )
    return path, engine, first, last


@pytest.mark.parametrize("mode", [None, "full", "last"])
@pytest.mark.parametrize("normalization", ["raw", "exact_affine"])
def test_path_mode_selects_predicates_and_keeps_all_input_bounds(
    monkeypatch, path_and_engine, mode, normalization
):
    path, engine, first, last = path_and_engine
    monkeypatch.setenv("PYCT_SMT_EXPERIMENT_MODE", normalization)
    if mode is None:
        monkeypatch.delenv("PYCT_SMT_PATH_MODE", raising=False)
    else:
        monkeypatch.setenv("PYCT_SMT_PATH_MODE", mode)

    formula = Solver._build_formulas_from_constraint(engine, path, {})

    expected_mode = mode or "full"
    first_formula = (first.get_formula() if normalization == "raw" else
                     first.get_formula_with_exact_affine(engine.var_to_types)[0])
    last_formula = (last.get_formula() if normalization == "raw" else
                    last.get_formula_with_exact_affine(engine.var_to_types)[0])
    assert (first_formula in formula) is (expected_mode == "full")
    assert last_formula in formula
    assert "(<= x_VAR 2.000000000000000)" in formula
    assert "(>= x_VAR (- 2.000000000000000))" in formula
    assert "(and (<= pixel_VAR 1) (>= pixel_VAR 0))" in formula
    assert path.get_all_asserts() == [first, last]
    stats = Solver._last_smt_transform_stats
    assert stats["path_mode"] == expected_mode
    assert stats["original_assertion_count"] == 2
    assert stats["retained_assertion_count"] == (1 if expected_mode == "last" else 2)
    assert stats["assertion_count"] == stats["retained_assertion_count"]
    assert stats["mode"] == normalization


def test_last_mode_handles_root_without_a_predicate(monkeypatch, path_and_engine):
    path, engine, _, _ = path_and_engine
    monkeypatch.setenv("PYCT_SMT_PATH_MODE", "last")
    root = Constraint.global_constraints[0]

    formula = Solver._build_formulas_from_constraint(engine, root, {})

    assert "(check-sat)" in formula
    assert Solver._last_smt_transform_stats["retained_assertion_count"] == 0


def test_invalid_path_mode_fails_before_building_formula(monkeypatch, path_and_engine):
    path, engine, _, _ = path_and_engine
    monkeypatch.setenv("PYCT_SMT_PATH_MODE", "typo")

    with pytest.raises(ValueError, match="PYCT_SMT_PATH_MODE"):
        Solver._build_formulas_from_constraint(engine, path, {})


@pytest.mark.parametrize("mode", ["full", "last"])
def test_path_relaxation_keeps_percentage_limits_for_ordinary_inputs(
    monkeypatch, path_and_engine, mode
):
    path, engine, _, _ = path_and_engine
    monkeypatch.setenv("PYCT_SMT_PATH_MODE", mode)
    monkeypatch.setattr(Solver, "limit_change_range", 0.1)

    formula = Solver._build_formulas_from_constraint(engine, path, {"pixel": 0.5})

    assert "(and (<= pixel_VAR 0.55) (>= pixel_VAR 0.45))" in formula
    assert "(<= x_VAR 2.000000000000000)" in formula


def test_cvc5_solves_relaxed_query_but_rejects_contradictory_prefix(monkeypatch, path_and_engine):
    cvc5 = shutil.which("cvc5")
    if cvc5 is None:
        pytest.skip("cvc5 unavailable")
    path, engine, first, _ = path_and_engine
    first.expr = [">", "x_VAR", "1"]
    monkeypatch.setenv("PYCT_SMT_EXPERIMENT_MODE", "exact_affine")

    for mode, expected in (("full", "unsat"), ("last", "sat")):
        monkeypatch.setenv("PYCT_SMT_PATH_MODE", mode)
        formula = Solver._build_formulas_from_constraint(engine, path, {})
        # The unsat query cannot supply get-value bindings.
        formula = formula.split("(get-value", 1)[0]
        result = subprocess.run(
            [cvc5, "--lang=smt2"], input=formula, text=True,
            capture_output=True, timeout=5, check=True,
        )
        assert result.stdout.strip() == expected
