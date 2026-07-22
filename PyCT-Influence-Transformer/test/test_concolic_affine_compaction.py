from __future__ import annotations

from libct.concolic import compact_affine_sum
from libct.predicate import Predicate
from libct.utils import ConcolicObject


class _Engine:
    symbolic_enabled = True


def _formula(value) -> str:
    return Predicate.get_formula_deep(value.expr)


def test_affine_chain_is_reduced_to_one_term_per_symbol() -> None:
    engine = _Engine()
    x = ConcolicObject(2.0, "x_VAR", engine)

    result = (x * 3.0 + 4.0) * 2.0

    assert float(result) == 20.0
    assert result._affine_form == ({"x_VAR": 6.0}, 8.0)
    assert _formula(result) == "(+ (* 6.000000000000000 x_VAR) 8.000000000000000)"


def test_affine_terms_from_multiple_symbols_are_combined() -> None:
    engine = _Engine()
    x = ConcolicObject(2.0, "x_VAR", engine)
    y = ConcolicObject(5.0, "y_VAR", engine)

    result = x * 2.0 + y * 3.0 + x * 4.0

    assert float(result) == 27.0
    assert result._affine_form == ({"x_VAR": 6.0, "y_VAR": 3.0}, 0.0)
    assert len(_formula(result)) < 100


def test_nonlinear_product_keeps_original_expression() -> None:
    engine = _Engine()
    x = ConcolicObject(2.0, "x_VAR", engine)

    result = x * x

    assert result._affine_form is None
    assert _formula(result) == "(* x_VAR x_VAR)"


def test_repeated_linear_projection_does_not_grow_expression_depth() -> None:
    engine = _Engine()
    result = ConcolicObject(0.25, "pixel_VAR", engine)

    for _ in range(1_000):
        result = result * 0.999 + 0.001

    formula = _formula(result)
    assert result._affine_form is not None
    assert formula.count("pixel_VAR") == 1
    assert len(formula) < 100


def test_affine_sum_aggregates_dot_product_without_intermediate_chain() -> None:
    engine = _Engine()
    x = ConcolicObject(2.0, "x_VAR", engine)
    y = ConcolicObject(5.0, "y_VAR", engine)

    result = compact_affine_sum([1.0, x * 2.0, 3.0, y * 4.0])

    assert float(result) == 28.0
    assert result._affine_form == ({"x_VAR": 2.0, "y_VAR": 4.0}, 4.0)
    assert _formula(result).count("x_VAR") == 1
    assert _formula(result).count("y_VAR") == 1
