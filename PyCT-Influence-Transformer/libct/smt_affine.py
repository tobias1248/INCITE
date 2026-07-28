"""Exact, bounded affine normalization for SMT Real predicates.

This module deliberately operates at serialization time. It does not mutate the
concolic expression tree, and unsupported or over-budget expressions are left
unchanged by the caller.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction
import re
from typing import Dict, Mapping, Optional

from libct.concolic import Concolic


class AffineNotApplicable(ValueError):
    """Raised for expressions outside the supported affine Real subset."""

    def __init__(self, reason: str, detail: Optional[str] = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class AffineBudgetExceeded(AffineNotApplicable):
    """Raised when normalization would exceed a configured work budget."""


@dataclass(frozen=True)
class AffineBudgets:
    max_nodes: int = 100_000
    max_variables: int = 256
    max_rational_digits: int = 4_096


@dataclass(frozen=True)
class AffineNormalization:
    formula_body: Optional[str]
    applied: bool
    fallback_reason: Optional[str]
    fallback_detail: Optional[str]
    node_count: int
    variable_count: int
    term_count: int
    rational_digits: int
    max_rational_digits: int
    input_bytes: int
    output_bytes: int

    def to_stats(self):
        return asdict(self)


@dataclass
class _WorkState:
    budgets: AffineBudgets
    node_count: int = 0
    rational_digits: int = 0
    max_rational_digits: int = 0

    def visit(self):
        self.node_count += 1
        if self.node_count > self.budgets.max_nodes:
            raise AffineBudgetExceeded("node_budget")

    def observe_fraction(self, value: Fraction):
        try:
            numerator_digits = len(str(abs(value.numerator)))
            denominator_digits = len(str(value.denominator))
        except ValueError as exc:
            raise AffineBudgetExceeded("rational_digit_budget") from exc
        digits = numerator_digits + denominator_digits
        self.rational_digits += digits
        self.max_rational_digits = max(self.max_rational_digits, digits)
        if digits > self.budgets.max_rational_digits:
            raise AffineBudgetExceeded("rational_digit_budget")


@dataclass
class _Affine:
    coefficients: Dict[str, Fraction]
    constant: Fraction


def _raw_formula_size(expr, memo=None) -> int:
    """Return expanded SMT byte size without materialising the raw formula."""
    if memo is None:
        memo = {}
    if isinstance(expr, Concolic):
        return _raw_formula_size(expr.expr, memo)
    if isinstance(expr, str):
        return len(expr.encode("utf-8"))
    if isinstance(expr, list):
        cache_key = id(expr)
        cached = memo.get(cache_key)
        if cached is not None:
            return cached
        size = 2 + max(0, len(expr) - 1)
        size += sum(_raw_formula_size(item, memo) for item in expr)
        memo[cache_key] = size
        return size
    raise AffineNotApplicable("unsupported_expression_node")


def _parse_numeric_smt(atom: str) -> Fraction:
    tokens = re.findall(r"\(|\)|[^\s()]+", atom.strip())
    if not tokens:
        raise ValueError("empty numeric SMT expression")

    def parse_at(index: int):
        if index >= len(tokens):
            raise ValueError("unexpected end of numeric SMT expression")
        token = tokens[index]
        if token != "(":
            if token == ")":
                raise ValueError("unexpected closing parenthesis")
            return Fraction(token), index + 1

        if index + 1 >= len(tokens):
            raise ValueError("missing numeric SMT operator")
        operator = tokens[index + 1]
        if operator == "-":
            operand, next_index = parse_at(index + 2)
            if next_index >= len(tokens) or tokens[next_index] != ")":
                raise ValueError("invalid unary minus")
            return -operand, next_index + 1
        if operator == "/":
            numerator, next_index = parse_at(index + 2)
            denominator, next_index = parse_at(next_index)
            if next_index >= len(tokens) or tokens[next_index] != ")":
                raise ValueError("invalid exact division")
            if denominator == 0:
                raise ZeroDivisionError
            return numerator / denominator, next_index + 1
        raise ValueError("unsupported numeric SMT operator")

    value, next_index = parse_at(0)
    if next_index != len(tokens):
        raise ValueError("trailing numeric SMT tokens")
    return value


def _fraction_atom(atom: str) -> Fraction:
    try:
        return _parse_numeric_smt(atom)
    except (ValueError, ZeroDivisionError) as exc:
        reason = "unknown_variable" if atom.endswith("_VAR") else "unsupported_atom"
        raise AffineNotApplicable(reason, atom[:160]) from exc


def _constant(value: Fraction, state: _WorkState) -> _Affine:
    state.observe_fraction(value)
    return _Affine({}, value)


def _check_variables(affine: _Affine, state: _WorkState):
    if len(affine.coefficients) > state.budgets.max_variables:
        raise AffineBudgetExceeded("variable_budget")


def _add(left: _Affine, right: _Affine, state: _WorkState) -> _Affine:
    coefficients = dict(left.coefficients)
    for name, coefficient in right.coefficients.items():
        combined = coefficients.get(name, Fraction(0)) + coefficient
        state.observe_fraction(combined)
        if combined:
            coefficients[name] = combined
        else:
            coefficients.pop(name, None)
    constant = left.constant + right.constant
    state.observe_fraction(constant)
    result = _Affine(coefficients, constant)
    _check_variables(result, state)
    return result


def _scale(affine: _Affine, factor: Fraction, state: _WorkState) -> _Affine:
    coefficients = {}
    for name, coefficient in affine.coefficients.items():
        scaled = coefficient * factor
        state.observe_fraction(scaled)
        if scaled:
            coefficients[name] = scaled
    constant = affine.constant * factor
    state.observe_fraction(constant)
    result = _Affine(coefficients, constant)
    _check_variables(result, state)
    return result


def _to_affine(expr, variable_types: Mapping[str, str], state: _WorkState) -> _Affine:
    state.visit()
    if isinstance(expr, Concolic):
        return _to_affine(expr.expr, variable_types, state)

    if isinstance(expr, str):
        if expr in variable_types:
            if variable_types[expr] != "Real":
                raise AffineNotApplicable("non_real_variable")
            result = _Affine({expr: Fraction(1)}, Fraction(0))
            _check_variables(result, state)
            return result
        return _constant(_fraction_atom(expr), state)

    if not isinstance(expr, list) or not expr or not isinstance(expr[0], str):
        raise AffineNotApplicable("unsupported_expression_node")

    operator = expr[0]
    operands = expr[1:]
    if operator == "+" and operands:
        result = _constant(Fraction(0), state)
        for operand in operands:
            result = _add(result, _to_affine(operand, variable_types, state), state)
        return result

    if operator == "-" and operands:
        result = _to_affine(operands[0], variable_types, state)
        if len(operands) == 1:
            return _scale(result, Fraction(-1), state)
        for operand in operands[1:]:
            result = _add(
                result,
                _scale(_to_affine(operand, variable_types, state), Fraction(-1), state),
                state,
            )
        return result

    if operator == "*" and operands:
        result = _constant(Fraction(1), state)
        for operand in operands:
            factor = _to_affine(operand, variable_types, state)
            if result.coefficients and factor.coefficients:
                raise AffineNotApplicable("nonlinear_multiplication")
            if factor.coefficients:
                result = _scale(factor, result.constant, state)
            else:
                result = _scale(result, factor.constant, state)
        return result

    if operator == "/" and len(operands) == 2:
        numerator = _to_affine(operands[0], variable_types, state)
        denominator = _to_affine(operands[1], variable_types, state)
        if denominator.coefficients:
            raise AffineNotApplicable("symbolic_denominator")
        if denominator.constant == 0:
            raise AffineNotApplicable("zero_denominator")
        return _scale(numerator, Fraction(1, 1) / denominator.constant, state)

    raise AffineNotApplicable("unsupported_operator")


def _fraction_formula(value: Fraction) -> str:
    magnitude = abs(value)
    if magnitude.denominator == 1:
        formula = str(magnitude.numerator)
    else:
        formula = "(/ {} {})".format(magnitude.numerator, magnitude.denominator)
    return "(- {})".format(formula) if value < 0 else formula


def _term_formula(name: str, coefficient: Fraction) -> str:
    if coefficient == 1:
        return name
    if coefficient == -1:
        return "(- {})".format(name)
    return "(* {} {})".format(_fraction_formula(coefficient), name)


def _affine_formula(affine: _Affine) -> str:
    terms = [
        _term_formula(name, coefficient)
        for name, coefficient in sorted(affine.coefficients.items())
        if coefficient
    ]
    if affine.constant or not terms:
        terms.append(_fraction_formula(affine.constant))
    if len(terms) == 1:
        return terms[0]
    return "(+ {})".format(" ".join(terms))


def normalize_affine_comparison(
    expr,
    variable_types: Mapping[str, str],
    raw_body: Optional[str] = None,
    budgets: AffineBudgets = AffineBudgets(),
) -> AffineNormalization:
    """Return a shorter, exactly equivalent affine comparison when possible."""
    input_bytes = (
        _raw_formula_size(expr)
        if raw_body is None
        else len(raw_body.encode("utf-8"))
    )
    state = _WorkState(budgets)
    term_count = 0
    variable_count = 0

    try:
        state.visit()
        if (
            not isinstance(expr, list)
            or len(expr) != 3
            or expr[0] not in {"<", "<=", ">", ">=", "="}
        ):
            raise AffineNotApplicable("unsupported_predicate")
        operator = expr[0]
        left = _to_affine(expr[1], variable_types, state)
        right = _to_affine(expr[2], variable_types, state)
        difference = _add(left, _scale(right, Fraction(-1), state), state)
        variable_count = len(difference.coefficients)
        term_count = variable_count + int(bool(difference.constant) or not variable_count)
        formula_body = "({} {} 0)".format(operator, _affine_formula(difference))
        output_bytes = len(formula_body.encode("utf-8"))
        if output_bytes >= input_bytes:
            raise AffineNotApplicable("not_smaller")
    except AffineNotApplicable as exc:
        return AffineNormalization(
            formula_body=None,
            applied=False,
            fallback_reason=exc.reason,
            fallback_detail=exc.detail,
            node_count=state.node_count,
            variable_count=variable_count,
            term_count=term_count,
            rational_digits=state.rational_digits,
            max_rational_digits=state.max_rational_digits,
            input_bytes=input_bytes,
            output_bytes=input_bytes,
        )

    return AffineNormalization(
        formula_body=formula_body,
        applied=True,
        fallback_reason=None,
        fallback_detail=None,
        node_count=state.node_count,
        variable_count=variable_count,
        term_count=term_count,
        rational_digits=state.rational_digits,
        max_rational_digits=state.max_rational_digits,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
    )
