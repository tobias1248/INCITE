"""Exact, bounded affine normalization for SMT Real predicates.

This module deliberately operates at serialization time. It does not mutate the
concolic expression tree, and unsupported or over-budget expressions are left
unchanged by the caller.
"""

from dataclasses import asdict, dataclass, field
from fractions import Fraction
import re
from typing import Dict, Mapping, Optional

from libct.concolic import Concolic
from libct.expression_graph import expanded_formula_size, MAX_ESTIMATED_BYTES


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
    subexpression_cache_hits: int = 0
    input_bytes_capped: bool = False

    def to_stats(self):
        return asdict(self)


@dataclass
class _WorkState:
    budgets: AffineBudgets
    node_count: int = 0
    rational_digits: int = 0
    max_rational_digits: int = 0
    memo: dict = field(default_factory=dict)
    cache_hits: int = 0

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


class AffineContext:
    """Memoized exact forms and sizes, owned by one SMT query only."""

    def __init__(self, variable_types, budgets=AffineBudgets()):
        self.signature = tuple(sorted(variable_types.items()))
        self.budgets = budgets
        self.memo = {}
        self.sizes = {}


def _raw_formula_size(expr, memo=None) -> int:
    """Return expanded SMT byte size without materialising the raw formula."""
    try:
        return expanded_formula_size(expr, memo)
    except TypeError as exc:
        raise AffineNotApplicable("unsupported_expression_node") from exc


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
    stack = [(expr, False)]
    active = set()
    while stack:
        node, exiting = stack.pop()
        key = id(node)
        if not exiting and key in state.memo:
            state.cache_hits += 1
            continue
        if exiting:
            if isinstance(node, Concolic):
                result = state.memo[id(node.expr)][1]
            else:
                result = _apply_affine_operator(
                    node[0], [state.memo[id(item)][1] for item in node[1:]], state,
                )
            state.memo[key] = (node, result)
            active.remove(key)
            continue
        state.visit()
        if key in active:
            raise AffineNotApplicable("cyclic_expression")
        if isinstance(node, str):
            if node in variable_types:
                if variable_types[node] != "Real":
                    raise AffineNotApplicable("non_real_variable")
                result = _Affine({node: Fraction(1)}, Fraction(0))
                _check_variables(result, state)
            else:
                result = _constant(_fraction_atom(node), state)
            state.memo[key] = (node, result)
            continue
        if isinstance(node, Concolic):
            children = (node.expr,)
        elif isinstance(node, list) and node and isinstance(node[0], str):
            operator, children = node[0], node[1:]
            if not ((operator in {"+", "-", "*"} and children)
                    or (operator == "/" and len(children) == 2)):
                raise AffineNotApplicable("unsupported_operator")
        else:
            raise AffineNotApplicable("unsupported_expression_node")
        active.add(key)
        stack.append((node, True))
        stack.extend((child, False) for child in reversed(children))
    return state.memo[id(expr)][1]


def _apply_affine_operator(operator, operands, state):
    if operator == "+" and operands:
        result = _constant(Fraction(0), state)
        for operand in operands:
            result = _add(result, operand, state)
        return result

    if operator == "-" and operands:
        result = operands[0]
        if len(operands) == 1:
            return _scale(result, Fraction(-1), state)
        for operand in operands[1:]:
            result = _add(
                result,
                _scale(operand, Fraction(-1), state),
                state,
            )
        return result

    if operator == "*" and operands:
        result = _constant(Fraction(1), state)
        for factor in operands:
            if result.coefficients and factor.coefficients:
                raise AffineNotApplicable("nonlinear_multiplication")
            if factor.coefficients:
                result = _scale(factor, result.constant, state)
            else:
                result = _scale(result, factor.constant, state)
        return result

    if operator == "/" and len(operands) == 2:
        numerator, denominator = operands
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
    context: Optional[AffineContext] = None,
) -> AffineNormalization:
    """Return a shorter, exactly equivalent affine comparison when possible."""
    if context is not None and (
        context.signature != tuple(sorted(variable_types.items())) or context.budgets != budgets
    ):
        raise ValueError("AffineContext must match query variable types and budgets")
    input_bytes = (
        _raw_formula_size(expr, context.sizes if context is not None else None)
        if raw_body is None
        else len(raw_body.encode("utf-8"))
    )
    state = _WorkState(budgets, memo=context.memo if context is not None else {})
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
            subexpression_cache_hits=state.cache_hits,
            input_bytes_capped=input_bytes >= MAX_ESTIMATED_BYTES,
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
        subexpression_cache_hits=state.cache_hits,
        input_bytes_capped=input_bytes >= MAX_ESTIMATED_BYTES,
    )
