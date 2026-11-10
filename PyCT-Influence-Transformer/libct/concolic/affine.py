"""Exact, single-variable forms used only by the opt-in float representation.

Forms contain no operand objects or engine references. Concrete float execution
stays separate; this module describes the existing SMT Real arithmetic only.
"""

import math
import sys
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from libct.concolic import Concolic


MAX_RATIONAL_DIGITS = 4096
NOT_COMPACT = object()


class NotAffine(ValueError):
    pass


def _check_budget(*values):
    get_limit = getattr(sys, "get_int_max_str_digits", None)
    string_limit = get_limit() if get_limit is not None else 0
    for value in values:
        # 30103/100000 is an upper bound for log10(2). Most coefficients are
        # safely below the budget, so avoid decimal conversion on every MAC.
        upper = [(part.bit_length() * 30103) // 100000 + 1
                 for part in (abs(value.numerator), value.denominator)]
        if sum(upper) <= MAX_RATIONAL_DIGITS and (not string_limit or max(upper) <= string_limit):
            continue
        try:
            digits = len(str(abs(value.numerator))) + len(str(value.denominator))
        except ValueError as exc:
            raise NotAffine("rational_digit_budget") from exc
        if digits > MAX_RATIONAL_DIGITS:
            raise NotAffine("rational_digit_budget")


def _atom(value):
    magnitude = abs(value)
    text = str(magnitude.numerator)
    if magnitude.denominator != 1:
        text = "(/ {} {})".format(text, magnitude.denominator)
    return "(- {})".format(text) if value < 0 else text


@dataclass(frozen=True)
class AffineForm:
    variable: Optional[str]
    slope: Fraction
    intercept: Fraction

    def expression(self):
        if not self.slope:
            return _atom(self.intercept)
        term = self.variable if self.slope == 1 else ["*", _atom(self.slope), self.variable]
        return ["+", term, _atom(self.intercept)] if self.intercept else term


def _number(atom):
    # Constants must use the same decimal spelling as the legacy SMT writer,
    # rather than Fraction(float), which describes a different rational value.
    from libct.smt_affine import _parse_numeric_smt
    try:
        result = _parse_numeric_smt(atom)
    except (ValueError, ZeroDivisionError, RecursionError) as exc:
        raise NotAffine("unsupported_atom") from exc
    _check_budget(result)
    return result


def _operand(value):
    form = getattr(value, "__dict__", {}).get("_affine_form")
    if form is not None:
        return form
    if isinstance(value, Concolic):
        # Do not treat unpickled symbolic objects as constants just because
        # their engine reference was removed during transfer.
        expr = value.expr
        if not isinstance(expr, str):
            raise NotAffine("non_affine_operand")
        if expr.endswith("_VAR"):
            # Only explicit float input roots are initialized as variables.
            raise NotAffine("unrepresented_symbolic_operand")
        return AffineForm(None, Fraction(0), _number(expr))
    if isinstance(value, str):
        return AffineForm(None, Fraction(0), _number(value))
    if type(value) in (int, float, bool):
        from libct.utils import py2smt
        if type(value) is bool:
            value = float(value)
        return AffineForm(None, Fraction(0), _number(py2smt(value)))
    raise NotAffine("unsupported_operand")


def _combine(operator, left, right):
    if left.variable and right.variable and left.variable != right.variable:
        raise NotAffine("multiple_variables")
    variable = left.variable or right.variable
    if operator == "+":
        slope, intercept = left.slope + right.slope, left.intercept + right.intercept
    elif operator == "*":
        if left.slope and right.slope:
            raise NotAffine("nonlinear_multiplication")
        slope = left.slope * right.intercept + right.slope * left.intercept
        intercept = left.intercept * right.intercept
    elif operator == "/":
        if right.slope:
            raise NotAffine("symbolic_denominator")
        if not right.intercept:
            raise NotAffine("zero_denominator")
        slope, intercept = left.slope / right.intercept, left.intercept / right.intercept
    else:
        raise NotAffine("unsupported_operator")
    _check_budget(slope, intercept)
    return AffineForm(variable if slope else None, slope, intercept)


def initialize(value):
    """Compress an input root or one shallow operation; never scan old history."""
    from libct.solver import Solver
    from libct.symbolic_config import resolve_symbolic_representation
    state = value.__dict__
    if value.engine is None or resolve_symbolic_representation() != "compact-affine":
        return
    state["_compact_enabled"] = True
    if Solver.safety > 0:
        state["_compact_enabled"] = False
        state["_compact_fallback_reason"] = "safety_validation"
        return
    if "_affine_form" in state:
        return
    expr = state["expr"]
    try:
        if not math.isfinite(float.__float__(value)):
            raise NotAffine("non_finite_value")
        if isinstance(expr, str) and expr.endswith("_VAR"):
            form = AffineForm(expr, Fraction(1), Fraction(0))
        elif isinstance(expr, str):
            form = _operand(expr)
        elif isinstance(expr, list) and len(expr) == 3:
            if any(isinstance(item, Concolic) and item.engine is not None
                   and item.engine is not value.engine for item in expr[1:]):
                raise NotAffine("mixed_engines")
            form = _combine(expr[0], _operand(expr[1]), _operand(expr[2]))
        else:
            raise NotAffine("unsupported_expression")
    except NotAffine as exc:
        state["_compact_fallback_reason"] = str(exc)
        return
    state.pop("expr", None)
    state["_affine_form"] = form


def try_binary(value, operation, other):
    """Return an optimized float, or a sentinel before any concrete side effect."""
    if not value.__dict__.get("_compact_enabled") or value.engine is None:
        return NOT_COMPACT
    if getattr(value.engine, "symbolic_enabled", True) is False:
        return NOT_COMPACT
    from libct.solver import Solver
    if Solver.safety > 0:
        return NOT_COMPACT
    operators = {"__add__": "+", "__radd__": "+", "__mul__": "*",
                 "__rmul__": "*", "__truediv__": "/"}
    if operation not in operators:
        return NOT_COMPACT
    # Preserve legacy type coercion, zero shortcuts, and overloaded operands by
    # using the fast path only for native scalars and the same float class.
    if type(other) not in (int, float, bool, type(value)):
        return NOT_COMPACT
    if isinstance(other, Concolic) and other.engine is not None and other.engine is not value.engine:
        return NOT_COMPACT
    if (operation in {"__add__", "__radd__"} and type(other) in (float, bool) and not other
            or operation in {"__mul__", "__rmul__"} and type(other) is float and not other
            or operation == "__rmul__" and type(other) is bool and not other):
        return NOT_COMPACT
    # Legacy float division coerces ordinary integer denominators to float.
    try:
        denominator = float(other) if operation == "__truediv__" and type(other) in (int, bool) else other
        form = _combine(operators[operation], _operand(value), _operand(denominator))
    except (NotAffine, OverflowError):
        return NOT_COMPACT
    from libct.utils import unwrap
    try:
        result = getattr(float, operation)(value, unwrap(other))
    except (ArithmeticError, TypeError, ValueError):
        # The legacy dispatcher has its own reflected-operation fallback and
        # exception behavior (including overflowing ordinary integers).
        return NOT_COMPACT
    if result is NotImplemented or not math.isfinite(result):
        return NOT_COMPACT
    if operation == "__truediv__":
        # Commit to this path only after all fallback decisions. Otherwise the
        # legacy dispatcher would evaluate the guard a second time.
        try:
            (other != 0).__bool__()
        except Exception:
            pass
    return type(value)(result, form, value.engine)


def query_statistics(expressions, max_nodes=100_000):
    """Count live query values, not all operations in a child forward run."""
    stack = list(expressions)
    seen = set()
    compact = 0
    reasons = Counter()
    while stack and len(seen) < max_nodes:
        node = stack.pop()
        if isinstance(node, str) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Concolic):
            state = node.__dict__
            if "_affine_form" in state:
                compact += 1
                continue
            reason = state.get("_compact_fallback_reason")
            if reason:
                reasons[reason] += 1
            stack.append(node.expr)
        elif isinstance(node, list):
            stack.extend(node[1:])
    return {"compact_nodes": compact, "fallback_reasons": dict(reasons),
            "visited_nodes": len(seen), "truncated": bool(stack)}
