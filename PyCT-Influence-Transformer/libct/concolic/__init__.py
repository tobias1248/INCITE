# Copyright: see copyright.txt

import builtins
import math


def _combine_coefficients(left, right, right_scale=1.0):
    coefficients = dict(left)
    for name, coefficient in right.items():
        combined = coefficients.get(name, 0.0) + right_scale * coefficient
        if combined == 0.0:
            coefficients.pop(name, None)
        else:
            coefficients[name] = combined
    return coefficients


def _scale_affine(form, scale):
    coefficients, constant = form
    return (
        {
            name: coefficient * scale
            for name, coefficient in coefficients.items()
            if coefficient * scale != 0.0
        },
        constant * scale,
    )


def _operand_affine_form(operand):
    if isinstance(operand, Concolic):
        return getattr(operand, "_affine_form", None)
    if type(operand) in (builtins.int, builtins.float) and math.isfinite(
        builtins.float(operand)
    ):
        return {}, builtins.float(operand)
    return None


def _derive_affine_form(value, expr, engine):
    if type(value) is not builtins.float or not math.isfinite(value):
        return None
    if expr is None and engine is None:
        return {}, value
    if isinstance(expr, builtins.str):
        if engine is not None and expr.endswith("_VAR"):
            return {expr: 1.0}, 0.0
        return {}, value
    if not isinstance(expr, list) or not expr:
        return None

    operator = expr[0]
    operands = [_operand_affine_form(item) for item in expr[1:]]
    if any(form is None for form in operands):
        return None

    if operator == "+" and operands:
        coefficients = {}
        constant = 0.0
        for operand_coefficients, operand_constant in operands:
            coefficients = _combine_coefficients(coefficients, operand_coefficients)
            constant += operand_constant
        return coefficients, constant
    if operator == "-" and len(operands) == 1:
        return _scale_affine(operands[0], -1.0)
    if operator == "-" and len(operands) == 2:
        left_coefficients, left_constant = operands[0]
        right_coefficients, right_constant = operands[1]
        return (
            _combine_coefficients(left_coefficients, right_coefficients, -1.0),
            left_constant - right_constant,
        )
    if operator == "*" and len(operands) == 2:
        left, right = operands
        if not left[0]:
            return _scale_affine(right, left[1])
        if not right[0]:
            return _scale_affine(left, right[1])
        return None
    if operator == "/" and len(operands) == 2:
        numerator, denominator = operands
        if denominator[0] or denominator[1] == 0.0:
            return None
        return _scale_affine(numerator, 1.0 / denominator[1])
    return None


def _affine_expression(form, py2smt):
    coefficients, constant = form
    if not all(math.isfinite(value) for value in [constant, *coefficients.values()]):
        return None

    terms = []
    for name in sorted(coefficients):
        coefficient = coefficients[name]
        if coefficient == 1.0:
            terms.append(name)
        else:
            terms.append(["*", py2smt(coefficient), name])
    if constant != 0.0 or not terms:
        terms.append(py2smt(constant))
    if len(terms) == 1:
        return terms[0]
    return ["+", *terms]


def compact_affine_sum(values):
    """Sum numeric values while creating at most one affine concolic object."""
    items = list(values)
    if not items:
        return 0.0

    coefficients = {}
    constant = 0.0
    concrete_total = 0.0
    engine = None
    for item in items:
        form = _operand_affine_form(item)
        if form is None:
            total = 0.0
            for fallback_item in items:
                total = total + fallback_item
            return total
        item_coefficients, item_constant = form
        coefficients = _combine_coefficients(coefficients, item_coefficients)
        constant += item_constant
        concrete_total += builtins.float(item)
        if isinstance(item, Concolic) and item.engine is not None:
            engine = item.engine

    if engine is None:
        return concrete_total

    from libct.utils import ConcolicObject, py2smt

    form = coefficients, constant
    expr = _affine_expression(form, py2smt)
    if expr is None:
        total = 0.0
        for fallback_item in items:
            total = total + fallback_item
        return total
    result = ConcolicObject(concrete_total, expr, engine)
    result._affine_form = form
    return result


class Concolic:
    def __init2__(self, value, expr=None, engine=None): # named __init2__ to be called "manually"
        from libct.solver import Solver
        from libct.utils import py2smt
        resolved_engine = engine if engine is not None else Solver._expr_has_engines_and_equals_value(expr, value)
        if resolved_engine is not None and getattr(resolved_engine, "symbolic_enabled", True) is False:
            resolved_engine = None
        self.engine = resolved_engine
        self.value = py2smt(value)
        self._affine_form = _derive_affine_form(value, expr, resolved_engine)
        compact_expr = None
        if self._affine_form is not None and resolved_engine is not None:
            compact_expr = _affine_expression(self._affine_form, py2smt)
            if compact_expr is None:
                self._affine_form = None
        self.expr = (
            compact_expr
            if compact_expr is not None
            else expr if expr is not None and self.engine is not None else self.value
        )
        self.formula = None

    def __getstate__(self):
        """Drop engine references when pickling to avoid carrying unpicklable objects (e.g., models)."""
        state = self.__dict__.copy()
        state["engine"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @staticmethod
    def find_engine_in_expr(expr):
        if isinstance(expr, Concolic):
            return expr.engine
        if isinstance(expr, list):
            for e in expr:
                if (engine := Concolic.find_engine_in_expr(e)) is not None:
                    return engine
        return None

# https://stackoverflow.com/questions/16056574/how-does-python-prevent-a-class-from-being-subclassed/16056691#16056691
class MetaFinal(type):
    def __new__(cls, name, bases, classdict):
        for b in bases:
            if isinstance(b, cls):
                raise TypeError(f"type '{b.__name__}' is not an acceptable base type")
        return type.__new__(cls, name, bases, dict(classdict))
