# Copyright: see copyright.txt

class Concolic:
    def __init2__(self, value, expr=None, engine=None): # named __init2__ to be called "manually"
        from libct.solver import Solver
        from libct.utils import py2smt
        self.engine = engine if engine is not None else Solver._expr_has_engines_and_equals_value(expr, value)
        self.value = py2smt(value)
        self.expr = expr if expr is not None and self.engine is not None else self.value

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
