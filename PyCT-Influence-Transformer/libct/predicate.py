# Copyright: see copyright.txt

from libct.concolic import Concolic

def depth(expr):
    if isinstance(expr, Concolic):
        return 1 + depth(expr.expr)
    if isinstance(expr, list):
        return depth(expr[1])
    return 1
# max(depth(expr[0]), expr[1]), depth(expr[2]), ..., depth(expr[...]))?
class Predicate:
    def __init__(self, expr, value):
        # print("line7",expr, value)
        self.expr = expr
        self.value = value
        # print("line10",self.expr, self.value)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.value == other.value and \
            self._eq_worker(self.expr, other.expr)

    def _eq_worker(self, expr1, expr2):
        if isinstance(expr1, Concolic) and isinstance(expr2, Concolic):
            return self._eq_worker(expr1.expr, expr2.expr)
        if isinstance(expr1, list) and isinstance(expr2, list) and len(expr1) == len(expr2):
            return next((False for (e1, e2) in zip(expr1, expr2) if not self._eq_worker(e1, e2)), True)
        return expr1 == expr2

    def get_formula(self):
        formula = self.get_formula_deep(self.expr)
        if not self.value: formula = "(not " + formula + ")"
        return "(assert " + formula + ")"

    @staticmethod
    def get_formula_deep(expr):
        # print("line30")
        return Predicate._get_formula(expr, True)

    @staticmethod
    def get_formula_shallow(expr):
        return Predicate._get_formula(expr, False)

    @staticmethod
    def _get_formula(expr, mode):
        # print("mode:",mode)
        if isinstance(expr, Concolic): # Please note that this branch must be placed first!
            # print("concolic")
            return Predicate._get_formula(expr.expr, mode) if mode else expr.value
        if isinstance(expr, str):
            # print("str")
            return expr
        if isinstance(expr, list):
            # print("list")
            return "(" + " ".join(Predicate._get_formula(exp, mode) for exp in expr) + ")"
        raise NotImplementedError

    def __str__(self):
        # print("line54")
        # print("self.expr:",self.expr)
        # print("self.value:",self.value)
        # print(type(self.expr))
        # print([(type(expr), depth(expr), expr.expr if isinstance(expr, Concolic) else None,isinstance(expr, str),isinstance(expr, list)) for expr in self.expr])
        return f"{Predicate.get_formula_deep(self.expr)} = {self.value}"
