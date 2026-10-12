"""Bounded numerical sensitivity of a recorded scalar-X branch predicate.

This is a priority heuristic on the existing PWL AST, not a solver or a
reference-model probe. It never modifies the AST or creates attack candidates.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter

from libct.concolic import Concolic


ALGORITHM_VERSION = "target-shap-constraint-sensitivity-v1"


class _EvaluationError(ValueError):
    pass


class SensitivityEstimator:
    def __init__(self, variable, x, lower, upper, *, step_fraction=0.01,
                 epsilon=1e-8, max_nodes=200000, wall_budget=2.0):
        values = (x, lower, upper, step_fraction, epsilon, wall_budget)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Sensitivity settings must be finite")
        if not lower < upper or not lower <= x <= upper:
            raise ValueError("Sensitivity X must lie inside ordered bounds")
        if not 0 < step_fraction <= 1 or epsilon <= 0 or wall_budget <= 0:
            raise ValueError("Invalid sensitivity step/epsilon/budget")
        if type(max_nodes) is not int or max_nodes < 1:
            raise ValueError("Sensitivity node budget must be positive")
        self.variable, self.x, self.lower, self.upper = variable, x, lower, upper
        self.step_fraction, self.epsilon = step_fraction, epsilon
        self.max_nodes, self.wall_budget = max_nodes, wall_budget
        step = (upper - lower) * step_fraction
        self.points = tuple(dict.fromkeys((x, max(lower, x - step), min(upper, x + step))))
        # Entries retain their AST object, so object ID reuse cannot corrupt memoization.
        self._memo = {point: {} for point in self.points}
        self._nodes = 0
        self._seconds = 0.0
        self._attempted = 0
        self._evaluated = 0
        self._reasons = Counter()

    @property
    def stats(self):
        return {"version": ALGORITHM_VERSION, "attempted": self._attempted,
                "evaluated": self._evaluated, "fallback": sum(self._reasons.values()),
                "interrupted": self._attempted - self._evaluated - sum(self._reasons.values()),
                "fallback_reasons": dict(self._reasons), "seconds": self._seconds,
                "nodes": self._nodes, "step_fraction": self.step_fraction,
                "epsilon": self.epsilon, "max_nodes": self.max_nodes,
                "wall_budget": self.wall_budget}

    def estimate(self, predicate_expr):
        started = time.perf_counter()
        self._attempted += 1
        try:
            expr = predicate_expr
            seen = set()
            while isinstance(expr, Concolic) or (
                isinstance(expr, list) and expr and expr[0] == "not"
            ):
                if id(expr) in seen:
                    raise _EvaluationError("cycle")
                seen.add(id(expr))
                expr = expr.expr if isinstance(expr, Concolic) else expr[1]
            if not isinstance(expr, list) or len(expr) != 3 or expr[0] not in (
                "<", "<=", ">", ">=", "=",
            ):
                raise _EvaluationError("unsupported_predicate")
            margins = [float(self._evaluate(expr[1], point, started)) -
                       float(self._evaluate(expr[2], point, started)) for point in self.points]
            if not all(math.isfinite(value) for value in margins):
                raise _EvaluationError("nonfinite")
            slope = max(abs((value - margins[0]) / (point - self.x))
                        for point, value in zip(self.points[1:], margins[1:]))
            displacement = max(self.upper - self.x, self.x - self.lower)
            ratio = displacement * slope / (abs(margins[0]) + self.epsilon)
            if not math.isfinite(ratio):
                raise _EvaluationError("nonfinite")
            self._evaluated += 1
            return min(1.0, ratio)
        except (_EvaluationError, ArithmeticError, TypeError, ValueError, IndexError) as exc:
            reason = str(exc) if isinstance(exc, _EvaluationError) else "numeric_error"
            self._reasons[reason] += 1
            return None
        finally:
            self._seconds += time.perf_counter() - started

    def _evaluate(self, root, point, started):
        memo = self._memo[point]
        visiting = set()
        stack = [(root, 0)]
        while stack:
            if self._seconds + time.perf_counter() - started > self.wall_budget:
                raise _EvaluationError("time_budget")
            node, stage = stack.pop()
            key = id(node)
            if key in memo:
                continue
            if stage == 0:
                if key in visiting:
                    raise _EvaluationError("cycle")
                if self._nodes >= self.max_nodes:
                    raise _EvaluationError("node_budget")
                self._nodes += 1
                visiting.add(key)
                if isinstance(node, Concolic):
                    stack.extend(((node, 1), (node.expr, 0)))
                    continue
                if isinstance(node, list):
                    if not node or not isinstance(node[0], str):
                        raise _EvaluationError("unsupported_expression")
                    if node[0] == "ite":
                        if len(node) != 4:
                            raise _EvaluationError("unsupported_expression")
                        stack.extend(((node, 1), (node[1], 0)))
                    else:
                        stack.append((node, 1))
                        stack.extend((child, 0) for child in reversed(node[1:]))
                    continue
                value = self._leaf(node, point)
            elif isinstance(node, Concolic):
                value = memo[id(node.expr)][1]
            elif node[0] == "ite":
                condition = memo[id(node[1])][1]
                if type(condition) is not bool:
                    raise _EvaluationError("invalid_condition")
                chosen = node[2] if condition else node[3]
                if stage == 1:
                    stack.extend(((node, 2), (chosen, 0)))
                    continue
                value = memo[id(chosen)][1]
            else:
                value = self._apply(node[0], [memo[id(child)][1] for child in node[1:]])
            if not isinstance(value, (int, float, bool)) or not math.isfinite(value):
                raise _EvaluationError("nonfinite")
            memo[key] = (node, value)
            visiting.remove(key)
        return memo[id(root)][1]

    def _leaf(self, node, point):
        if isinstance(node, str):
            if node == self.variable:
                return point
            if node in ("true", "false"):
                return node == "true"
            negative = re.fullmatch(r"\(-\s+([^\s()]+)\)", node)
            try:
                return -float(negative.group(1)) if negative else float(node)
            except ValueError:
                raise _EvaluationError("unknown_symbol")
        if type(node) in (int, float, bool):
            return node
        raise _EvaluationError("unsupported_expression")

    @staticmethod
    def _apply(op, args):
        if op == "+" and args:
            result = args[0]
            for value in args[1:]:
                result += value
            return result
        if op == "*" and args:
            result = args[0]
            for value in args[1:]:
                result *= value
            return result
        if op == "-" and len(args) == 1:
            return -args[0]
        if op == "-" and len(args) == 2:
            return args[0] - args[1]
        if op == "/" and len(args) == 2:
            return args[0] / args[1]
        if len(args) == 2 and op in ("<", "<=", ">", ">=", "="):
            left, right = args
            if op == "<": return left < right
            if op == "<=": return left <= right
            if op == ">": return left > right
            if op == ">=": return left >= right
            return left == right
        if op == "not" and len(args) == 1:
            return not args[0]
        if op == "and": return all(args)
        if op == "or": return any(args)
        if op == "to_real" and len(args) == 1: return float(args[0])
        if op == "to_int" and len(args) == 1: return math.floor(args[0])
        raise _EvaluationError("unsupported_operator")
