# Copyright: see copyright.txt

from libct.concolic import Concolic


class _CseBudgetExceeded(RuntimeError):
    """Raised when CSE analysis would exceed its bounded work budget."""


class _ExpressionNode:
    """Internable expression node whose structural hash is computed once."""

    __slots__ = ("items", "_hash")

    def __init__(self, items):
        self.items = tuple(items)
        self._hash = hash(self.items)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, _ExpressionNode) and self.items == other.items

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class _CommonSubexpressionSerializer:
    """Serialize an expression with bounded, deterministic SMT ``let`` sharing."""

    _MIN_EXPRESSION_LENGTH = 48
    _MAX_BINDINGS = 512
    _MAX_UNIQUE_NODES = 50_000
    _MAX_TRAVERSAL_EDGES = 250_000
    _MAX_OCCURRENCE_COUNT = 1_000_000_000

    def __init__(self, expr):
        self._concolic_cache = {}
        self._list_cache = {}
        self._interned_nodes = {}
        self._unique_node_count = 0
        self._traversal_edge_count = 0
        self.root = self._normalize(expr)
        self._counts = {}
        self._first_seen = {}
        self._sizes = {}
        self._visit_index = 0
        self._index_subexpressions()

    def _normalize(self, expr):
        if isinstance(expr, Concolic):
            cache_key = id(expr)
            if cache_key not in self._concolic_cache:
                self._concolic_cache[cache_key] = self._normalize(expr.expr)
            return self._concolic_cache[cache_key]
        if isinstance(expr, list):
            cache_key = id(expr)
            cached = self._list_cache.get(cache_key)
            if cached is not None:
                return cached
            self._unique_node_count += 1
            self._traversal_edge_count += len(expr)
            if self._unique_node_count > self._MAX_UNIQUE_NODES:
                raise _CseBudgetExceeded(
                    f"unique-node budget exceeded ({self._MAX_UNIQUE_NODES})"
                )
            if self._traversal_edge_count > self._MAX_TRAVERSAL_EDGES:
                raise _CseBudgetExceeded(
                    f"edge budget exceeded ({self._MAX_TRAVERSAL_EDGES})"
                )
            node = _ExpressionNode(self._normalize(item) for item in expr)
            interned = self._interned_nodes.get(node)
            if interned is None:
                self._interned_nodes[node] = node
                interned = node
            self._list_cache[cache_key] = interned
            return interned
        if isinstance(expr, str):
            return expr
        raise NotImplementedError

    def _index_subexpressions(self):
        """Count expanded occurrences while visiting each DAG node once."""
        visited = set()
        postorder = []

        def visit(expr):
            if isinstance(expr, str) or expr in visited:
                return
            visited.add(expr)
            self._first_seen[expr] = self._visit_index
            self._visit_index += 1
            for item in expr:
                visit(item)
            postorder.append(expr)

        visit(self.root)
        self._counts[self.root] = 1
        for expr in reversed(postorder):
            parent_count = self._counts.get(expr, 0)
            for item in expr:
                if isinstance(item, str):
                    continue
                child_count = self._counts.get(item, 0) + parent_count
                self._counts[item] = min(
                    child_count,
                    self._MAX_OCCURRENCE_COUNT,
                )

    def _serialized_size(self, expr):
        if isinstance(expr, str):
            return len(expr)
        cached = self._sizes.get(expr)
        if cached is not None:
            return cached
        size = 2 + max(0, len(expr) - 1)
        size += sum(self._serialized_size(item) for item in expr)
        self._sizes[expr] = size
        return size

    def _binding_candidates(self):
        candidates = []
        assumed_name_length = len("_pyct_cse_511")
        for expr, count in self._counts.items():
            if count < 2:
                continue
            expression_length = self._serialized_size(expr)
            if expression_length < self._MIN_EXPRESSION_LENGTH:
                continue
            estimated_saving = (
                (count - 1) * expression_length
                - count * assumed_name_length
                - 12
            )
            if estimated_saving <= 0:
                continue
            candidates.append(
                (
                    estimated_saving,
                    expression_length,
                    self._first_seen[expr],
                    expr,
                )
            )
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected = candidates[:self._MAX_BINDINGS]
        return [item[3] for item in selected]

    def _serialize(self, expr, aliases):
        alias = aliases.get(expr) if not isinstance(expr, str) else None
        if alias is not None:
            return alias
        if isinstance(expr, str):
            return expr
        return "(" + " ".join(self._serialize(item, aliases) for item in expr) + ")"

    def serialize(self):
        candidates = self._binding_candidates()
        candidates.sort(key=lambda expr: (self._serialized_size(expr), self._first_seen[expr]))
        aliases = {}
        bindings = []
        for index, expr in enumerate(candidates):
            name = f"_pyct_cse_{index}"
            binding_expression = self._serialize(expr, aliases)
            aliases[expr] = name
            bindings.append((name, binding_expression))

        result = self._serialize(self.root, aliases)
        if bindings:
            prefix = "".join(
                f"(let (({name} {binding_expression})) "
                for name, binding_expression in bindings
            )
            result = prefix + result + ")" * len(bindings)
        return result, len(bindings)

    def original_size(self):
        return self._serialized_size(self.root)


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
            self.expressions_equal(self.expr, other.expr)

    @staticmethod
    def expressions_equal(expr1, expr2):
        if isinstance(expr1, Concolic):
            expr1 = expr1.expr
        if isinstance(expr2, Concolic):
            expr2 = expr2.expr
        if isinstance(expr1, list) and isinstance(expr2, list) and len(expr1) == len(expr2):
            return all(
                Predicate.expressions_equal(e1, e2)
                for e1, e2 in zip(expr1, expr2)
            )
        return expr1 == expr2

    @staticmethod
    def trivial_truth_value(expr):
        """Return the fixed truth value of a reflexive comparison, if known."""
        if isinstance(expr, Concolic):
            return Predicate.trivial_truth_value(expr.expr)
        if not isinstance(expr, list):
            return None
        if len(expr) == 2 and expr[0] == "not":
            inner = Predicate.trivial_truth_value(expr[1])
            return None if inner is None else not inner
        if (
            len(expr) == 3
            and expr[0] in {"=", "<", "<=", ">", ">="}
            and Predicate.expressions_equal(expr[1], expr[2])
        ):
            return expr[0] in {"=", "<=", ">="}
        return None

    def get_formula(self):
        formula = self.get_formula_deep(self.expr)
        if not self.value: formula = "(not " + formula + ")"
        return "(assert " + formula + ")"

    def get_formula_with_sharing(self):
        """Return an equivalent assertion using SMT ``let`` common subexpressions."""
        expr = self.expr if self.value else ["not", self.expr]
        try:
            serializer = _CommonSubexpressionSerializer(expr)
        except _CseBudgetExceeded as exc:
            formula = self.get_formula()
            return formula, {
                "binding_count": 0,
                "bytes_before": len(formula),
                "bytes_after": len(formula),
                "fallback_reason": str(exc),
            }
        shared_body, binding_count = serializer.serialize()
        shared_formula = "(assert " + shared_body + ")"
        original_size = len("(assert ") + serializer.original_size() + len(")")
        if len(shared_formula) >= original_size:
            return self.get_formula(), {
                "binding_count": 0,
                "bytes_before": original_size,
                "bytes_after": original_size,
            }
        return shared_formula, {
            "binding_count": binding_count,
            "bytes_before": original_size,
            "bytes_after": len(shared_formula),
        }

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
