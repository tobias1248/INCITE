"""Iterative operations on the existing, identity-shared expression graph.

All memo tables belong to a single operation or query. They retain their keys'
objects so Python object ID reuse cannot make a cached result refer to a new node.
"""

from typing import Any, Dict, Iterable, Optional

from libct.concolic import Concolic


MAX_ESTIMATED_BYTES = (1 << 63) - 1


def expression_children(node):
    if isinstance(node, Concolic):
        return (node.expr,)
    if isinstance(node, str):
        return ()
    if isinstance(node, list):
        return tuple(node)
    raise TypeError("unsupported expression node: {}".format(type(node).__name__))


class ExpressionGraph:
    """An acyclic postorder graph, including incoming edge multiplicities."""

    def __init__(self, roots: Iterable[Any], max_nodes: int = 1_000_000):
        self.roots = tuple(roots)
        self.nodes: Dict[int, Any] = {}
        self.children = {}
        self.references: Dict[int, int] = {}
        self.postorder = []
        active = set()
        for root in self.roots:
            key = id(root)
            self.references[key] = self.references.get(key, 0) + 1
            stack = [(root, False)]
            while stack:
                node, exiting = stack.pop()
                key = id(node)
                if exiting:
                    active.remove(key)
                    self.postorder.append(key)
                    continue
                if key in active:
                    raise ValueError("cycle in symbolic expression graph")
                if key in self.nodes:
                    continue
                if len(self.nodes) >= max_nodes:
                    raise ValueError("symbolic expression graph exceeds node budget")
                children = expression_children(node)
                self.nodes[key] = node
                self.children[key] = children
                active.add(key)
                stack.append((node, True))
                for child in reversed(children):
                    child_key = id(child)
                    self.references[child_key] = self.references.get(child_key, 0) + 1
                    stack.append((child, False))


def expanded_formula_size(expr, memo: Optional[dict] = None) -> int:
    """Compute raw UTF-8 size with no recursion or expanded intermediate text."""
    if memo is None:
        memo = {}
    active = set()
    stack = [(expr, False)]
    while stack:
        node, exiting = stack.pop()
        key = id(node)
        if not exiting and key in memo:
            continue
        if exiting:
            children = expression_children(node)
            size = sum(memo[id(child)][1] for child in children)
            if isinstance(node, list):
                size += 2 + max(0, len(node) - 1)
            memo[key] = (node, min(size, MAX_ESTIMATED_BYTES))
            active.remove(key)
            continue
        if key in active:
            raise ValueError("cycle in symbolic expression graph")
        if isinstance(node, str):
            memo[key] = (node, len(node.encode("utf-8")))
            continue
        children = expression_children(node)
        active.add(key)
        stack.append((node, True))
        stack.extend((child, False) for child in reversed(children))
    return memo[id(expr)][1]


def expressions_equal(left, right) -> bool:
    """Compare syntax, not algebra or branch positions; visit each pair once."""
    seen = set()
    stack = [(left, right)]
    while stack:
        first, second = stack.pop()
        if first is second:
            continue
        key = (id(first), id(second))
        if key in seen:
            continue
        seen.add(key)
        if isinstance(first, Concolic) and isinstance(second, Concolic):
            stack.append((first.expr, second.expr))
        elif isinstance(first, list) and isinstance(second, list):
            if len(first) != len(second):
                return False
            stack.extend(reversed(list(zip(first, second))))
        elif not (first == second):
            return False
    return True
