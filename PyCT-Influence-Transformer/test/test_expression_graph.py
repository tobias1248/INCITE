import pytest

from libct.concolic import Concolic
from libct.expression_graph import MAX_ESTIMATED_BYTES, ExpressionGraph, expanded_formula_size, expressions_equal
from libct.predicate import Predicate


def test_graph_keeps_identity_sharing_and_edge_multiplicity():
    shared = ["+", "x_VAR", "1"]
    separate = ["+", "x_VAR", "1"]
    root = ["+", shared, shared, separate]
    graph = ExpressionGraph([root, shared])

    assert graph.references[id(shared)] == 3
    assert graph.references[id(separate)] == 1
    assert id(shared) != id(separate)
    assert len(graph.postorder) == len(graph.nodes)
    positions = {key: index for index, key in enumerate(graph.postorder)}
    for key, children in graph.children.items():
        assert all(positions[id(child)] < positions[key] for child in children)


def test_size_counts_utf8_and_concolic_wrappers_without_counting_unique_only():
    wrapper = Concolic()
    wrapper.expr = ["+", "é_VAR", "(- (/ 1 3))"]
    root = ["=", wrapper, wrapper]
    memo = {}
    assert expanded_formula_size(root, memo) == len(Predicate.get_formula_deep(root).encode("utf-8"))
    assert memo[id(wrapper)][0] is wrapper
    assert expanded_formula_size(root, memo) == memo[id(root)][1]


def test_deep_dag_size_and_equality_are_iterative_and_memoized():
    left = "x_VAR"
    right = "x_VAR"
    for _ in range(1500):
        left = ["+", left, left]
        right = ["+", right, right]
    assert expanded_formula_size(left) == MAX_ESTIMATED_BYTES
    assert expressions_equal(left, right)
    assert Predicate(left, True) == Predicate(right, True)
    assert Predicate(left, True) != Predicate(right, False)
    right[1] = "y_VAR"
    assert not expressions_equal(left, right)


def test_expanded_size_is_exact_below_cap_and_saturates_at_cap():
    expression = "x_VAR"
    for depth in range(1, 81):
        expression = ["+", expression, expression]
        # size(n + 1) = 2 * size(n) + len('(+  )').
        exact_size = (len("x_VAR") + 5) * 2 ** depth - 5
        assert expanded_formula_size(expression) == min(exact_size, MAX_ESTIMATED_BYTES)


def test_syntax_equality_preserves_operand_order_and_concolic_structure():
    first = Concolic()
    first.expr = ["+", "x_VAR", "1"]
    second = Concolic()
    second.expr = ["+", "x_VAR", "1"]
    assert expressions_equal(first, second)
    assert not expressions_equal(first, ["+", "x_VAR", "1"])
    assert not expressions_equal(first.expr, ["+", "1", "x_VAR"])
    assert not expressions_equal(["+", "x_VAR"], ["+", "x_VAR", "0"])


def test_graph_rejects_cycles_and_honors_node_budget():
    cycle = ["+"]
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cycle"):
        ExpressionGraph([cycle])
    with pytest.raises(ValueError, match="cycle"):
        expanded_formula_size(cycle)
    with pytest.raises(ValueError, match="node budget"):
        ExpressionGraph([["+", "x_VAR", "1"]], max_nodes=2)
