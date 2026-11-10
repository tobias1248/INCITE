"""Serialize multiple predicates using scoped, dependency-ordered SMT lets."""

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from libct.concolic import Concolic
from libct.expression_graph import ExpressionGraph, MAX_ESTIMATED_BYTES


@dataclass(frozen=True)
class SharedFormula:
    formula: str
    stats: dict


def serialize_shared_assertions(expressions: Iterable[Any], variable_types: Mapping[str, str]) -> SharedFormula:
    graph = ExpressionGraph(expressions)
    if not graph.roots:
        return SharedFormula("", {"unique_nodes": 0, "shared_nodes": 0,
                                  "input_bytes": 0, "input_bytes_capped": False,
                                  "output_bytes": 0})
    # Include identifiers in opaque SMT atoms as well as declared input names.
    # A generated binding must never capture an existing free or bound symbol.
    reserved = set(variable_types)
    for node in graph.nodes.values():
        if isinstance(node, str):
            reserved.update(token.strip("|") for token in re.findall(r"[^()\s]+", node))
    names = {}
    levels = {}
    groups = {}
    sizes = {}
    serial = 0
    for key in graph.postorder:
        node = graph.nodes[key]
        children = graph.children[key]
        level = max((levels[id(child)] for child in children), default=0)
        if isinstance(node, str):
            sizes[key] = len(node.encode("utf-8"))
        else:
            sizes[key] = sum(sizes[id(child)] for child in children)
            if isinstance(node, list):
                sizes[key] += 2 + max(0, len(node) - 1)
            sizes[key] = min(sizes[key], MAX_ESTIMATED_BYTES)
        complex_node = isinstance(node, list) or (
            isinstance(node, Concolic) and not isinstance(node.expr, str)
        )
        if complex_node and graph.references[key] > 1:
            name = "__pyct_shared_{}".format(serial)
            while name in reserved:
                serial += 1
                name = "__pyct_shared_{}".format(serial)
            serial += 1
            reserved.add(name)
            names[key] = name
            level += 1
            groups.setdefault(level, []).append(key)
        levels[key] = level

    def append_expression(parts, root, defining=None):
        # Token streaming avoids quadratic intermediate strings on deep chains.
        stack = [(root, False)]
        while stack:
            node, literal = stack.pop()
            if literal:
                parts.append(node)
                continue
            key = id(node)
            if key in names and key != defining:
                parts.append(names[key])
            elif isinstance(node, str):
                parts.append(node)
            elif isinstance(node, Concolic):
                stack.append((node.expr, False))
            else:
                parts.append("(")
                stack.append((")", True))
                for index in range(len(node) - 1, -1, -1):
                    stack.append((node[index], False))
                    if index:
                        stack.append((" ", True))

    parts = ["(assert "]
    for level in sorted(groups):
        parts.append("(let (")
        for index, key in enumerate(groups[level]):
            if index:
                parts.append(" ")
            parts.extend(("(", names[key], " "))
            append_expression(parts, graph.nodes[key], defining=key)
            parts.append(")")
        parts.append(") ")
    if len(graph.roots) > 1:
        parts.append("(and ")
    for index, root in enumerate(graph.roots):
        if index:
            parts.append(" ")
        append_expression(parts, root)
    if len(graph.roots) > 1:
        parts.append(")")
    parts.append(")" * (len(groups) + 1))
    formula = "".join(parts)
    input_bytes = min(MAX_ESTIMATED_BYTES, sum(
        sizes[id(root)] + len("(assert )") for root in graph.roots
    ))
    return SharedFormula(formula, {
        "unique_nodes": len(graph.nodes),
        "shared_nodes": len(names),
        "input_bytes": input_bytes,
        "input_bytes_capped": input_bytes == MAX_ESTIMATED_BYTES,
        "output_bytes": len(formula.encode("utf-8")),
    })
