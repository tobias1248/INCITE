"""Streaming attention preserves predicates while stopping symbolic arithmetic."""
from __future__ import annotations

import copy
import math
from types import MethodType

import numpy as np
import pytest

import dnnct.myDNN as mydnn
import libct.position as position
from libct.concolic import Concolic
from libct.concolic.float import ConcolicFloat
from libct.constraint import Constraint
from libct.path import PathToConstraint
from libct.predicate import Predicate
from libct.solver import Solver
from libct.utils import ConcolicObject, unwrap


@pytest.fixture(autouse=True)
def isolated_symbolic_runtime(monkeypatch):
    monkeypatch.setattr(Constraint, "global_constraints", [])
    monkeypatch.setattr(Solver, "safety", 0, raising=False)
    monkeypatch.setattr(position, "current_layer_number", 5, raising=False)
    monkeypatch.setattr(position, "current_indices_in_current_layer", (0,), raising=False)


class _RecorderEngine:
    def __init__(self, threshold=None):
        self.symbolic_enabled = True
        self.symbolic_disabled_at_path_len = None
        self.threshold = threshold
        self.path = PathToConstraint()
        self.records = []
        self.constraints = []
        self.on_branch = None

    def _maybe_disable_symbolic(self, height):
        if self.threshold is not None and height >= self.threshold:
            self.symbolic_enabled = False
            self.symbolic_disabled_at_path_len = height

    def push_constraint(self, constraint, where):
        self.constraints.append(constraint)
        self.records.append((
            Predicate.get_formula_deep(constraint.last_predicate.expr),
            constraint.last_predicate.value, copy.deepcopy(where),
        ))
        if self.on_branch is not None:
            self.on_branch(len(self.records))


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def numpy(self):
        return self.value


def _layer(heads=2, width=2):
    projection = np.array([
        [[0.5 + head * 0.25, -0.25 + head * 0.1] for head in range(heads)],
        [[-0.125, 0.75 - head * 0.125] for head in range(heads)],
    ])[:, :, :width]
    return mydnn.MultiHeadAttentionLayer(
        num_heads=heads, key_dim_per_heads=width,
        wq=_Tensor(projection), bq=_Tensor(np.full((heads, width), 0.1)),
        wk=_Tensor(projection * 0.75), bk=_Tensor(np.full((heads, width), -0.2)),
        wv=_Tensor(projection * -0.5), bv=_Tensor(np.full((heads, width), 0.05)),
        output_weights=_Tensor(np.arange(heads * width * 2).reshape(heads, width, 2) * 0.1 + 0.1),
        output_bias=_Tensor([0.1, -0.2]),
    )


def _legacy_multiply(left, right):
    result = [[0] * len(right[0]) for _ in range(len(left))]
    for i in range(len(left)):
        for j in range(len(right[0])):
            for k in range(len(right)):
                result[i][j] += left[i][k] * right[k][j]
    return result


def _legacy_attention(layer, Q, K, V, meta=None, mask=None):
    """Independent old full-matrix / all-max-before-context implementation."""
    scores = _legacy_multiply(Q, list(zip(*K)))
    scores = [[value / (layer.key_dim_per_heads ** 0.5) for value in row] for row in scores]
    maxima = []
    for i, row in enumerate(scores):
        maximum = row[0]
        for j, value in enumerate(row):
            layer._register_attention_position(i, j, meta)
            if value > maximum:
                maximum = value
        maxima.append(maximum)
    concrete = [[float(unwrap(value)) for value in row] for row in scores]
    maxima = [float(unwrap(value)) for value in maxima]
    exponentials = [[math.exp(value - maxima[i]) for value in row] for i, row in enumerate(concrete)]
    totals = [layer.mySum(row) for row in exponentials]
    probabilities = [[value / totals[i] for value in row] for i, row in enumerate(exponentials)]
    return _legacy_multiply(probabilities, V)


def _symbolic_input(engine, spatial=False):
    values = [[0.2, -0.1], [0.5, 0.25], [-0.3, 0.4]]
    if spatial:
        values.append([0.1, 0.6])
    symbolic = [[ConcolicObject(float(value), "X_{}_{}".format(i, j), engine)
                 for j, value in enumerate(row)] for i, row in enumerate(values)]
    return [symbolic[:2], symbolic[2:]] if spatial else symbolic


def _flatten(value):
    if isinstance(value, (list, tuple)):
        return [item for part in value for item in _flatten(part)]
    return [value]


def _snapshots(value):
    return [(id(item), float(unwrap(item)), Predicate.get_formula_deep(item.expr))
            for item in _flatten(value) if isinstance(item, Concolic)]


@pytest.mark.parametrize("spatial", [False, True])
def test_concrete_streamed_attention_is_exactly_legacy_order(spatial):
    layer = _layer()
    values = [[0.2, -0.1], [0.5, 0.25], [-0.3, 0.4]]
    if spatial:
        values = [[values[0], values[1]], [values[2], [0.1, 0.6]]]
    reference = _layer()
    reference.dot_product_attention = MethodType(_legacy_attention, reference)
    assert layer.forwardSingle(values) == reference.forwardSingle(values)


@pytest.mark.parametrize("threshold", [None, 1000, 2, 3, 9, 10])
@pytest.mark.parametrize("spatial", [False, True])
def test_symbolic_predicates_positions_and_values_match_legacy_across_heads(threshold, spatial):
    reference = _layer()
    reference.dot_product_attention = MethodType(_legacy_attention, reference)
    legacy_engine = _RecorderEngine(threshold)
    legacy_output = reference.forwardSingle(_symbolic_input(legacy_engine, spatial))
    expected_records = copy.deepcopy(legacy_engine.records)
    engine = _RecorderEngine(threshold)
    inputs = _symbolic_input(engine, spatial)
    snapshots = _snapshots(inputs)
    output = _layer().forwardSingle(inputs)

    assert [float(unwrap(value)) for value in _flatten(output)] == [
        float(unwrap(value)) for value in _flatten(legacy_output)
    ]
    assert engine.records == expected_records
    assert engine.symbolic_disabled_at_path_len == legacy_engine.symbolic_disabled_at_path_len
    assert _snapshots(inputs) == snapshots
    assert [(Predicate.get_formula_deep(c.last_predicate.expr), c.last_predicate.value)
            for c in engine.constraints] == [(record[0], record[1]) for record in engine.records]
    total = (4 if spatial else 3) ** 2 * 2
    expected_count = min(total, threshold) if threshold is not None else total
    assert len(engine.records) == expected_count


def _qkv(engine):
    Q = [[ConcolicObject(value, "Q_{}_{}".format(i, j), engine)
          for j, value in enumerate(row)] for i, row in enumerate([[0.2, 0.5], [0.3, -0.2], [-0.1, 0.4]])]
    K = [[ConcolicObject(value, "K_{}_{}".format(i, j), engine)
          for j, value in enumerate(row)] for i, row in enumerate([[0.1, 0.3], [0.2, -0.5], [-0.4, 0.6]])]
    V = [[ConcolicObject(value, "V_{}_{}".format(i, j), engine)
          for j, value in enumerate(row)] for i, row in enumerate([[0.8, 0.1], [0.2, 0.7], [-0.3, 0.4]])]
    return Q, K, V


def _meta():
    return {"mode": "sequence", "feature_dim": 2, "token_shape": (3,)}


def test_first_row_is_recorded_before_next_query_row_multiply(monkeypatch):
    engine = _RecorderEngine()
    Q, K, V = _qkv(engine)
    original = ConcolicFloat.__mul__
    next_row_calls = []

    def multiplied(value, other):
        if any(value is item for item in Q[1]):
            next_row_calls.append(len(engine.records))
        return original(value, other)

    monkeypatch.setattr(ConcolicFloat, "__mul__", multiplied)
    _layer(heads=1).dot_product_attention(Q, K, V, meta=_meta())
    assert next_row_calls
    assert min(next_row_calls) == 3


@pytest.mark.parametrize("threshold", [1, 2, 3, 4])
def test_no_symbolic_arithmetic_after_all_engines_stop(monkeypatch, threshold):
    engine = _RecorderEngine(threshold)
    Q, K, V = _qkv(engine)
    snapshot = _snapshots([Q, K, V])
    stopped_dispatches = []
    for name in ("__mul__", "__rmul__", "__add__", "__radd__", "__truediv__"):
        original = getattr(ConcolicFloat, name)

        def tracked(value, other, _original=original, _name=name):
            if not engine.symbolic_enabled:
                stopped_dispatches.append(_name)
            return _original(value, other)

        monkeypatch.setattr(ConcolicFloat, name, tracked)
    output = _layer(heads=1).dot_product_attention(Q, K, V, meta=_meta())
    assert engine.symbolic_enabled is False
    assert stopped_dispatches == []
    assert all(type(item) is float for item in _flatten(output))
    assert _snapshots([Q, K, V]) == snapshot


def test_disabled_on_entry_unwraps_stale_qkv_without_arithmetic_dispatch(monkeypatch):
    engine = _RecorderEngine()
    Q, K, V = _qkv(engine)
    snapshot = _snapshots([Q, K, V])
    engine.symbolic_enabled = False

    def forbidden(*args):
        pytest.fail("disabled symbolic operand still used in attention arithmetic")

    monkeypatch.setattr(ConcolicFloat, "__mul__", forbidden)
    monkeypatch.setattr(ConcolicFloat, "__rmul__", forbidden)
    output = _layer(heads=1).dot_product_attention(Q, K, V, meta=_meta())
    assert engine.records == []
    assert all(type(item) is float for item in _flatten(output))
    assert _snapshots([Q, K, V]) == snapshot


def test_value_only_symbolic_input_stays_symbolic_when_no_score_branches_exist():
    engine = _RecorderEngine(1)
    Q, K, V = _qkv(engine)
    Q, K = unwrap(Q), unwrap(K)
    output = _layer(heads=1).dot_product_attention(Q, K, V, meta=_meta())
    assert engine.symbolic_enabled is True
    assert engine.records == []
    assert all(isinstance(item, Concolic) for item in _flatten(output))
    assert "V_" in Predicate.get_formula_deep(output[0][0])


def test_active_value_engine_prevents_concretization_of_mixed_engines():
    stopped_engine = _RecorderEngine()
    active_engine = _RecorderEngine()
    Q, K, _ = _qkv(stopped_engine)
    _, _, V = _qkv(active_engine)
    stopped_engine.symbolic_enabled = False
    output = _layer(heads=1).dot_product_attention(Q, K, V, meta=_meta())
    assert active_engine.symbolic_enabled is True
    assert all(isinstance(item, Concolic) for item in _flatten(output))
    assert "V_" in Predicate.get_formula_deep(output[0][0])


def test_context_multiplication_starts_after_every_score_row_has_recorded_branches(monkeypatch):
    engine = _RecorderEngine()
    Q, K, V = _qkv(engine)
    original = ConcolicFloat.__rmul__
    context_branch_counts = []

    def multiplied(value, other):
        if any(value is item for row in V for item in row):
            context_branch_counts.append(len(engine.records))
        return original(value, other)

    monkeypatch.setattr(ConcolicFloat, "__rmul__", multiplied)
    _layer(heads=1).dot_product_attention(Q, K, V, meta=_meta())
    assert context_branch_counts
    assert set(context_branch_counts) == {9}


@pytest.mark.parametrize("threshold", [2, 3, 9])
def test_full_multihead_forward_avoids_stopped_arithmetic_in_remaining_heads_and_projection(
    monkeypatch, threshold
):
    engine = _RecorderEngine(threshold)
    inputs = _symbolic_input(engine)
    original = ConcolicFloat._bin_op
    stopped_dispatches = []
    arithmetic = {"__mul__", "__rmul__", "__add__", "__radd__", "__truediv__"}

    def operation(value, op, other):
        if op in arithmetic and not engine.symbolic_enabled:
            stopped_dispatches.append(op)
        return original(value, op, other)

    monkeypatch.setattr(ConcolicFloat, "_bin_op", operation)
    output = _layer().forwardSingle(inputs)
    assert engine.symbolic_enabled is False
    assert stopped_dispatches == []
    assert all(type(item) is float for item in _flatten(output))


def test_missing_engine_flag_is_not_permission_to_drop_symbolic_value():
    engine = _RecorderEngine()
    Q, K, V = _qkv(engine)
    Q, K = unwrap(Q), unwrap(K)
    del engine.symbolic_enabled
    output = _layer(heads=1).dot_product_attention(Q, K, V, meta=_meta())
    assert all(isinstance(item, Concolic) for item in _flatten(output))
    assert "V_" in Predicate.get_formula_deep(output[0][0])
