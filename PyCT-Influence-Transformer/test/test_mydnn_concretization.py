"""Layer boundary concretization preserves shallow CNN paths and residual inputs."""
from __future__ import annotations

import copy

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
    monkeypatch.setattr(position, "my_layer_number_to_Keras_layer_number", {
        i: i for i in range(20)
    })
    monkeypatch.setattr(position, "current_layer_number", 0, raising=False)
    monkeypatch.setattr(position, "current_indices_in_current_layer", (0,), raising=False)


class _RecorderEngine:
    def __init__(self, threshold=None):
        self.symbolic_enabled = True
        self.symbolic_disabled_at_path_len = None
        self.threshold = threshold
        self.path = PathToConstraint()
        self.records = []
        self.constraints = []

    def _maybe_disable_symbolic(self, height):
        if self.threshold is not None and height >= self.threshold:
            self.symbolic_enabled = False
            self.symbolic_disabled_at_path_len = height

    def push_constraint(self, constraint, where):
        self.constraints.append(constraint)
        self.records.append((
            constraint.last_predicate.get_formula(),
            constraint.last_predicate.value,
            copy.deepcopy(where),
            tuple(predicate.get_formula() for predicate in constraint.get_all_asserts()),
        ))


def _flatten(tensor):
    if isinstance(tensor, (list, tuple)):
        return [value for part in tensor for value in _flatten(part)]
    return [tensor]


def _numbers(tensor):
    return [float(unwrap(value)) for value in _flatten(tensor)]


def _snapshots(tensor):
    return [(id(value), float(unwrap(value)), Predicate.get_formula_deep(value.expr))
            for value in _flatten(tensor) if isinstance(value, Concolic)]


def _symbolic_image(engine):
    return [[[ConcolicObject(value, "X_{}_{}".format(row, col), engine)]
             for col, value in enumerate(values)]
            for row, values in enumerate([[0.2, -0.4], [0.7, -0.1]])]


def _conv(weight, bias=0.0):
    weights = np.array([[[[weight]]]])
    return mydnn.Conv2DLayer(weights, np.array([bias]), weights.shape)


def _dense(width):
    weights = np.array([[0.3] * width, [-0.2] * width])
    return mydnn.DenseLayer(weights, np.array([0.1, -0.05]), weights.shape)


def _vgg():
    model = mydnn.NNModel()
    for layer in (
        _conv(0.75, 0.05), mydnn.ActivationLayer("relu"),
        _conv(-0.5, 0.25), mydnn.ActivationLayer("relu"),
        mydnn.MaxPool2DLayer((2, 2)), mydnn.FlattenLayer(), _dense(1),
    ):
        model._append_layer(layer)
    return model


def _resnet():
    model = mydnn.NNModel()
    skip = model._append_layer(_conv(0.75, 0.05))
    norm = mydnn.BatchNormLayer([1.2], [0.1], [0.05], [0.5])
    norm.input_from = [skip]
    model._append_layer(norm)
    model._append_layer(mydnn.ActivationLayer("relu"))
    model._append_layer(_conv(-0.5, 0.25))
    model._append_layer(mydnn.BatchNormLayer([0.8], [-0.1], [0.02], [0.75]))
    branch = model._append_layer(mydnn.ActivationLayer("relu"))
    model._append_layer(mydnn.AddLayer([skip, branch]))
    model._append_layer(mydnn.ActivationLayer("relu"))
    model._append_layer(mydnn.FlattenLayer())
    model._append_layer(_dense(4))
    return model


def _legacy_forward(model, inputs):
    """Run original routing without the new boundary conversion helpers."""
    saved = {"layer_input": inputs}
    output = inputs
    for index, layer in enumerate(model.layers):
        position.register_current_layer_number(position.to_Keras_layer_number(index))
        sources = getattr(layer, "input_from", None)
        if sources:
            inbound = [saved[name] for name in sources]
            output = layer.forward(inbound if getattr(layer, "multi_input", False) else inbound[0])
        else:
            output = layer.forward(output)
        saved[model.my_layer_keys[index]] = output
    return output


def _recorded_constraints(engine):
    return [(
        constraint.last_predicate.get_formula(), constraint.last_predicate.value,
        tuple(predicate.get_formula() for predicate in constraint.get_all_asserts()),
    ) for constraint in engine.constraints]


@pytest.mark.parametrize("factory", [_vgg, _resnet], ids=["vgg", "resnet"])
@pytest.mark.parametrize("threshold", [None, 1000, 1, 3, 4, 5, 9])
def test_cnn_values_and_complete_branch_prefix_match_legacy(factory, threshold):
    reference = factory()
    reference_engine = _RecorderEngine(threshold)
    expected = _legacy_forward(reference, _symbolic_image(reference_engine))
    expected_records = copy.deepcopy(reference_engine.records)
    expected_constraints = _recorded_constraints(reference_engine)

    model = factory()
    engine = _RecorderEngine(threshold)
    inputs = _symbolic_image(engine)
    original_inputs = _snapshots(inputs)
    output = model.forward(inputs)

    assert _numbers(output) == _numbers(expected)
    assert engine.records == expected_records
    assert _recorded_constraints(engine) == expected_constraints
    assert engine.symbolic_disabled_at_path_len == reference_engine.symbolic_disabled_at_path_len
    assert engine.path.current_constraint.height == reference_engine.path.current_constraint.height
    assert _snapshots(inputs) == original_inputs
    # The skip output is retained on its producer and can occur in saved formulas.
    assert [entry[1:] for entry in _snapshots(model.layers[0].getOutput())] == [
        entry[1:] for entry in _snapshots(reference.layers[0].getOutput())
    ]
    if engine.symbolic_enabled is False:
        assert all(not isinstance(value, Concolic) for value in _flatten(output))


@pytest.mark.parametrize("factory, first_concrete", [(_vgg, 2), (_resnet, 3)],
                         ids=["vgg", "resnet"])
def test_later_cnn_layers_do_not_dispatch_symbolic_arithmetic(monkeypatch, factory, first_concrete):
    model = factory()
    engine = _RecorderEngine(1)
    inputs = _symbolic_image(engine)
    dispatches = []
    inbound_types = {}
    for index, layer in enumerate(model.layers):
        forward = layer.forward

        def observed(value, _forward=forward, _index=index):
            inbound_types[_index] = [type(item) for item in _flatten(value)]
            return _forward(value)

        monkeypatch.setattr(layer, "forward", observed)
    for name in ("__mul__", "__rmul__", "__add__", "__radd__", "__truediv__", "__gt__", "__lt__"):
        original = getattr(ConcolicFloat, name)

        def tracked(value, other, _original=original, _name=name):
            dispatches.append((position.current_layer_number, _name, engine.symbolic_enabled))
            return _original(value, other)

        monkeypatch.setattr(ConcolicFloat, name, tracked)

    model.forward(inputs)
    assert engine.symbolic_disabled_at_path_len == 1
    assert dispatches
    # Stopping within ReLU does not unwrap its remaining elements mid-layer.
    assert any(not enabled for layer, name, enabled in dispatches)
    assert not [(layer, name) for layer, name, enabled in dispatches if layer >= first_concrete]
    assert all(ConcolicFloat not in types for index, types in inbound_types.items()
               if index >= first_concrete)
    if factory is _resnet:
        assert any(isinstance(value, Concolic) for value in _flatten(model.layers[0].getOutput()))
        assert all(value is float for value in inbound_types[6])  # Add's stale skip is converted.


class _Observe:
    def __init__(self):
        self.seen = None

    def forward(self, value):
        self.seen = value
        return value


@pytest.mark.parametrize("other_state", [True, "missing", False])
def test_all_original_engines_must_explicitly_stop(other_state):
    first = _RecorderEngine()
    second = _RecorderEngine()
    inputs = [ConcolicObject(0.1, "A", first), ConcolicObject(0.2, "B", second)]
    first.symbolic_enabled = False
    if other_state == "missing":
        del second.symbolic_enabled
    else:
        second.symbolic_enabled = other_state
    model = mydnn.NNModel()
    observer = _Observe()
    model._append_layer(observer)
    snapshot = _snapshots(inputs)
    output = model.forward(inputs)
    if other_state is False:
        assert observer.seen is not inputs
        assert all(type(value) is float for value in output)
    else:
        assert observer.seen is inputs
        assert all(isinstance(value, Concolic) for value in output)
    assert _snapshots(inputs) == snapshot


def test_engine_is_remembered_when_layer_output_drops_owner():
    engine = _RecorderEngine(0)
    inputs = [ConcolicObject(0.25, "X", engine)]

    class StopAndMultiply:
        def forward(self, value):
            assert value[0] > 0  # Disables tracking at depth zero.
            # An ownerless runtime wrapper still needs boundary conversion.
            self.output = [ConcolicObject(float(unwrap(value[0])) * 0.5)]
            assert isinstance(self.output[0], Concolic)
            assert self.output[0].engine is None
            return self.output

    model = mydnn.NNModel()
    producer = StopAndMultiply()
    observer = _Observe()
    model._append_layer(producer)
    model._append_layer(observer)
    assert model.forward(inputs) == [0.125]
    assert type(observer.seen[0]) is float
    assert isinstance(producer.output[0], Concolic)


def test_repeated_forward_resets_engines_and_concrete_cache():
    model = _resnet()
    stopped = _RecorderEngine(1)
    stopped_output = model.forward(_symbolic_image(stopped))
    assert all(type(value) is float for value in stopped_output)

    active = _RecorderEngine()
    active_inputs = _symbolic_image(active)
    active_output = model.forward(active_inputs)
    reference = _resnet()
    reference_engine = _RecorderEngine()
    expected = _legacy_forward(reference, _symbolic_image(reference_engine))
    assert active.symbolic_enabled is True
    assert active.records == reference_engine.records
    assert _numbers(active_output) == _numbers(expected)
    assert all(isinstance(value, Concolic) for value in active_output)


@pytest.mark.parametrize("container", [list, tuple])
def test_plain_inputs_preserve_identity_and_disabled_inputs_preserve_container_type(container):
    observer = _Observe()
    model = mydnn.NNModel()
    model._append_layer(observer)
    plain = container([container([0.25, -0.5])])
    assert model.forward(plain) is plain
    assert observer.seen is plain

    engine = _RecorderEngine()
    symbolic = container([container([ConcolicObject(0.25, "X", engine)])])
    snapshot = _snapshots(symbolic)
    engine.symbolic_enabled = False
    output = model.forward(symbolic)
    assert type(output) is container
    assert type(output[0]) is container
    assert output is not symbolic
    assert output[0] is not symbolic[0]
    assert type(output[0][0]) is float
    assert _snapshots(symbolic) == snapshot


def test_repeated_residual_sources_reuse_copy_without_mutating_saved_output():
    engine = _RecorderEngine()
    inputs = [ConcolicObject(0.25, "X", engine)]
    snapshot = _snapshots(inputs)

    class Stop:
        def forward(self, value):
            engine.symbolic_enabled = False
            return [0.0]

    model = mydnn.NNModel()
    model._append_layer(Stop())
    first = _Observe()
    first.input_from = ["layer_input"]
    model._append_layer(first)
    second = _Observe()
    second.input_from = ["layer_input"]
    model._append_layer(second)
    model.forward(inputs)
    assert first.seen is second.seen
    assert first.seen is not inputs
    assert all(type(value) is float for value in first.seen)
    assert _snapshots(inputs) == snapshot
