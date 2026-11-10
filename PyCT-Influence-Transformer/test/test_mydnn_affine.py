"""Real CNN/residual forwards preserve concrete bits and branch semantics."""

import copy
import shutil
import struct
import subprocess

import numpy as np
import pytest

import dnnct.myDNN as mydnn
import libct.position as position
from libct.concolic import Concolic
from libct.concolic.float import ConcolicFloat
from libct.constraint import Constraint
from libct.path import PathToConstraint
from libct.solver import Solver
from libct.utils import unwrap


class _Engine:
    def __init__(self, threshold):
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
        self.records.append((constraint.last_predicate.value, copy.deepcopy(where),
                             len(constraint.get_all_asserts())))


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    monkeypatch.setattr(Solver, "safety", 0, raising=False)
    monkeypatch.setattr(Constraint, "global_constraints", [])
    monkeypatch.setattr(position, "my_layer_number_to_Keras_layer_number", {i: i for i in range(20)})
    monkeypatch.setattr(position, "current_layer_number", 0, raising=False)
    monkeypatch.setattr(position, "current_indices_in_current_layer", (0,), raising=False)


def _flatten(value):
    if isinstance(value, (list, tuple)):
        return [item for part in value for item in _flatten(part)]
    return [value]


def _bits(value):
    return [struct.pack("!d", float(unwrap(item))) for item in _flatten(value)]


def _model(residual, seed):
    rng = np.random.RandomState(seed)

    def conv(weight, bias):
        weights = np.asarray([[[[weight]]]])
        return mydnn.Conv2DLayer(weights, np.asarray([bias]), weights.shape)

    model = mydnn.NNModel()
    skip = model._append_layer(conv(float(rng.uniform(0.5, 1)), -0.2))
    if residual:
        model._append_layer(mydnn.BatchNormLayer([1.2], [0.1], [0.05], [0.5]))
    model._append_layer(mydnn.ActivationLayer("relu"))
    model._append_layer(conv(-0.5, 0.25))
    if residual:
        model._append_layer(mydnn.BatchNormLayer([0.8], [-0.1], [0.02], [0.75]))
    branch = model._append_layer(mydnn.ActivationLayer("relu"))
    if residual:
        model._append_layer(mydnn.AddLayer([skip, branch]))
        model._append_layer(mydnn.ActivationLayer("relu"))
    else:
        model._append_layer(mydnn.MaxPool2DLayer((2, 2)))
    model._append_layer(mydnn.FlattenLayer())
    width = 4 if residual else 1
    weights = rng.uniform(-0.5, 0.5, size=(2, width))
    model._append_layer(mydnn.DenseLayer(weights, np.asarray([0.1, -0.05]), weights.shape))
    return model


def _inputs(engine, seed):
    rng = np.random.RandomState(seed)
    shared = ConcolicFloat(float(rng.uniform(-0.1, 0.1)), "x_VAR", engine)
    pixels = rng.uniform(0.05, 0.95, size=(2, 2))
    return [[[(shared * (1.0 if (row + col) % 2 else -0.5)) + float(pixels[row, col])]
             for col in range(2)] for row in range(2)]


def _assert_branch_equivalence(left, right):
    executable = shutil.which("cvc5")
    if executable is None:
        pytest.skip("cvc5 is required for CNN compact branch equivalence")
    differences = []
    for first, second in zip(left.constraints, right.constraints):
        first_prefix, second_prefix = first.get_all_asserts(), second.get_all_asserts()
        assert len(first_prefix) == len(second_prefix)
        for a, b in zip(first_prefix, second_prefix):
            differences.append("(xor {} {})".format(a.get_formula()[8:-1], b.get_formula()[8:-1]))
    if not differences:
        return
    query = "\n".join(["(set-logic ALL)", "(declare-const x_VAR Real)",
                       "(assert (or {}))".format(" ".join(differences)), "(check-sat)"])
    result = subprocess.run([executable, "--lang=smt2", "--quiet"], input=query,
                            text=True, capture_output=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "unsat", result.stdout + result.stderr


@pytest.mark.parametrize("residual", [False, True], ids=["cnn", "residual"])
@pytest.mark.parametrize("seed", [0, 4, 17])
@pytest.mark.parametrize("threshold", [None, 0, 1, 3, 7, 1000])
def test_shared_variable_cnn_outputs_and_complete_branch_prefix_match_legacy(
    monkeypatch, residual, seed, threshold,
):
    runs = []
    for representation in ("legacy", "compact-affine"):
        monkeypatch.setenv("PYCT_SYMBOLIC_REPRESENTATION", representation)
        monkeypatch.setattr(Constraint, "global_constraints", [])
        engine = _Engine(threshold)
        model = _model(residual, seed)
        inputs = _inputs(engine, seed)
        original_bits = _bits(inputs)
        output = model.forward(inputs)
        assert _bits(inputs) == original_bits
        runs.append((engine, model, output))

    legacy, compact = runs
    assert _bits(legacy[2]) == _bits(compact[2])
    assert legacy[0].records == compact[0].records
    assert legacy[0].symbolic_enabled == compact[0].symbolic_enabled
    assert legacy[0].symbolic_disabled_at_path_len == compact[0].symbolic_disabled_at_path_len
    assert legacy[0].path.current_constraint.height == compact[0].path.current_constraint.height
    _assert_branch_equivalence(legacy[0], compact[0])
    if not compact[0].symbolic_enabled:
        assert all(type(item) is float for item in _flatten(compact[2]))
    assert any(isinstance(item, Concolic) and "_affine_form" in item.__dict__
               for item in _flatten(compact[1].layers[0].getOutput()))
    if residual:
        assert _bits(legacy[1].layers[0].getOutput()) == _bits(compact[1].layers[0].getOutput())
