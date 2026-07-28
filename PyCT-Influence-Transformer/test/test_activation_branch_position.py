from __future__ import annotations

from dnnct.myDNN import ActivationLayer
from libct.constraint import Constraint
from libct.path import PathToConstraint
from libct.position import register_current_indices, register_current_layer_number
from libct.utils import ConcolicObject


class _Engine:
    symbolic_enabled = True

    def __init__(self) -> None:
        self.path = PathToConstraint()
        self.pushed = []

    def push_constraint(self, constraint, position) -> None:
        self.pushed.append((constraint, position))


def setup_function() -> None:
    Constraint.global_constraints.clear()
    PathToConstraint.root_constraint = None
    register_current_layer_number(0)
    register_current_indices((0,))


def test_activation_layer_registers_each_rank_one_branch_position() -> None:
    engine = _Engine()
    values = [
        ConcolicObject(-0.5, "x0_VAR", engine),
        ConcolicObject(0.5, "x1_VAR", engine),
        ConcolicObject(-1.0, "x2_VAR", engine),
    ]
    register_current_layer_number(4)

    ActivationLayer("relu").forward(values)

    assert [position for _, position in engine.pushed] == [
        (4, (0,)),
        (4, (1,)),
        (4, (2,)),
    ]
