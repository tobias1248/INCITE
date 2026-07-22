from __future__ import annotations

from libct.constraint import Constraint
from libct.path import PathToConstraint
from libct.position import register_current_indices, register_current_layer_number
from libct.utils import ConcolicObject


class _Engine:
    symbolic_enabled = True

    def __init__(self) -> None:
        self.path = PathToConstraint()
        self.pushed = []
        self.trivial_branch_pruned_count = 0

    def push_constraint(self, constraint, position) -> None:
        self.pushed.append((constraint, position))


def setup_function() -> None:
    Constraint.global_constraints.clear()
    PathToConstraint.root_constraint = None
    register_current_layer_number(0)
    register_current_indices((0,))


def test_reflexive_false_comparison_is_pruned() -> None:
    engine = _Engine()
    value = ConcolicObject(0.25, "x_VAR", engine)

    engine.path.add_branch(value > value)

    assert engine.trivial_branch_pruned_count == 1
    assert engine.pushed == []
    assert engine.path.current_constraint is engine.path.root_constraint
    assert engine.path.root_constraint.children == []


def test_reflexive_true_comparison_is_pruned() -> None:
    engine = _Engine()
    value = ConcolicObject(0.25, "x_VAR", engine)

    engine.path.add_branch(value <= value)

    assert engine.trivial_branch_pruned_count == 1
    assert engine.pushed == []
    assert engine.path.root_constraint.children == []


def test_non_reflexive_comparison_still_creates_both_paths() -> None:
    engine = _Engine()
    left = ConcolicObject(0.75, "left_VAR", engine)
    right = ConcolicObject(0.25, "right_VAR", engine)

    engine.path.add_branch(left > right)

    assert engine.trivial_branch_pruned_count == 0
    assert len(engine.pushed) == 1
    assert len(engine.path.root_constraint.children) == 2
    assert engine.path.current_constraint.height == 1


def test_reflexive_comparison_with_non_real_concrete_semantics_is_kept() -> None:
    engine = _Engine()
    value = ConcolicObject(float("nan"), "x_VAR", engine)

    engine.path.add_branch(value >= value)

    assert engine.trivial_branch_pruned_count == 0
    assert len(engine.pushed) == 1
    assert len(engine.path.root_constraint.children) == 2
