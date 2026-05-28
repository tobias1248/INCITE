from __future__ import annotations

import heapq
from collections import deque
from typing import Any

import pytest

from libct.constraint import Constraint
from libct.searcher import ConstraintScheduler, create_constraint_searcher


class _Recorder:
    def __init__(self) -> None:
        self.queue_last = 0
        self.queue_max = 0


class _Engine:
    SHAP_SCORE_EPS = 1e-12

    def __init__(self) -> None:
        self.idx = 3
        self.constraints_collection_type = "priority_queue"
        self.constraints_to_solve = create_constraint_searcher("priority_queue")
        self.comparator = None
        self.shap_score_alpha = 0.5
        self.constraint_log_enabled = False
        self.popped_log_attack_mode = "test"
        self.recorder = _Recorder()

    def _get_recorder(self) -> _Recorder:
        return self.recorder


class _Comparator:
    def get_shap_influence(self, layer_number: Any, indices: Any) -> float:
        assert layer_number == "layer"
        assert indices == (1, 2)
        return -0.75


def setup_function() -> None:
    Constraint.global_constraints.clear()


def test_scheduler_push_pop_priority_modular_searcher_records_metadata() -> None:
    engine = _Engine()
    engine.comparator = _Comparator()
    scheduler = ConstraintScheduler(engine)
    constraint = Constraint(None, None, height=2)

    scheduler.push_constraint(constraint, ("layer", (1, 2)))
    popped_constraint, shap_value, position = scheduler.pop_constraint()

    assert popped_constraint is constraint
    assert shap_value == pytest.approx(0.75)
    assert position == ("layer", (1, 2))
    assert len(engine.constraints_to_solve) == 0
    assert engine.recorder.queue_last == 1
    assert engine.recorder.queue_max == 1


def test_scheduler_stack_and_queue_modes_return_constraints() -> None:
    stack_engine = _Engine()
    stack_engine.constraints_collection_type = "stack"
    stack_engine.constraints_to_solve = create_constraint_searcher("stack")
    stack_scheduler = ConstraintScheduler(stack_engine)

    queue_engine = _Engine()
    queue_engine.constraints_collection_type = "queue"
    queue_engine.constraints_to_solve = create_constraint_searcher("queue")
    queue_scheduler = ConstraintScheduler(queue_engine)

    first = Constraint(None, None, height=1)
    second = Constraint(None, None, height=2)
    stack_scheduler.push_constraint(first, None)
    stack_scheduler.push_constraint(second, None)
    queue_scheduler.push_constraint(first, None)
    queue_scheduler.push_constraint(second, None)

    assert stack_scheduler.pop_constraint() is second
    assert queue_scheduler.pop_constraint() is first


def test_scheduler_preserves_legacy_priority_heap_shape() -> None:
    engine = _Engine()
    engine.constraints_to_solve = []
    scheduler = ConstraintScheduler(engine)
    constraint = Constraint(None, None, height=4)

    heapq.heappush(
        engine.constraints_to_solve,
        (-0.5, constraint.id, ("layer", (0,)), constraint, 0.5),
    )
    popped_constraint, shap_value, position = scheduler.pop_constraint()

    assert popped_constraint is constraint
    assert shap_value == 0.5
    assert position == ("layer", (0,))


def test_scheduler_preserves_legacy_queue_shape() -> None:
    engine = _Engine()
    engine.constraints_collection_type = "queue"
    engine.constraints_to_solve = deque()
    scheduler = ConstraintScheduler(engine)
    first = Constraint(None, None, height=1)
    second = Constraint(None, None, height=2)

    engine.constraints_to_solve.append(first)
    engine.constraints_to_solve.append(second)

    assert scheduler.pop_constraint() is first


def test_scheduler_priority_requires_score_alpha() -> None:
    engine = _Engine()
    engine.shap_score_alpha = None
    scheduler = ConstraintScheduler(engine)

    with pytest.raises(ValueError, match="shap_score_alpha"):
        scheduler.push_constraint(Constraint(None, None), None)
