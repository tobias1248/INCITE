from __future__ import annotations

import pickle
import threading

import numpy as np
import pytest

from libct.constraint import Constraint
from libct.state import ConstraintWorkItem, ExecutionState, StateManager


def setup_function() -> None:
    Constraint.global_constraints.clear()


def test_execution_state_defaults_and_pickle_serializable() -> None:
    state = ExecutionState()

    payload = pickle.dumps(state)

    assert pickle.loads(payload).depth == 0
    assert state.status == "pending"
    assert state.constraints == []


def test_execution_state_fork_and_helpers() -> None:
    root_constraint = Constraint(None, None)
    branch_constraint = Constraint(root_constraint.id, None, height=1)
    state = ExecutionState(path_id="root", depth=0, constraints=[root_constraint])

    successor = state.fork(0, branch_constraint)
    state.set_shap_scores(np.array([0.1, -0.5, 0.3, 0.9]))
    state.add_coverage({"a.py:1"})
    state.add_coverage({"a.py:2"})
    state.mark_adversarial()

    assert successor.path_id == "root_0"
    assert successor.parent_id == "root"
    assert successor.depth == 1
    assert successor.constraints == [branch_constraint]
    assert state.get_max_shap_score() == pytest.approx(0.9)
    assert state.get_top_k_shap_indices(2) == [3, 1]
    assert state.coverage_count() == 2
    assert state.is_adversarial()


def test_execution_state_shap_none_returns_empty_defaults() -> None:
    state = ExecutionState(path_id="s")

    assert state.get_max_shap_score() == 0.0
    assert state.get_top_k_shap_indices(3) == []


def test_state_manager_add_get_remove_stats() -> None:
    manager = StateManager()
    state = ExecutionState(path_id="s1")
    adversarial = ExecutionState(path_id="s2")
    adversarial.mark_adversarial()

    manager.add(state)
    manager.add(adversarial)
    removed = manager.remove("s1")

    assert removed is state
    assert manager.get("s1") is None
    assert manager.get_adversarial_states() == [adversarial]
    assert manager.stats() == {
        "total_created": 2,
        "total_completed": 1,
        "active_count": 1,
        "adversarial_count": 1,
    }


def test_state_manager_duplicate_raises() -> None:
    manager = StateManager()
    manager.add(ExecutionState(path_id="dup"))

    with pytest.raises(ValueError, match="dup"):
        manager.add(ExecutionState(path_id="dup"))


def test_state_manager_thread_safety() -> None:
    manager = StateManager()

    def add_states(offset: int) -> None:
        for index in range(100):
            manager.add(ExecutionState(path_id=f"s{offset + index}"))

    threads = [threading.Thread(target=add_states, args=(i * 100,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert manager.count() == 1000
    assert manager.stats()["total_created"] == 1000


def test_constraint_work_item_preserves_constraint_metadata() -> None:
    constraint = Constraint(None, None, height=7)

    item = ConstraintWorkItem.from_constraint(
        constraint,
        position=("layer", (1, 2)),
        shap_value=-0.25,
        score=1.5,
    )

    assert item.constraint is constraint
    assert item.position == ("layer", (1, 2))
    assert item.shap_value == 0.25
    assert item.score == 1.5
    assert item.path_len == 7
