from __future__ import annotations

import pytest

from libct.constraint import Constraint
from libct.searcher import (
    PrioritySearcher,
    QueueSearcher,
    RandomSearcher,
    StackSearcher,
    create_constraint_searcher,
)
from libct.state import ConstraintWorkItem


def setup_function() -> None:
    Constraint.global_constraints.clear()


def _item(height: int, score: float = 0.0) -> ConstraintWorkItem:
    constraint = Constraint(None, None, height=height)
    return ConstraintWorkItem.from_constraint(constraint, shap_value=score, score=score)


def test_stack_searcher_lifo_order() -> None:
    searcher = StackSearcher()
    items = [_item(1), _item(2), _item(3)]
    for item in items:
        searcher.push(item)

    assert [searcher.pop().constraint.height for _ in range(3)] == [3, 2, 1]
    assert searcher.empty()


def test_queue_searcher_fifo_order() -> None:
    searcher = QueueSearcher()
    items = [_item(1), _item(2), _item(3)]
    for item in items:
        searcher.push(item)

    assert [searcher.pop().constraint.height for _ in range(3)] == [1, 2, 3]
    assert searcher.empty()


def test_priority_searcher_selects_highest_score_and_len() -> None:
    searcher = PrioritySearcher()
    searcher.push(_item(1, score=0.2))
    searcher.push(_item(2, score=3.0))
    searcher.push(_item(3, score=1.0))

    assert len(searcher) == 3
    assert searcher.pop().score == 3.0
    assert searcher.pop().score == 1.0
    assert searcher.pop().score == 0.2


def test_priority_searcher_requires_score() -> None:
    searcher = PrioritySearcher()
    constraint = Constraint(None, None)

    with pytest.raises(ValueError, match="requires"):
        searcher.push(ConstraintWorkItem.from_constraint(constraint))


def test_random_searcher_reproducible_with_seed() -> None:
    first = RandomSearcher(seed=7)
    second = RandomSearcher(seed=7)
    first_items = [_item(i, score=float(i)) for i in range(6)]
    second_items = [_item(i, score=float(i)) for i in range(6, 12)]
    for item in first_items:
        first.push(item)
    for item in second_items:
        second.push(item)

    first_order = [first.pop().score for _ in range(6)]
    second_order = [second.pop().score - 6 for _ in range(6)]

    assert first_order == second_order


def test_empty_searchers_raise_index_error() -> None:
    for searcher in (StackSearcher(), QueueSearcher(), PrioritySearcher(), RandomSearcher()):
        assert searcher.empty()
        with pytest.raises(IndexError):
            searcher.pop()


def test_create_constraint_searcher_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_constraint_searcher("unknown")  # type: ignore[arg-type]
