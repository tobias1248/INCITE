from __future__ import annotations

from collections import deque
import heapq
import random
from typing import Deque, List, Literal, Optional, Tuple

from libct.searcher.base import Searcher
from libct.state.work_item import ConstraintWorkItem


class StackSearcher(Searcher):
    def __init__(self) -> None:
        self._items: List[ConstraintWorkItem] = []

    def push(self, item: ConstraintWorkItem) -> None:
        self._items.append(item)

    def pop(self) -> ConstraintWorkItem:
        if not self._items:
            raise IndexError("StackSearcher: no constraints to pop")
        return self._items.pop()

    def empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)


class QueueSearcher(Searcher):
    def __init__(self) -> None:
        self._items: Deque[ConstraintWorkItem] = deque()

    def push(self, item: ConstraintWorkItem) -> None:
        self._items.append(item)

    def pop(self) -> ConstraintWorkItem:
        if not self._items:
            raise IndexError("QueueSearcher: no constraints to pop")
        return self._items.popleft()

    def empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)


class PrioritySearcher(Searcher):
    def __init__(self) -> None:
        self._items: List[Tuple[float, int, ConstraintWorkItem]] = []

    def push(self, item: ConstraintWorkItem) -> None:
        if item.score is None:
            raise ValueError("PrioritySearcher requires ConstraintWorkItem.score")
        heapq.heappush(self._items, (-item.score, item.constraint.id, item))

    def pop(self) -> ConstraintWorkItem:
        if not self._items:
            raise IndexError("PrioritySearcher: no constraints to pop")
        _, _, item = heapq.heappop(self._items)
        return item

    def empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)


class RandomSearcher(Searcher):
    def __init__(self, seed: Optional[int] = None) -> None:
        self._items: List[ConstraintWorkItem] = []
        self._random = random.Random(seed)

    def push(self, item: ConstraintWorkItem) -> None:
        self._items.append(item)

    def pop(self) -> ConstraintWorkItem:
        if not self._items:
            raise IndexError("RandomSearcher: no constraints to pop")
        index = self._random.randrange(len(self._items))
        return self._items.pop(index)

    def empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)


def create_constraint_searcher(
    mode: Literal["stack", "queue", "priority_queue"],
) -> Searcher:
    if mode == "stack":
        return StackSearcher()
    if mode == "queue":
        return QueueSearcher()
    if mode == "priority_queue":
        return PrioritySearcher()
    raise ValueError(f"Unsupported constraint search mode: {mode}")
