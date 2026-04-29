from __future__ import annotations

from abc import ABC, abstractmethod

from libct.state.work_item import ConstraintWorkItem


class Searcher(ABC):
    """Pending constraint selection strategy."""

    @abstractmethod
    def push(self, item: ConstraintWorkItem) -> None:
        ...

    @abstractmethod
    def pop(self) -> ConstraintWorkItem:
        ...

    @abstractmethod
    def empty(self) -> bool:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
