from __future__ import annotations

import threading
from typing import Dict, Iterator, List, Optional

from libct.state.execution_state import ExecutionState


class StateManager:
    """Thread-safe active state registry."""

    def __init__(self) -> None:
        self._states: Dict[str, ExecutionState] = {}
        self._lock = threading.Lock()
        self._total_created = 0
        self._total_completed = 0

    def add(self, state: ExecutionState) -> None:
        with self._lock:
            if state.path_id in self._states:
                raise ValueError(f"StateManager: path_id '{state.path_id}' already exists")
            self._states[state.path_id] = state
            self._total_created += 1

    def add_batch(self, states: List[ExecutionState]) -> None:
        with self._lock:
            for state in states:
                if state.path_id in self._states:
                    raise ValueError(f"StateManager: path_id '{state.path_id}' already exists")
                self._states[state.path_id] = state
                self._total_created += 1

    def get(self, path_id: str) -> Optional[ExecutionState]:
        with self._lock:
            return self._states.get(path_id)

    def remove(self, path_id: str) -> Optional[ExecutionState]:
        with self._lock:
            state = self._states.pop(path_id, None)
            if state is not None:
                self._total_completed += 1
            return state

    def remove_batch(self, path_ids: List[str]) -> int:
        removed = 0
        with self._lock:
            for path_id in path_ids:
                if self._states.pop(path_id, None) is not None:
                    removed += 1
                    self._total_completed += 1
        return removed

    def get_by_status(self, status: str) -> List[ExecutionState]:
        with self._lock:
            return [state for state in self._states.values() if state.status == status]

    def get_adversarial_states(self) -> List[ExecutionState]:
        return self.get_by_status("adversarial")

    def get_all(self) -> List[ExecutionState]:
        with self._lock:
            return list(self._states.values())

    def count(self) -> int:
        with self._lock:
            return len(self._states)

    def clear(self) -> None:
        with self._lock:
            self._states.clear()

    def stats(self) -> dict:
        with self._lock:
            adversarial = sum(1 for state in self._states.values() if state.is_adversarial())
            return {
                "total_created": self._total_created,
                "total_completed": self._total_completed,
                "active_count": len(self._states),
                "adversarial_count": adversarial,
            }

    def __iter__(self) -> Iterator[ExecutionState]:
        return iter(self.get_all())

    def __len__(self) -> int:
        return self.count()
