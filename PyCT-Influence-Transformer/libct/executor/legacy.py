from __future__ import annotations

from typing import Any, Dict

from libct.executor.base import Executor


class LegacyConcolicExecutor(Executor):
    """Adapter around the existing ExplorationEngine execution methods."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def run_concolic(self, all_args: Dict[str, Any], concolic_dict: Dict[str, Any]) -> Any:
        return self._engine._one_execution_concolic(all_args, concolic_dict)

    def run_primitive(self, primitive_inputs: Dict[str, Any]) -> Any:
        return self._engine._one_execution_primitive(primitive_inputs)
