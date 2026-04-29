from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Executor(ABC):
    """Execution boundary used by ExplorationEngine."""

    @abstractmethod
    def run_concolic(self, all_args: Dict[str, Any], concolic_dict: Dict[str, Any]) -> Any:
        ...

    @abstractmethod
    def run_primitive(self, primitive_inputs: Dict[str, Any]) -> Any:
        ...
