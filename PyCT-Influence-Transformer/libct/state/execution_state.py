from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from libct.constraint import Constraint
from libct.path import PathToConstraint


@dataclass
class ExecutionState:
    """Structured path state for the modular runtime."""

    path_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    depth: int = 0
    input_data: Optional[Any] = None
    output: Optional[Any] = None
    parent_id: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)
    constraints: List[Constraint] = field(default_factory=list)
    path_constraint: Optional[PathToConstraint] = None
    shap_scores: Optional[np.ndarray] = None
    shap_computed: bool = False
    coverage: Optional[Set[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def get_constraint_count(self) -> int:
        return len(self.constraints)

    def fork(self, branch_idx: int, negated_constraint: Constraint) -> "ExecutionState":
        return ExecutionState(
            path_id=f"{self.path_id}_{branch_idx}",
            depth=self.depth + 1,
            parent_id=self.path_id,
            constraints=self.constraints[:branch_idx] + [negated_constraint],
        )

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def record_execution_time(self, elapsed_seconds: float) -> None:
        self.set_metadata("execution_time_s", elapsed_seconds)

    def get_execution_time(self) -> Optional[float]:
        return self.get_metadata("execution_time_s")

    def set_shap_scores(self, scores: np.ndarray) -> None:
        self.shap_scores = scores
        self.shap_computed = True

    def get_max_shap_score(self) -> float:
        if self.shap_scores is None or len(self.shap_scores) == 0:
            return 0.0
        return float(np.max(np.abs(self.shap_scores)))

    def get_top_k_shap_indices(self, k: int) -> List[int]:
        if self.shap_scores is None:
            return []
        flat_scores = np.abs(self.shap_scores).flatten()
        top_k = min(k, len(flat_scores))
        return [int(i) for i in np.argsort(flat_scores)[::-1][:top_k]]

    def add_coverage(self, lines: Set[str]) -> None:
        if self.coverage is None:
            self.coverage = set()
        self.coverage.update(lines)

    def coverage_count(self) -> int:
        return len(self.coverage) if self.coverage else 0

    def is_adversarial(self) -> bool:
        return self.status == "adversarial"

    def mark_adversarial(self) -> None:
        self.status = "adversarial"
