from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from libct.constraint import Constraint


@dataclass(frozen=True)
class ConstraintWorkItem:
    """A pending constraint plus metadata needed to preserve legacy queue behavior."""

    constraint: Constraint
    position: Optional[Any] = None
    shap_value: float = 0.0
    score: Optional[float] = None
    path_len: Optional[int] = None

    @classmethod
    def from_constraint(
        cls,
        constraint: Constraint,
        *,
        position: Optional[Any] = None,
        shap_value: float = 0.0,
        score: Optional[float] = None,
    ) -> "ConstraintWorkItem":
        return cls(
            constraint=constraint,
            position=position,
            shap_value=abs(float(shap_value)),
            score=score,
            path_len=getattr(constraint, "height", None),
        )
