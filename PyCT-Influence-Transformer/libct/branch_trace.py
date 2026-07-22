from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from libct.predicate import Predicate


def _normalize_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_normalize_json(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in sorted(value.items())}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def canonical_position(position: Any) -> Any:
    return _normalize_json(position)


def branch_site_digest(expr: Any, position: Any, model_sha256: str) -> str:
    payload = {
        "model_sha256": str(model_sha256),
        "position": canonical_position(position),
        "formula": Predicate.get_formula_deep(expr),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BranchTraceEvent:
    site_digest: str
    observed_outcome: bool
    depth: int
    position: Any

    @property
    def transition_key(self) -> str:
        return f"{self.site_digest}:{int(self.observed_outcome)}"


__all__ = [
    "BranchTraceEvent",
    "branch_site_digest",
    "canonical_position",
]
