from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


PixelCoord = Tuple[int, ...]


class DeScoutPixelProvider:
    """Load DE-selected pixel coordinates from an offline scout artifact."""

    def __init__(
        self,
        *,
        path: str,
        dataset: str,
        model_name: str,
        coordinate_bounds: Sequence[int],
    ) -> None:
        self.path = Path(path)
        self.dataset = dataset
        self.model_name = model_name
        self.coordinate_bounds = tuple(int(bound) for bound in coordinate_bounds)
        if not self.coordinate_bounds:
            raise ValueError("coordinate_bounds must not be empty.")
        if any(bound <= 0 for bound in self.coordinate_bounds):
            raise ValueError(f"coordinate_bounds must be positive, got {self.coordinate_bounds}.")

        payload = self._load_payload(self.path)
        self._validate_payload(payload)
        self._candidates = payload["candidates"]

    @staticmethod
    def _load_payload(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"DE scout artifact '{path}' must contain a JSON object.")
        return payload

    def _validate_payload(self, payload: Dict[str, Any]) -> None:
        schema_version = payload.get("schema_version")
        if schema_version != 1:
            raise ValueError(f"Unsupported DE scout schema_version {schema_version!r}; expected 1.")
        if payload.get("dataset") != self.dataset:
            raise ValueError(
                f"DE scout dataset mismatch: expected {self.dataset!r}, got {payload.get('dataset')!r}."
            )
        if payload.get("model_name") != self.model_name:
            raise ValueError(
                "DE scout model_name mismatch: "
                f"expected {self.model_name!r}, got {payload.get('model_name')!r}."
            )
        candidates = payload.get("candidates")
        if not isinstance(candidates, dict):
            raise ValueError("DE scout artifact must contain an object-valued 'candidates' field.")

    def top_pixels(self, idx: int, ton: int) -> List[PixelCoord]:
        if ton < 1:
            raise ValueError("ton must be >= 1.")

        case_key = str(int(idx))
        raw_candidates = self._candidates.get(case_key)
        if not isinstance(raw_candidates, list):
            raise ValueError(f"DE scout artifact has no candidate list for case {case_key}.")

        coords: List[PixelCoord] = []
        seen = set()
        for candidate in self._ranked_candidates(raw_candidates):
            coord = self._normalize_coord(candidate, case_key)
            if coord in seen:
                continue
            seen.add(coord)
            coords.append(coord)

        if len(coords) < ton:
            raise ValueError(
                f"DE scout artifact has only {len(coords)} unique candidates for case {case_key}; "
                f"requested {ton}."
            )
        return coords[:ton]

    @staticmethod
    def _ranked_candidates(raw_candidates: List[Any]) -> List[Any]:
        def sort_key(item: Tuple[int, Any]) -> Tuple[int, int]:
            index, candidate = item
            if isinstance(candidate, dict):
                rank = candidate.get("rank")
                if isinstance(rank, int):
                    return rank, index
            return index, index

        return [candidate for _, candidate in sorted(enumerate(raw_candidates), key=sort_key)]

    def _normalize_coord(self, candidate: Any, case_key: str) -> PixelCoord:
        if not isinstance(candidate, dict):
            raise ValueError(f"DE scout candidate for case {case_key} must be an object.")
        raw_coord = candidate.get("coord")
        if not isinstance(raw_coord, list):
            raise ValueError(f"DE scout candidate for case {case_key} is missing list field 'coord'.")
        if len(raw_coord) != len(self.coordinate_bounds):
            raise ValueError(
                f"DE scout coord rank mismatch for case {case_key}: "
                f"expected {len(self.coordinate_bounds)}, got {len(raw_coord)}."
            )

        coord_parts: List[int] = []
        for axis, (raw_value, upper_bound) in enumerate(zip(raw_coord, self.coordinate_bounds)):
            if not isinstance(raw_value, int):
                raise ValueError(
                    f"DE scout coord axis {axis} for case {case_key} must be an integer, "
                    f"got {raw_value!r}."
                )
            if raw_value < 0 or raw_value >= upper_bound:
                raise ValueError(
                    f"DE scout coord axis {axis} for case {case_key} is out of bounds: "
                    f"{raw_value} not in [0, {upper_bound})."
                )
            coord_parts.append(raw_value)
        return tuple(coord_parts)
