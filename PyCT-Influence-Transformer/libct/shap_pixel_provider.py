from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


class JsonShapPixelProvider:
    """Load SHAP caches and expose sorted pixel coordinates per input."""

    def __init__(
        self,
        *,
        model_name: str,
        shap_root: str = "shap_value_all_layer",
        pixel_prefix: str = "-1",
        coordinate_dims: int | None = 3,
        fill_value: int = 0,
    ) -> None:
        self.model_name = model_name
        self.shap_root = Path(shap_root)
        self.pixel_prefix = pixel_prefix
        self.coordinate_dims = coordinate_dims
        self.fill_value = fill_value
        self._cache: Dict[int, List[Tuple[int, ...]]] = {}

        if coordinate_dims is not None and coordinate_dims <= 0:
            raise ValueError("coordinate_dims must be positive when provided.")

    def top_pixels(self, idx: int, ton: int | None = None) -> List[Tuple[int, ...]]:
        """Return the top-k pixel coordinates for a given input index."""
        coords = self._load_sorted(idx)
        if ton is None:
            return list(coords)
        if ton <= 0:
            raise ValueError("ton must be a positive integer.")
        return coords[: min(ton, len(coords))]

    def as_array(self, idx: int, ton: int | None = None) -> np.ndarray:
        """Expose sorted coordinates as a NumPy array (ton defaults to all pixels)."""
        coords = self.top_pixels(idx, ton)
        return np.asarray(coords, dtype=np.int64)

    def build_tensor(self, indices: Sequence[int], topk: int | None = None) -> np.ndarray:
        """Return a stacked tensor matching the historical *_sort_pixel_3d.npy layout."""
        if not indices:
            raise ValueError("indices must be non-empty to build a tensor.")
        per_row = []
        for idx in indices:
            per_row.append(self.top_pixels(idx, topk))

        limit = len(per_row[0])
        if topk is not None:
            limit = min(limit, topk)

        tensor = np.zeros((len(per_row), limit, self._coord_rank()), dtype=np.int64)
        for row_id, coords in enumerate(per_row):
            tensor[row_id] = np.asarray(coords[:limit], dtype=np.int64)
        return tensor

    def _coord_rank(self) -> int:
        if self.coordinate_dims is None:
            raise ValueError("coordinate_dims must be set to materialize arrays.")
        return self.coordinate_dims

    def _load_sorted(self, idx: int) -> List[Tuple[int, ...]]:
        if idx in self._cache:
            return self._cache[idx]

        path = self._resolve_json_path(idx)
        shap_values = self._load_json(path)
        coords = self._extract_sorted_coordinates(shap_values, path)
        self._cache[idx] = coords
        return coords

    def _resolve_json_path(self, idx: int) -> Path:
        path = self.shap_root / self.model_name / f"shap_value_{idx}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing SHAP cache: {path}")
        return path

    @staticmethod
    def _load_json(path: Path) -> Dict[str, float]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise TypeError(f"Expected SHAP cache {path} to be a JSON dict.")
        if "values" in data:
            data = data.get("values", {})
        if not isinstance(data, dict):
            raise TypeError(f"Expected SHAP cache {path} to contain a JSON dict of values.")
        return {str(k): float(v) for k, v in data.items()}

    def _extract_sorted_coordinates(
        self,
        shap_values: Dict[str, float],
        path: Path,
    ) -> List[Tuple[int, ...]]:
        prefix = f"{self.pixel_prefix}_"
        items: List[Tuple[Tuple[int, ...], float]] = []

        for key, value in shap_values.items():
            if not key.startswith(prefix):
                continue
            coords = self._normalize_coords(tuple(int(part) for part in key.split("_")[1:]))
            items.append((coords, abs(float(value))))

        if not items:
            raise ValueError(f"No pixel-level SHAP entries found in {path}")

        items.sort(key=lambda entry: entry[1], reverse=True)
        return [coords for coords, _ in items]

    def _normalize_coords(self, coords: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.coordinate_dims is None:
            return coords
        if len(coords) > self.coordinate_dims:
            raise ValueError(
                f"Coordinate {coords} exceeds configured dimensionality {self.coordinate_dims}"
            )
        if len(coords) == self.coordinate_dims:
            return coords
        padding = (self.fill_value,) * (self.coordinate_dims - len(coords))
        return coords + padding


def build_shap_tensor_from_json(
    indices: Iterable[int],
    *,
    model_name: str,
    shap_root: str = "shap_value_all_layer",
    pixel_prefix: str = "-1",
    coordinate_dims: int = 3,
) -> np.ndarray:
    """Helper for scripts/tests to rebuild the legacy *_sort_pixel_3d.npy tensor."""
    provider = JsonShapPixelProvider(
        model_name=model_name,
        shap_root=shap_root,
        pixel_prefix=pixel_prefix,
        coordinate_dims=coordinate_dims,
    )
    return provider.build_tensor(list(indices))
