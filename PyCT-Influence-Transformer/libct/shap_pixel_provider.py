from __future__ import annotations

import json
from math import isqrt
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


PixelCoord = Tuple[int, ...]
RankedPixel = Tuple[PixelCoord, float]


def _resolve_model_path(model_name: str, *, model_root: str = "model") -> Path:
    path = Path(model_name)
    if path.suffix == ".h5":
        return path
    return Path(model_root) / f"{model_name}.h5"


def _load_model_config(model_path: Path) -> Dict[str, object]:
    if h5py is None:
        raise RuntimeError("h5py is required to infer patch size from model structure.")
    try:
        with h5py.File(model_path, "r") as handle:
            raw = handle.attrs.get("model_config")
            if raw is None:
                raise ValueError(f"Model '{model_path}' does not contain a model_config attribute.")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            config = json.loads(raw)
    except OSError as exc:
        raise FileNotFoundError(f"Unable to read model file '{model_path}'.") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Model config for '{model_path}' is not a JSON object.")
    return config


def infer_patch_size_for_model(model_name: str, *, model_root: str = "model") -> int:
    model_path = _resolve_model_path(model_name, model_root=model_root)
    config = _load_model_config(model_path)
    layers = config.get("config", {}).get("layers", [])
    if not isinstance(layers, list):
        raise ValueError(f"Model '{model_path}' has an unsupported layer config structure.")

    input_hw: Tuple[int, int] | None = None
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        if layer.get("class_name") != "InputLayer":
            continue
        batch_shape = layer.get("config", {}).get("batch_input_shape")
        if isinstance(batch_shape, list) and len(batch_shape) >= 3:
            try:
                input_hw = (int(batch_shape[1]), int(batch_shape[2]))
            except (TypeError, ValueError):
                input_hw = None
            break

    for idx, layer in enumerate(layers):
        if not isinstance(layer, dict) or layer.get("class_name") != "Conv2D":
            continue
        lcfg = layer.get("config", {})
        kernel_size = lcfg.get("kernel_size")
        strides = lcfg.get("strides")
        if not (
            isinstance(kernel_size, list)
            and len(kernel_size) == 2
            and isinstance(strides, list)
            and len(strides) == 2
        ):
            continue
        if kernel_size[0] != kernel_size[1] or strides[0] != strides[1]:
            continue
        if kernel_size[0] != strides[0]:
            continue

        patch_size = int(kernel_size[0])
        next_layer = layers[idx + 1] if idx + 1 < len(layers) else None
        if not isinstance(next_layer, dict) or next_layer.get("class_name") != "Reshape":
            continue
        target_shape = next_layer.get("config", {}).get("target_shape")
        if not (isinstance(target_shape, list) and len(target_shape) == 2):
            continue

        try:
            token_count = int(target_shape[0])
        except (TypeError, ValueError):
            continue

        if input_hw is not None:
            patch_side = isqrt(token_count)
            if patch_side * patch_side != token_count:
                raise ValueError(
                    f"Model '{model_path}' has non-square token count {token_count}; unable to infer patch size."
                )
            expected_h = input_hw[0] // patch_side
            expected_w = input_hw[1] // patch_side
            if input_hw[0] % patch_side != 0 or input_hw[1] % patch_side != 0:
                raise ValueError(
                    f"Model '{model_path}' input shape {input_hw} is incompatible with token count {token_count}."
                )
            if expected_h != patch_size or expected_w != patch_size:
                raise ValueError(
                    f"Model '{model_path}' patch embedding and token reshape disagree on patch size."
                )
        return patch_size

    raise ValueError(
        f"Unable to infer patch size for model '{model_path}'; expected a patch-embedding Conv2D followed by flatten_patches Reshape."
    )


class JsonShapPixelProvider:
    """Load SHAP caches and expose sorted pixel coordinates per input."""

    def __init__(
        self,
        *,
        model_name: str,
        shap_root: str = "shap_value_all_layer",
        pixel_prefix: str = "-1",
        selector: str = "pixel-shap",
        coordinate_dims: int | None = 3,
        coordinate_bounds: Tuple[int, ...] | None = None,
        fill_value: int = 0,
        model_root: str = "model",
        patch_size: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.shap_root = Path(shap_root)
        self.pixel_prefix = pixel_prefix
        self.selector = selector
        self.coordinate_dims = coordinate_dims
        self.coordinate_bounds = tuple(int(v) for v in coordinate_bounds) if coordinate_bounds else None
        self.fill_value = fill_value
        self.model_root = model_root
        self.patch_size = patch_size
        self._cache: Dict[int, List[Tuple[int, ...]]] = {}

        if self.selector not in {"pixel-shap", "patch-shap"}:
            raise ValueError(f"Unsupported selector: {self.selector}")
        if coordinate_dims is not None and coordinate_dims <= 0:
            raise ValueError("coordinate_dims must be positive when provided.")
        if self.coordinate_bounds is not None and any(v <= 0 for v in self.coordinate_bounds):
            raise ValueError("coordinate_bounds must contain positive integers.")
        if self.selector == "patch-shap":
            if self.patch_size is None:
                self.patch_size = infer_patch_size_for_model(self.model_name, model_root=self.model_root)
            if self.patch_size <= 0:
                raise ValueError("patch_size must be positive when using patch-shap.")

    def top_pixels(self, idx: int, ton: int | None = None) -> List[Tuple[int, ...]]:
        """Return the top-k pixel coordinates for a given input index."""
        coords = self._load_sorted(idx)
        if self.selector == "patch-shap":
            if ton is None:
                return list(coords)
            if ton != 1:
                raise ValueError("patch-shap supports only ton=1 in v1.")
            return list(coords)
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
        ranked = self._extract_ranked_pixel_items(shap_values, path)
        if self.selector == "patch-shap":
            coords = self._select_patch_shap_coordinates(ranked, path)
        else:
            coords = [coords for coords, _ in ranked]
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

    def _extract_ranked_pixel_items(
        self,
        shap_values: Dict[str, float],
        path: Path,
    ) -> List[RankedPixel]:
        prefix = f"{self.pixel_prefix}_"
        items: List[RankedPixel] = []

        for key, value in shap_values.items():
            if not key.startswith(prefix):
                continue
            coords = self._normalize_coords(tuple(int(part) for part in key.split("_")[1:]))
            if coords is None:
                continue
            items.append((coords, abs(float(value))))

        if not items:
            raise ValueError(f"No pixel-level SHAP entries found in {path}")

        items.sort(key=lambda entry: (-entry[1], entry[0]))
        return items

    def _select_patch_shap_coordinates(self, ranked: List[RankedPixel], path: Path) -> List[PixelCoord]:
        if self.patch_size is None:
            raise ValueError("patch_size is required for patch-shap selection.")
        patch_scores: Dict[Tuple[int, int], float] = {}
        patch_items: Dict[Tuple[int, int], List[RankedPixel]] = {}

        for coords, score in ranked:
            if len(coords) < 2:
                raise ValueError(f"Patch-aware selection requires at least 2 spatial dimensions in {path}.")
            patch_id = (coords[0] // self.patch_size, coords[1] // self.patch_size)
            patch_scores[patch_id] = patch_scores.get(patch_id, 0.0) + score
            patch_items.setdefault(patch_id, []).append((coords, score))

        if not patch_scores:
            raise ValueError(f"No patch-aware SHAP entries found in {path}")

        best_patch = sorted(patch_scores.items(), key=lambda entry: (-entry[1], entry[0]))[0][0]
        best_coord = sorted(patch_items[best_patch], key=lambda entry: (-entry[1], entry[0]))[0][0]
        return [best_coord]

    def _normalize_coords(self, coords: Tuple[int, ...]) -> Tuple[int, ...] | None:
        if self.coordinate_dims is None:
            normalized = coords
        else:
            if len(coords) > self.coordinate_dims:
                return None
            if len(coords) == self.coordinate_dims:
                normalized = coords
            else:
                padding = (self.fill_value,) * (self.coordinate_dims - len(coords))
                normalized = coords + padding

        if self.coordinate_bounds is None:
            return normalized
        if len(normalized) != len(self.coordinate_bounds):
            return None
        if any(axis < 0 or axis >= bound for axis, bound in zip(normalized, self.coordinate_bounds)):
            return None
        return normalized


def build_shap_tensor_from_json(
    indices: Iterable[int],
    *,
    model_name: str,
    shap_root: str = "shap_value_all_layer",
    pixel_prefix: str = "-1",
    coordinate_dims: int = 3,
    coordinate_bounds: Tuple[int, ...] | None = None,
) -> np.ndarray:
    """Helper for scripts/tests to rebuild the legacy *_sort_pixel_3d.npy tensor."""
    provider = JsonShapPixelProvider(
        model_name=model_name,
        shap_root=shap_root,
        pixel_prefix=pixel_prefix,
        coordinate_dims=coordinate_dims,
        coordinate_bounds=coordinate_bounds,
    )
    return provider.build_tensor(list(indices))
