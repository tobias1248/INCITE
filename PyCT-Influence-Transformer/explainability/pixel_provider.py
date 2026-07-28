from __future__ import annotations

import json
from dataclasses import dataclass
from math import isqrt
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from explainability.shap_contract import (
    DEFAULT_TARGET_CLASS_SHAP_ROOT,
    load_target_class_cache,
)

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


PixelCoord = Tuple[int, ...]
RankedPixel = Tuple[PixelCoord, float]


@dataclass(frozen=True)
class TokenizerSpec:
    kind: str
    input_hw: Tuple[int, int]
    token_count_before: int
    token_count_after: int | None = None
    patch_size: int | None = None
    pool_size: int | None = None
    stride: int | None = None


_TOKENIZER_KIND_OVERRIDES = {
    "cifar10_cctlike_eight_mha": "sequence_pool_1d",
}


def _resolve_model_path(model_name: str, *, model_root: str = "model") -> Path:
    path = Path(model_name)
    if path.suffix == ".h5":
        return path
    return Path(model_root) / f"{model_name}.h5"


def _load_model_config(model_path: Path) -> Dict[str, object]:
    if h5py is None:
        raise RuntimeError("h5py is required to infer tokenizer structure from model metadata.")
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


def _extract_layers(config: Dict[str, object], model_path: Path) -> List[Dict[str, object]]:
    layers = config.get("config", {}).get("layers", [])
    if not isinstance(layers, list):
        raise ValueError(f"Model '{model_path}' has an unsupported layer config structure.")
    return [layer for layer in layers if isinstance(layer, dict)]


def _coerce_int_list(value: object, *, expected_len: int) -> Tuple[int, ...] | None:
    if not isinstance(value, list) or len(value) != expected_len:
        return None
    try:
        return tuple(int(part) for part in value)
    except (TypeError, ValueError):
        return None


def _extract_input_hw(layers: Sequence[Dict[str, object]]) -> Tuple[int, int] | None:
    for layer in layers:
        if layer.get("class_name") != "InputLayer":
            continue
        batch_shape = layer.get("config", {}).get("batch_input_shape")
        if not isinstance(batch_shape, list) or len(batch_shape) < 3:
            continue
        try:
            return int(batch_shape[1]), int(batch_shape[2])
        except (TypeError, ValueError):
            return None
    return None


def _infer_patch_2d_tokenizer_spec(
    layers: Sequence[Dict[str, object]],
    *,
    input_hw: Tuple[int, int] | None,
    model_path: Path,
) -> TokenizerSpec | None:
    if input_hw is None:
        return None

    for idx, layer in enumerate(layers):
        if layer.get("class_name") != "Conv2D":
            continue
        lcfg = layer.get("config", {})
        kernel_size = _coerce_int_list(lcfg.get("kernel_size"), expected_len=2)
        strides = _coerce_int_list(lcfg.get("strides"), expected_len=2)
        if kernel_size is None or strides is None:
            continue
        if kernel_size[0] != kernel_size[1] or strides[0] != strides[1]:
            continue
        if kernel_size[0] != strides[0]:
            continue

        patch_size = int(kernel_size[0])
        next_layer = layers[idx + 1] if idx + 1 < len(layers) else None
        if not isinstance(next_layer, dict) or next_layer.get("class_name") != "Reshape":
            continue
        target_shape = _coerce_int_list(next_layer.get("config", {}).get("target_shape"), expected_len=2)
        if target_shape is None:
            continue

        token_count = int(target_shape[0])
        patch_side = isqrt(token_count)
        if patch_side * patch_side != token_count:
            raise ValueError(
                f"Model '{model_path}' has non-square token count {token_count}; unable to infer patch-based tokenizer."
            )
        if input_hw[0] % patch_side != 0 or input_hw[1] % patch_side != 0:
            raise ValueError(
                f"Model '{model_path}' input shape {input_hw} is incompatible with token count {token_count}."
            )
        expected_h = input_hw[0] // patch_side
        expected_w = input_hw[1] // patch_side
        if expected_h != patch_size or expected_w != patch_size:
            raise ValueError(
                f"Model '{model_path}' patch embedding and token reshape disagree on patch size."
            )
        return TokenizerSpec(
            kind="patch_2d",
            input_hw=input_hw,
            token_count_before=token_count,
            token_count_after=token_count,
            patch_size=patch_size,
        )
    return None


def _infer_sequence_pool_tokenizer_spec(
    layers: Sequence[Dict[str, object]],
    *,
    input_hw: Tuple[int, int] | None,
) -> TokenizerSpec | None:
    if input_hw is None:
        return None

    for idx, layer in enumerate(layers):
        if layer.get("class_name") != "Reshape":
            continue
        target_shape = _coerce_int_list(layer.get("config", {}).get("target_shape"), expected_len=2)
        if target_shape is None:
            continue
        token_count_before, feature_dim = target_shape
        expected_tokens = input_hw[0] * input_hw[1]
        if token_count_before != expected_tokens or feature_dim <= 0:
            continue

        next_layer = layers[idx + 1] if idx + 1 < len(layers) else None
        if isinstance(next_layer, dict) and next_layer.get("class_name") == "AveragePooling1D":
            pcfg = next_layer.get("config", {})
            pool_size_vals = _coerce_int_list(pcfg.get("pool_size"), expected_len=1)
            stride_vals = _coerce_int_list(pcfg.get("strides"), expected_len=1)
            if pool_size_vals is None or stride_vals is None:
                continue

            pool_size = int(pool_size_vals[0])
            stride = int(stride_vals[0])
            if pool_size <= 0 or stride <= 0:
                continue
            token_count_after = ((token_count_before - pool_size) // stride) + 1
            if token_count_after <= 0:
                continue
        else:
            pool_size = 1
            stride = 1
            token_count_after = token_count_before
        return TokenizerSpec(
            kind="sequence_pool_1d",
            input_hw=input_hw,
            token_count_before=token_count_before,
            token_count_after=token_count_after,
            pool_size=pool_size,
            stride=stride,
        )
    return None


def infer_tokenizer_spec_for_model(model_name: str, *, model_root: str = "model") -> TokenizerSpec:
    model_path = _resolve_model_path(model_name, model_root=model_root)
    config = _load_model_config(model_path)
    layers = _extract_layers(config, model_path)
    input_hw = _extract_input_hw(layers)

    patch_spec = _infer_patch_2d_tokenizer_spec(layers, input_hw=input_hw, model_path=model_path)
    seq_spec = _infer_sequence_pool_tokenizer_spec(layers, input_hw=input_hw)
    candidates = {
        "patch_2d": patch_spec,
        "sequence_pool_1d": seq_spec,
    }

    override_kind = _TOKENIZER_KIND_OVERRIDES.get(model_path.stem)
    if override_kind is not None:
        override_spec = candidates.get(override_kind)
        if override_spec is None:
            raise ValueError(
                f"Model '{model_path}' is configured as {override_kind}, but the tokenizer heuristic could not confirm that structure."
            )
        return override_spec

    if patch_spec is not None and seq_spec is not None:
        raise ValueError(f"Model '{model_path}' matched multiple tokenizer heuristics; manual disambiguation is required.")
    if patch_spec is not None:
        return patch_spec
    if seq_spec is not None:
        return seq_spec
    raise ValueError(
        f"Unsupported model tokenizer for '{model_path}'; expected either Conv2D patch embedding or Reshape + AveragePooling1D."
    )


def infer_patch_size_for_model(model_name: str, *, model_root: str = "model") -> int:
    model_path = _resolve_model_path(model_name, model_root=model_root)
    try:
        spec = infer_tokenizer_spec_for_model(model_name, model_root=model_root)
    except ValueError as exc:
        raise ValueError(
            f"Unable to infer patch size for model '{model_path}'; expected a patch-embedding Conv2D followed by flatten_patches Reshape."
        ) from exc
    if spec.kind != "patch_2d" or spec.patch_size is None:
        raise ValueError(
            f"Unable to infer patch size for model '{model_path}'; expected a patch-embedding Conv2D followed by flatten_patches Reshape."
        )
    return spec.patch_size


class JsonShapPixelProvider:
    """Load SHAP caches and expose sorted pixel coordinates per input."""

    def __init__(
        self,
        *,
        model_name: str,
        shap_root: str = DEFAULT_TARGET_CLASS_SHAP_ROOT,
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
        self.tokenizer_spec: TokenizerSpec | None = None
        self._cache: Dict[int, List[Tuple[int, ...]]] = {}

        if self.selector not in {"pixel-shap", "patch-shap", "token-shap"}:
            raise ValueError(f"Unsupported selector: {self.selector}")
        if coordinate_dims is not None and coordinate_dims <= 0:
            raise ValueError("coordinate_dims must be positive when provided.")
        if self.coordinate_bounds is not None and any(v <= 0 for v in self.coordinate_bounds):
            raise ValueError("coordinate_bounds must contain positive integers.")
        if self.selector in {"patch-shap", "token-shap"}:
            self.tokenizer_spec = infer_tokenizer_spec_for_model(self.model_name, model_root=self.model_root)
            self._validate_selector_compatibility()

    def top_pixels(self, idx: int, ton: int | None = None) -> List[Tuple[int, ...]]:
        """Return the top-k pixel coordinates for a given input index."""
        coords = self._load_sorted(idx)
        if self.selector in {"patch-shap", "token-shap"}:
            if ton is None:
                return list(coords)
            if ton != 1:
                raise ValueError(f"{self.selector} supports only ton=1 in v1.")
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

    def _validate_selector_compatibility(self) -> None:
        if self.selector == "patch-shap":
            if self.tokenizer_spec is None or self.tokenizer_spec.kind != "patch_2d":
                raise ValueError(
                    f"patch-shap only supports patch-embedding tokenizers; model '{self.model_name}' uses {self._describe_tokenizer_kind()}."
                )
            self.patch_size = self.tokenizer_spec.patch_size
            if self.patch_size is None or self.patch_size <= 0:
                raise ValueError("patch_size must be positive when using patch-shap.")
            return

        if self.selector == "token-shap":
            if self.tokenizer_spec is None or self.tokenizer_spec.kind != "sequence_pool_1d":
                raise ValueError(
                    f"token-shap only supports sequence tokenizers; model '{self.model_name}' uses {self._describe_tokenizer_kind()}."
                )
            if self.tokenizer_spec.pool_size != self.tokenizer_spec.stride:
                raise ValueError(
                    f"token-shap supports only non-overlapping sequence pooling in v1; got pool_size={self.tokenizer_spec.pool_size}, stride={self.tokenizer_spec.stride}."
                )

    def _describe_tokenizer_kind(self) -> str:
        if self.tokenizer_spec is None:
            return "an unknown tokenizer"
        if self.tokenizer_spec.kind == "patch_2d":
            return "a patch-embedding tokenizer"
        if self.tokenizer_spec.kind == "sequence_pool_1d":
            return "a sequence tokenizer"
        return self.tokenizer_spec.kind

    def _load_sorted(self, idx: int) -> List[Tuple[int, ...]]:
        if idx in self._cache:
            return self._cache[idx]

        path = self._resolve_json_path(idx)
        shap_values = self._load_json(
            path,
            case_index=idx,
            expected_model_name=self.model_name,
        )
        ranked = self._extract_ranked_pixel_items(shap_values, path)
        if self.selector == "patch-shap":
            coords = self._select_patch_shap_coordinates(ranked, path)
        elif self.selector == "token-shap":
            coords = self._select_token_shap_coordinates(ranked, path)
        else:
            coords = [coords for coords, _ in ranked]
        self._cache[idx] = coords
        return coords

    def _resolve_json_path(self, idx: int) -> Path:
        path = self.shap_root / self.model_name / f"shap_value_{idx}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing target-class SHAP cache: {path}. Generate it with "
                "python -m pyct.shap --force-refresh"
            )
        return path

    @staticmethod
    def _load_json(
        path: Path,
        *,
        case_index: int | None = None,
        expected_model_name: str | None = None,
    ) -> Dict[str, float]:
        metadata, values = load_target_class_cache(path, case_index=case_index)
        if expected_model_name is not None:
            cached_model_name = Path(metadata["model"]["name"]).stem
            if cached_model_name != Path(expected_model_name).stem:
                raise ValueError(
                    f"SHAP cache model={cached_model_name!r}, expected "
                    f"{Path(expected_model_name).stem!r}"
                )
        return values

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

    def _select_token_shap_coordinates(self, ranked: List[RankedPixel], path: Path) -> List[PixelCoord]:
        if self.tokenizer_spec is None or self.tokenizer_spec.kind != "sequence_pool_1d":
            raise ValueError("tokenizer_spec is required for token-shap selection.")
        width = self.tokenizer_spec.input_hw[1]
        stride = self.tokenizer_spec.stride
        if stride is None or stride <= 0:
            raise ValueError("stride is required for token-shap selection.")

        group_scores: Dict[int, float] = {}
        group_items: Dict[int, List[RankedPixel]] = {}

        for coords, score in ranked:
            if len(coords) < 2:
                raise ValueError(f"Token-aware selection requires at least 2 spatial dimensions in {path}.")
            token_idx = coords[0] * width + coords[1]
            if token_idx < 0 or token_idx >= self.tokenizer_spec.token_count_before:
                raise ValueError(f"Coordinate {coords} maps outside the tokenizer domain for {path}.")
            group_id = token_idx // stride
            if self.tokenizer_spec.token_count_after is not None and group_id >= self.tokenizer_spec.token_count_after:
                raise ValueError(f"Coordinate {coords} maps outside pooled token groups for {path}.")
            group_scores[group_id] = group_scores.get(group_id, 0.0) + score
            group_items.setdefault(group_id, []).append((coords, score))

        if not group_scores:
            raise ValueError(f"No token-aware SHAP entries found in {path}")

        best_group = sorted(group_scores.items(), key=lambda entry: (-entry[1], entry[0]))[0][0]
        best_coord = sorted(group_items[best_group], key=lambda entry: (-entry[1], entry[0]))[0][0]
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
    shap_root: str = DEFAULT_TARGET_CLASS_SHAP_ROOT,
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
