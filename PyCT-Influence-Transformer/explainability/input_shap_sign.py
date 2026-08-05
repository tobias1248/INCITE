from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from explainability.shap_contract import (
    DEFAULT_TARGET_CLASS_SHAP_ROOT,
    ShapCacheContractError,
    build_cache_identity,
    load_target_class_cache,
)


BOUNDS_MODE_CLIP = "clip"
BOUNDS_MODE_STRICT = "strict"
BOUNDS_MODES = (BOUNDS_MODE_CLIP, BOUNDS_MODE_STRICT)


@dataclass(frozen=True)
class TargetClassInputShap:
    values: np.ndarray
    target_class: int
    cache_path: Path
    was_cached: bool
    metadata: Mapping[str, Any]


def build_sign_mask(values: np.ndarray, *, epsilon: float = 0.0) -> np.ndarray:
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"epsilon must be finite and non-negative, got {epsilon}")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("SHAP values contain NaN or Inf")
    return np.where(array > epsilon, 1, np.where(array < -epsilon, -1, 0)).astype(
        np.int8
    )


def _validated_shift_inputs(
    sample: np.ndarray,
    sign_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    image = np.asarray(sample, dtype=np.float64)
    signs = np.asarray(sign_mask, dtype=np.int8)
    if image.shape != signs.shape:
        raise ValueError(f"sample shape {image.shape} does not match sign mask {signs.shape}")
    if not np.isfinite(image).all():
        raise ValueError("sample contains NaN or Inf")
    if np.any(image < 0.0) or np.any(image > 1.0):
        raise ValueError("sample must be inside [0, 1]")
    if not np.isin(signs, (-1, 0, 1)).all():
        raise ValueError("sign mask must contain only -1, 0, and 1")
    return image, signs


def derive_valid_shift_interval(
    sample: np.ndarray,
    sign_mask: np.ndarray,
    *,
    requested_min: float = -0.1,
    requested_max: float = 0.1,
) -> Tuple[float, float]:
    """Return the shared-X interval that needs no element-wise clipping."""
    if not math.isfinite(requested_min) or not math.isfinite(requested_max):
        raise ValueError("requested shift bounds must be finite")
    if requested_min > requested_max:
        raise ValueError("requested_min must be <= requested_max")

    image, signs = _validated_shift_inputs(sample, sign_mask)
    lower = float(requested_min)
    upper = float(requested_max)
    positive = signs == 1
    negative = signs == -1
    if np.any(positive):
        lower = max(lower, float(np.max(-image[positive])))
        upper = min(upper, float(np.min(1.0 - image[positive])))
    if np.any(negative):
        lower = max(lower, float(np.max(image[negative] - 1.0)))
        upper = min(upper, float(np.min(image[negative])))
    if lower > upper + 1e-12:
        raise ValueError(f"valid shift interval is empty: [{lower}, {upper}]")
    return max(lower, requested_min), min(upper, requested_max)


def count_clipped_values(
    sample: np.ndarray,
    sign_mask: np.ndarray,
    shift: float,
) -> int:
    if not math.isfinite(shift):
        raise ValueError(f"shift must be finite, got {shift}")
    image, signs = _validated_shift_inputs(sample, sign_mask)
    shifted = image + signs * float(shift)
    return int(np.count_nonzero((shifted < 0.0) | (shifted > 1.0)))


def materialize_shifted_input(
    sample: np.ndarray,
    sign_mask: np.ndarray,
    shift: float,
    *,
    bounds_mode: str = BOUNDS_MODE_STRICT,
) -> np.ndarray:
    if bounds_mode not in BOUNDS_MODES:
        raise ValueError(
            f"bounds_mode must be one of {', '.join(BOUNDS_MODES)}, got {bounds_mode!r}"
        )
    if not math.isfinite(shift):
        raise ValueError(f"shift must be finite, got {shift}")
    image, signs = _validated_shift_inputs(sample, sign_mask)
    shifted = image + signs * float(shift)
    if bounds_mode == BOUNDS_MODE_STRICT and (
        np.any(shifted < -1e-7) or np.any(shifted > 1.0 + 1e-7)
    ):
        raise ValueError("materialized input is outside [0, 1] in strict mode")
    return np.clip(shifted, 0.0, 1.0).astype(np.float32)


def _mapping_to_input_values(
    values: Mapping[str, Any],
    shape: Sequence[int],
) -> np.ndarray:
    normalized_shape = tuple(int(dim) for dim in shape)
    if not normalized_shape or any(dim < 1 for dim in normalized_shape):
        raise ValueError(f"input shape must contain positive dimensions, got {shape!r}")

    result = np.empty(normalized_shape, dtype=np.float64)
    seen = set()
    for key, value in values.items():
        if not isinstance(key, str) or not key.startswith("-1_"):
            continue
        try:
            index = tuple(int(part) for part in key.split("_")[1:])
        except ValueError as exc:
            raise ShapCacheContractError(f"invalid input SHAP key: {key!r}") from exc
        if len(index) != len(normalized_shape) or any(
            axis < 0 or axis >= normalized_shape[dim]
            for dim, axis in enumerate(index)
        ):
            raise ShapCacheContractError(
                f"input SHAP coordinate is out of bounds: {key!r}"
            )
        try:
            result[index] = float(value)
        except (TypeError, ValueError) as exc:
            raise ShapCacheContractError(
                f"input SHAP value for {key!r} is not numeric"
            ) from exc
        seen.add(index)

    expected_count = int(np.prod(normalized_shape))
    if len(seen) != expected_count:
        raise ShapCacheContractError(
            f"input SHAP cache has {len(seen)} coordinates; expected {expected_count}"
        )
    if not np.isfinite(result).all():
        raise ShapCacheContractError("input SHAP cache contains NaN or Inf")
    return result


class TargetClassInputShapProvider:
    """Load or generate canonical target-class SHAP and expose input values."""

    def __init__(
        self,
        *,
        model_path: Path,
        output_root: Path = Path(DEFAULT_TARGET_CLASS_SHAP_ROOT),
        explainer_type: str = "gradient",
        calculator_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model file is unavailable: {self.model_path}")
        if explainer_type not in {"gradient", "kernel"}:
            raise ValueError(f"unsupported SHAP explainer type: {explainer_type}")
        self.output_dir = Path(output_root) / self.model_path.stem
        self.explainer_type = explainer_type
        self.calculator_factory = calculator_factory

    def cache_path(self, case_index: int) -> Path:
        return self.output_dir / f"shap_value_{int(case_index)}.json"

    def _identity(
        self,
        *,
        case_index: int,
        sample: np.ndarray,
        background: np.ndarray,
    ) -> Dict[str, Any]:
        return build_cache_identity(
            case_index=case_index,
            model_path=self.model_path,
            input_data=np.expand_dims(np.asarray(sample), axis=0),
            background_dataset=np.asarray(background),
            explainer_type=self.explainer_type,
        )

    def _load(
        self,
        *,
        path: Path,
        identity: Mapping[str, Any],
        case_index: int,
        target_class: int,
        sample_shape: Sequence[int],
        was_cached: bool,
    ) -> TargetClassInputShap:
        metadata, values = load_target_class_cache(
            path,
            expected_identity=identity,
            case_index=case_index,
        )
        if metadata["target_class"] != int(target_class):
            raise ShapCacheContractError(
                f"SHAP cache target_class={metadata['target_class']}, expected {target_class}"
            )
        return TargetClassInputShap(
            values=_mapping_to_input_values(values, sample_shape),
            target_class=int(target_class),
            cache_path=path,
            was_cached=bool(was_cached),
            metadata=metadata,
        )

    def ensure(
        self,
        *,
        case_index: int,
        sample: np.ndarray,
        background: np.ndarray,
        target_class: int,
        force_refresh: bool = False,
    ) -> TargetClassInputShap:
        sample_array = np.asarray(sample)
        background_array = np.asarray(background)
        identity = self._identity(
            case_index=case_index,
            sample=sample_array,
            background=background_array,
        )
        path = self.cache_path(case_index)
        cache_requires_refresh = False
        if path.is_file() and not force_refresh:
            try:
                return self._load(
                    path=path,
                    identity=identity,
                    case_index=case_index,
                    target_class=target_class,
                    sample_shape=sample_array.shape,
                    was_cached=True,
                )
            except ShapCacheContractError:
                cache_requires_refresh = True

        calculator_factory = self.calculator_factory
        if calculator_factory is None:
            from explainability.shap_calculator import ShapValuesCalculator

            calculator_factory = ShapValuesCalculator
        calculator = calculator_factory(
            model_path=str(self.model_path),
            background_dataset=background_array,
            input_data=np.expand_dims(sample_array, axis=0),
            idx=case_index,
            explainer_type=self.explainer_type,
            output_root=str(self.output_dir.parent),
        )
        calculator.ensure(
            assume_cached=path.is_file() and not (
                force_refresh or cache_requires_refresh
            ),
            force_refresh=force_refresh or cache_requires_refresh,
        )
        if int(calculator.target_class) != int(target_class):
            raise ValueError(
                f"SHAP target class {calculator.target_class} does not match "
                f"original prediction {target_class}"
            )
        was_cached = bool(calculator.last_timing.get("was_cached", False))
        return self._load(
            path=path,
            identity=identity,
            case_index=case_index,
            target_class=target_class,
            sample_shape=sample_array.shape,
            was_cached=was_cached,
        )


__all__ = [
    "BOUNDS_MODE_CLIP",
    "BOUNDS_MODE_STRICT",
    "BOUNDS_MODES",
    "TargetClassInputShap",
    "TargetClassInputShapProvider",
    "build_sign_mask",
    "count_clipped_values",
    "derive_valid_shift_interval",
    "materialize_shifted_input",
]
