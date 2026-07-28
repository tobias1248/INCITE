from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np


DEFAULT_TARGET_CLASS_SHAP_ROOT = "shap_target_class"
TARGET_CLASS_SHAP_SCHEMA_VERSION = 2
TARGET_CLASS_ATTRIBUTION = "original_prediction"


class ShapCacheContractError(ValueError):
    """Raised when a SHAP artifact does not satisfy the target-class contract."""


def _sha256_bytes(chunks: Tuple[bytes, ...]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def fingerprint_array(value: np.ndarray) -> Dict[str, Any]:
    import numpy as np

    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    shape = tuple(int(dim) for dim in contiguous.shape)
    return {
        "shape": list(shape),
        "dtype": contiguous.dtype.str,
        "sha256": _sha256_bytes(
            (
                contiguous.dtype.str.encode("utf-8"),
                repr(shape).encode("utf-8"),
                contiguous.tobytes(),
            )
        ),
    }


def fingerprint_file(path: str | Path) -> Dict[str, Any]:
    model_path = Path(path).expanduser().resolve()
    try:
        stat = model_path.stat()
    except OSError as exc:
        raise ShapCacheContractError(
            f"Unable to fingerprint SHAP model '{model_path}': {exc}"
        ) from exc

    digest = hashlib.sha256()
    try:
        with model_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ShapCacheContractError(
            f"Unable to read SHAP model '{model_path}': {exc}"
        ) from exc

    return {
        "name": model_path.name,
        "size": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def build_cache_identity(
    *,
    case_index: int,
    model_path: str | Path,
    input_data: np.ndarray,
    background_dataset: np.ndarray,
    explainer_type: str,
) -> Dict[str, Any]:
    if explainer_type not in {"gradient", "kernel"}:
        raise ValueError(f"Unsupported SHAP explainer type: {explainer_type}")
    return {
        "schema_version": TARGET_CLASS_SHAP_SCHEMA_VERSION,
        "attribution_target": TARGET_CLASS_ATTRIBUTION,
        "case_index": int(case_index),
        "explainer_type": explainer_type,
        "model": fingerprint_file(model_path),
        "input": fingerprint_array(input_data),
        "background": fingerprint_array(background_dataset),
    }


def build_cache_metadata(
    *,
    case_index: int,
    model_path: str | Path,
    input_data: np.ndarray,
    background_dataset: np.ndarray,
    explainer_type: str,
    target_class: int,
    class_count: int,
    background_per_class: Optional[int] = None,
    background_seed: Optional[int] = None,
) -> Dict[str, Any]:
    if class_count < 2:
        raise ValueError("Target-class SHAP requires a multiclass model")
    if target_class < 0 or target_class >= class_count:
        raise ValueError(
            f"target_class {target_class} is outside class_count {class_count}"
        )

    metadata = build_cache_identity(
        case_index=case_index,
        model_path=model_path,
        input_data=input_data,
        background_dataset=background_dataset,
        explainer_type=explainer_type,
    )
    metadata.update(
        {
            "target_class": int(target_class),
            "class_count": int(class_count),
        }
    )
    if background_per_class is not None:
        metadata["background_per_class"] = int(background_per_class)
    if background_seed is not None:
        metadata["background_seed"] = int(background_seed)
    return metadata


def validate_cache_metadata(
    metadata: object,
    *,
    expected_identity: Optional[Mapping[str, Any]] = None,
    case_index: Optional[int] = None,
) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ShapCacheContractError(
            "SHAP cache has no target-class metadata; regenerate it with "
            "python -m pyct.shap --force-refresh"
        )

    required = {
        "schema_version": TARGET_CLASS_SHAP_SCHEMA_VERSION,
        "attribution_target": TARGET_CLASS_ATTRIBUTION,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ShapCacheContractError(
                f"SHAP cache metadata {key}={metadata.get(key)!r}, expected {expected!r}; "
                "regenerate it with python -m pyct.shap --force-refresh"
            )

    target_class = metadata.get("target_class")
    class_count = metadata.get("class_count")
    if (
        not isinstance(target_class, int)
        or isinstance(target_class, bool)
        or not isinstance(class_count, int)
        or isinstance(class_count, bool)
        or class_count < 2
        or target_class < 0
        or target_class >= class_count
    ):
        raise ShapCacheContractError(
            "SHAP cache has invalid target_class/class_count metadata"
        )

    if metadata.get("explainer_type") not in {"gradient", "kernel"}:
        raise ShapCacheContractError("SHAP cache has invalid explainer_type metadata")

    model = metadata.get("model")
    if (
        not isinstance(model, dict)
        or not isinstance(model.get("name"), str)
        or not isinstance(model.get("size"), int)
        or not isinstance(model.get("sha256"), str)
    ):
        raise ShapCacheContractError("SHAP cache has invalid model fingerprint metadata")

    for key in ("input", "background"):
        fingerprint = metadata.get(key)
        if (
            not isinstance(fingerprint, dict)
            or not isinstance(fingerprint.get("shape"), list)
            or not isinstance(fingerprint.get("dtype"), str)
            or not isinstance(fingerprint.get("sha256"), str)
        ):
            raise ShapCacheContractError(
                f"SHAP cache has invalid {key} fingerprint metadata"
            )

    if case_index is not None and metadata.get("case_index") != int(case_index):
        raise ShapCacheContractError(
            f"SHAP cache case_index={metadata.get('case_index')!r}, expected {case_index}"
        )

    if expected_identity is not None:
        for key, expected in expected_identity.items():
            if metadata.get(key) != expected:
                raise ShapCacheContractError(
                    f"SHAP cache identity mismatch for {key}; regenerate the cache"
                )
    return dict(metadata)


def load_target_class_cache(
    path: Path,
    *,
    expected_identity: Optional[Mapping[str, Any]] = None,
    case_index: Optional[int] = None,
) -> tuple[Dict[str, Any], Dict[str, float]]:
    import numpy as np

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise ShapCacheContractError(f"Unable to read SHAP cache {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ShapCacheContractError(f"SHAP cache {path} is not a JSON object")
    metadata = validate_cache_metadata(
        payload.get("__meta__"),
        expected_identity=expected_identity,
        case_index=case_index,
    )
    values = payload.get("values")
    if not isinstance(values, dict):
        raise ShapCacheContractError(
            f"SHAP cache {path} does not contain a values object"
        )
    try:
        numeric_values = {str(key): float(value) for key, value in values.items()}
    except (TypeError, ValueError) as exc:
        raise ShapCacheContractError(
            f"SHAP cache {path} contains a non-numeric value"
        ) from exc
    if not np.isfinite(np.fromiter(numeric_values.values(), dtype=np.float64)).all():
        raise ShapCacheContractError(f"SHAP cache {path} contains NaN or Inf")
    return metadata, numeric_values


def select_target_class_values(
    raw_values: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    target_class: int,
    batched_input_shape: Tuple[int, ...],
) -> np.ndarray:
    import numpy as np

    expected = tuple(int(dim) for dim in batched_input_shape)
    if not expected or expected[0] != 1:
        raise ValueError(
            "Target-class SHAP cache generation requires exactly one input sample; "
            f"got shape {expected}"
        )

    if isinstance(raw_values, (list, tuple)):
        if target_class < 0 or target_class >= len(raw_values):
            raise ValueError(
                f"target_class {target_class} is outside {len(raw_values)} SHAP outputs"
            )
        selected = np.asarray(raw_values[target_class])
    else:
        array = np.asarray(raw_values)
        if array.shape == expected:
            if target_class != 0:
                raise ValueError(
                    f"Single-output SHAP values cannot select target class {target_class}"
                )
            selected = array
        elif array.ndim == len(expected) + 1 and array.shape[:-1] == expected:
            if target_class >= array.shape[-1]:
                raise ValueError(
                    f"target_class {target_class} is outside trailing SHAP class axis {array.shape[-1]}"
                )
            selected = array[..., target_class]
        elif array.ndim == len(expected) + 1 and array.shape[1:] == expected:
            if target_class >= array.shape[0]:
                raise ValueError(
                    f"target_class {target_class} is outside leading SHAP class axis {array.shape[0]}"
                )
            selected = array[target_class]
        else:
            raise ValueError(
                f"Unsupported SHAP output shape {array.shape}; expected input shape {expected} "
                "with one class axis"
            )

    if selected.shape != expected:
        raise ValueError(
            f"Selected SHAP shape {selected.shape} does not match input shape {expected}"
        )
    if not np.isfinite(selected).all():
        raise ValueError("Selected SHAP values contain NaN or Inf")
    return np.asarray(selected[0])
