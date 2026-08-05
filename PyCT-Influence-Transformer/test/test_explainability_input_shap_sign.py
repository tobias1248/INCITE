import json
from pathlib import Path

import numpy as np
import pytest

from explainability.input_shap_sign import (
    BOUNDS_MODE_CLIP,
    BOUNDS_MODE_STRICT,
    TargetClassInputShapProvider,
    build_sign_mask,
    count_clipped_values,
    derive_valid_shift_interval,
    materialize_shifted_input,
)
from explainability.shap_contract import build_cache_metadata


def _input_values(values: np.ndarray):
    return {
        "-1_" + "_".join(str(axis) for axis in index): float(values[index])
        for index in np.ndindex(values.shape)
    }


def _write_canonical_cache(
    path: Path,
    *,
    model_path: Path,
    case_index: int,
    sample: np.ndarray,
    background: np.ndarray,
    target_class: int,
    values: np.ndarray,
) -> None:
    metadata = build_cache_metadata(
        case_index=case_index,
        model_path=model_path,
        input_data=sample[np.newaxis, ...],
        background_dataset=background,
        explainer_type="gradient",
        target_class=target_class,
        class_count=2,
        background_per_class=1,
        background_seed=2233,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"__meta__": metadata, "values": _input_values(values)}),
        encoding="utf-8",
    )


def test_build_sign_mask_honors_epsilon() -> None:
    values = np.array([-0.2, -0.01, 0.0, 0.01, 0.2])

    signs = build_sign_mask(values, epsilon=0.05)

    np.testing.assert_array_equal(signs, np.array([-1, 0, 0, 0, 1], dtype=np.int8))


def test_build_sign_mask_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_sign_mask(np.array([0.1, np.nan]))


def test_derive_valid_shift_interval_intersects_all_input_bounds() -> None:
    sample = np.array([0.02, 0.98, 0.50], dtype=np.float32)
    signs = np.array([1, 1, -1], dtype=np.int8)

    lower, upper = derive_valid_shift_interval(
        sample,
        signs,
        requested_min=-0.1,
        requested_max=0.1,
    )

    assert lower == pytest.approx(-0.02)
    assert upper == pytest.approx(0.02)


def test_strict_materialization_rejects_out_of_range_candidate() -> None:
    sample = np.array([0.95, 0.2], dtype=np.float32)
    signs = np.array([1, -1], dtype=np.int8)

    with pytest.raises(ValueError, match="strict mode"):
        materialize_shifted_input(
            sample,
            signs,
            0.1,
            bounds_mode=BOUNDS_MODE_STRICT,
        )


def test_clip_materialization_clips_and_counts_saturated_values() -> None:
    sample = np.array([0.95, 0.2], dtype=np.float32)
    signs = np.array([1, -1], dtype=np.int8)

    shifted = materialize_shifted_input(
        sample,
        signs,
        0.1,
        bounds_mode=BOUNDS_MODE_CLIP,
    )

    np.testing.assert_allclose(shifted, np.array([1.0, 0.1], dtype=np.float32))
    assert count_clipped_values(sample, signs, 0.1) == 1


def test_provider_loads_canonical_target_class_cache(tmp_path: Path) -> None:
    model_path = tmp_path / "model" / "demo.h5"
    model_path.parent.mkdir()
    model_path.write_bytes(b"model-v1")
    sample = np.array([[[0.2], [0.8]]], dtype=np.float32)
    background = np.stack([sample, sample])
    expected_values = np.array([[[2.0], [-5.0]]], dtype=np.float64)
    output_root = tmp_path / "shap_target_class"
    cache_path = output_root / "demo" / "shap_value_3.json"
    _write_canonical_cache(
        cache_path,
        model_path=model_path,
        case_index=3,
        sample=sample,
        background=background,
        target_class=1,
        values=expected_values,
    )
    provider = TargetClassInputShapProvider(
        model_path=model_path,
        output_root=output_root,
    )

    artifact = provider.ensure(
        case_index=3,
        sample=sample,
        background=background,
        target_class=1,
    )

    assert artifact.was_cached is True
    assert artifact.target_class == 1
    assert artifact.metadata["attribution_target"] == "original_prediction"
    np.testing.assert_array_equal(artifact.values, expected_values)


def test_provider_refreshes_cache_with_wrong_target_class(tmp_path: Path) -> None:
    model_path = tmp_path / "model" / "demo.h5"
    model_path.parent.mkdir()
    model_path.write_bytes(b"model-v1")
    sample = np.array([[[0.2], [0.8]]], dtype=np.float32)
    background = np.stack([sample, sample])
    output_root = tmp_path / "shap_target_class"
    cache_path = output_root / "demo" / "shap_value_3.json"
    _write_canonical_cache(
        cache_path,
        model_path=model_path,
        case_index=3,
        sample=sample,
        background=background,
        target_class=0,
        values=np.zeros_like(sample),
    )
    ensure_calls = []

    class _Calculator:
        target_class = 1
        last_timing = {"was_cached": False}

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def ensure(self, *, assume_cached, force_refresh):
            ensure_calls.append((assume_cached, force_refresh))
            _write_canonical_cache(
                cache_path,
                model_path=model_path,
                case_index=3,
                sample=sample,
                background=background,
                target_class=1,
                values=np.ones_like(sample),
            )

    provider = TargetClassInputShapProvider(
        model_path=model_path,
        output_root=output_root,
        calculator_factory=_Calculator,
    )

    artifact = provider.ensure(
        case_index=3,
        sample=sample,
        background=background,
        target_class=1,
    )

    assert ensure_calls == [(False, True)]
    assert artifact.was_cached is False
    np.testing.assert_array_equal(artifact.values, np.ones_like(sample))
