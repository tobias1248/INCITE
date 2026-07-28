from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from explainability.shap_contract import (
    DEFAULT_TARGET_CLASS_SHAP_ROOT,
    ShapCacheContractError,
    build_cache_metadata,
    fingerprint_array,
    load_target_class_cache,
    select_target_class_values,
)


def _metadata(tmp_path: Path) -> dict[str, object]:
    model_path = tmp_path / "demo.h5"
    model_path.write_bytes(b"model-v1")
    return build_cache_metadata(
        case_index=4,
        model_path=model_path,
        input_data=np.zeros((1, 2, 2, 1), dtype=np.float32),
        background_dataset=np.ones((3, 2, 2, 1), dtype=np.float32),
        explainer_type="gradient",
        target_class=2,
        class_count=3,
        background_per_class=1,
        background_seed=7,
    )


def test_target_class_contract_uses_isolated_root_and_stable_fingerprints() -> None:
    assert DEFAULT_TARGET_CLASS_SHAP_ROOT == "shap_target_class"
    first = fingerprint_array(np.array([[1.0, 2.0]], dtype=np.float32))
    same = fingerprint_array(np.array([[1.0, 2.0]], dtype=np.float32))
    changed = fingerprint_array(np.array([[1.0, 3.0]], dtype=np.float32))

    assert first == same
    assert first["sha256"] != changed["sha256"]


def test_load_target_class_cache_validates_metadata_and_numeric_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shap_value_4.json"
    metadata = _metadata(tmp_path)
    path.write_text(
        json.dumps({"__meta__": metadata, "values": {"-1_0_0_0": 0.25}}),
        encoding="utf-8",
    )

    loaded_meta, values = load_target_class_cache(path, case_index=4)

    assert loaded_meta["target_class"] == 2
    assert values == {"-1_0_0_0": 0.25}


def test_load_target_class_cache_rejects_legacy_or_wrong_case(tmp_path: Path) -> None:
    path = tmp_path / "shap_value_4.json"
    path.write_text(json.dumps({"-1_0_0_0": 0.25}), encoding="utf-8")

    with pytest.raises(ShapCacheContractError, match="no target-class metadata"):
        load_target_class_cache(path, case_index=4)

    metadata = _metadata(tmp_path)
    path.write_text(
        json.dumps({"__meta__": metadata, "values": {"-1_0_0_0": 0.25}}),
        encoding="utf-8",
    )
    with pytest.raises(ShapCacheContractError, match="case_index"):
        load_target_class_cache(path, case_index=5)


def test_select_target_class_values_supports_list_and_class_axes() -> None:
    class_maps = [
        np.full((1, 2, 2, 1), value, dtype=np.float32)
        for value in (1.0, 2.0, 3.0)
    ]
    expected = np.full((2, 2, 1), 3.0, dtype=np.float32)

    from_list = select_target_class_values(
        class_maps,
        target_class=2,
        batched_input_shape=(1, 2, 2, 1),
    )
    trailing = select_target_class_values(
        np.stack(class_maps, axis=-1),
        target_class=2,
        batched_input_shape=(1, 2, 2, 1),
    )
    leading = select_target_class_values(
        np.stack(class_maps, axis=0),
        target_class=2,
        batched_input_shape=(1, 2, 2, 1),
    )

    assert np.array_equal(from_list, expected)
    assert np.array_equal(trailing, expected)
    assert np.array_equal(leading, expected)


def test_select_target_class_values_rejects_batches_and_unknown_shapes() -> None:
    with pytest.raises(ValueError, match="exactly one input"):
        select_target_class_values(
            [np.zeros((2, 3), dtype=np.float32)],
            target_class=0,
            batched_input_shape=(2, 3),
        )
    with pytest.raises(ValueError, match="Unsupported SHAP output shape"):
        select_target_class_values(
            np.zeros((4, 5), dtype=np.float32),
            target_class=1,
            batched_input_shape=(1, 3),
        )
