from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import h5py

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explainability.pixel_provider import (
    JsonShapPixelProvider,
    TokenizerSpec,
    _coerce_int_list,
    _extract_input_hw,
    _extract_layers,
    _load_model_config,
    _resolve_model_path,
    build_shap_tensor_from_json,
    infer_patch_size_for_model,
    infer_tokenizer_spec_for_model,
)
from explainability.shap_contract import (
    DEFAULT_TARGET_CLASS_SHAP_ROOT,
    ShapCacheContractError,
)


def _target_class_payload(
    idx: int,
    values: object,
    *,
    model_name: str = "dummy",
) -> dict[str, object]:
    return {
        "__meta__": {
            "schema_version": 2,
            "attribution_target": "original_prediction",
            "case_index": idx,
            "target_class": 1,
            "class_count": 3,
            "explainer_type": "gradient",
            "model": {
                "name": f"{model_name}.h5",
                "size": 1,
                "sha256": "model-hash",
            },
            "input": {
                "shape": [1, 2, 2, 1],
                "dtype": "<f4",
                "sha256": "input-hash",
            },
            "background": {
                "shape": [3, 2, 2, 1],
                "dtype": "<f4",
                "sha256": "background-hash",
            },
        },
        "values": values,
    }


def _write_shap_json(root: Path, model: str, idx: int, payload: dict[str, float]) -> None:
    path = root / "shap_value_all_layer" / model / f"shap_value_{idx}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_target_class_payload(idx, payload, model_name=model), handle)


def _write_model_config(root: Path, model: str, layers: list[dict[str, object]]) -> None:
    path = root / "model" / f"{model}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_name": "Functional",
        "config": {
            "layers": layers,
        },
    }
    with h5py.File(path, "w") as handle:
        handle.attrs["model_config"] = json.dumps(payload)


def test_resolve_model_path_accepts_explicit_h5_or_model_name() -> None:
    assert _resolve_model_path("demo") == Path("model") / "demo.h5"
    assert _resolve_model_path("/tmp/demo.h5") == Path("/tmp/demo.h5")


def test_load_model_config_requires_json_object_and_readable_file(tmp_path: Path) -> None:
    bad_path = tmp_path / "model" / "bad.h5"
    bad_path.parent.mkdir(parents=True)
    with h5py.File(bad_path, "w") as handle:
        handle.attrs["model_config"] = json.dumps(["not", "object"])

    with pytest.raises(ValueError, match="not a JSON object"):
        _load_model_config(bad_path)

    with pytest.raises(FileNotFoundError, match="Unable to read model file"):
        _load_model_config(tmp_path / "model" / "missing.h5")


def test_extract_layers_and_input_hw_validate_structure() -> None:
    with pytest.raises(ValueError, match="unsupported layer config structure"):
        _extract_layers({"config": {"layers": {}}}, Path("demo.h5"))

    assert _coerce_int_list([1, "2"], expected_len=2) == (1, 2)
    assert _coerce_int_list("12", expected_len=2) is None
    assert _extract_input_hw(
        [
            {"class_name": "InputLayer", "config": {"batch_input_shape": [None, 28, 28, 1]}},
        ]
    ) == (28, 28)


def test_top_pixels_sorted_and_padded(tmp_path: Path) -> None:
    model = "dummy"
    _write_shap_json(
        tmp_path,
        model,
        0,
        {
            "-1_0_0": 0.1,
            "-1_1_2": -0.9,
            "-1_5_5": 0.4,
            "0_0": 2.0,
        },
    )

    provider = JsonShapPixelProvider(
        model_name=model,
        shap_root=str(tmp_path / "shap_value_all_layer"),
        coordinate_dims=3,
    )
    coords = provider.top_pixels(0, ton=2)
    assert coords == [(1, 2, 0), (5, 5, 0)]


def test_pixel_provider_defaults_to_target_class_root() -> None:
    provider = JsonShapPixelProvider(model_name="demo")

    assert provider.shap_root == Path(DEFAULT_TARGET_CLASS_SHAP_ROOT)


def test_pixel_provider_rejects_cache_for_different_model(tmp_path: Path) -> None:
    path = tmp_path / "shap_value_all_layer" / "cache_folder" / "shap_value_0.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(_target_class_payload(0, {"-1_0_0": 1.0})),
        encoding="utf-8",
    )
    provider = JsonShapPixelProvider(
        model_name="cache_folder",
        shap_root=str(tmp_path / "shap_value_all_layer"),
    )

    with pytest.raises(ValueError, match="expected 'cache_folder'"):
        provider.top_pixels(0, ton=1)


def test_build_tensor_matches_topk(tmp_path: Path) -> None:
    model = "dummy"
    _write_shap_json(
        tmp_path,
        model,
        0,
        {
            "-1_0_0": 0.1,
            "-1_1_1": 0.9,
        },
    )
    _write_shap_json(
        tmp_path,
        model,
        1,
        {
            "-1_0_0": 0.5,
            "-1_2_2": 0.7,
        },
    )

    provider = JsonShapPixelProvider(
        model_name=model,
        shap_root=str(tmp_path / "shap_value_all_layer"),
        coordinate_dims=3,
    )
    tensor = provider.build_tensor([0, 1], topk=1)
    assert tensor.shape == (2, 1, 3)
    assert np.array_equal(tensor[0, 0], np.array([1, 1, 0]))
    assert np.array_equal(tensor[1, 0], np.array([2, 2, 0]))


def test_as_array_and_build_shap_tensor_helper(tmp_path: Path) -> None:
    model = "dummy"
    _write_shap_json(tmp_path, model, 0, {"-1_0_0": 0.2, "-1_1_1": 0.8})

    provider = JsonShapPixelProvider(
        model_name=model,
        shap_root=str(tmp_path / "shap_value_all_layer"),
        coordinate_dims=3,
    )

    array = provider.as_array(0, ton=1)
    rebuilt = build_shap_tensor_from_json(
        [0],
        model_name=model,
        shap_root=str(tmp_path / "shap_value_all_layer"),
        coordinate_dims=3,
    )

    assert array.shape == (1, 3)
    assert np.array_equal(array[0], np.array([1, 1, 0]))
    assert rebuilt.shape == (1, 2, 3)


def test_invalid_ton_raises(tmp_path: Path) -> None:
    model = "dummy"
    _write_shap_json(tmp_path, model, 0, {"-1_0_0": 0.1})
    provider = JsonShapPixelProvider(
        model_name=model,
        shap_root=str(tmp_path / "shap_value_all_layer"),
        coordinate_dims=3,
    )
    with pytest.raises(ValueError):
        provider.top_pixels(0, ton=0)


def test_provider_rejects_invalid_selector_and_coordinate_args() -> None:
    with pytest.raises(ValueError, match="Unsupported selector"):
        JsonShapPixelProvider(model_name="demo", selector="bad-selector")
    with pytest.raises(ValueError, match="coordinate_dims must be positive"):
        JsonShapPixelProvider(model_name="demo", coordinate_dims=0)
    with pytest.raises(ValueError, match="coordinate_bounds must contain positive integers"):
        JsonShapPixelProvider(model_name="demo", coordinate_bounds=(0, 1, 2))


def test_infer_patch_size_from_model_config(tmp_path: Path) -> None:
    model = "tiny"
    _write_model_config(
        tmp_path,
        model,
        [
            {
                "class_name": "InputLayer",
                "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"},
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "name": "patch_embedding",
                    "kernel_size": [4, 4],
                    "strides": [4, 4],
                },
            },
            {
                "class_name": "Reshape",
                "config": {"name": "flatten_patches", "target_shape": [64, 96]},
            },
        ],
    )

    assert infer_patch_size_for_model(model, model_root=str(tmp_path / "model")) == 4


def test_infer_tokenizer_spec_detects_patch_model(tmp_path: Path) -> None:
    model = "tiny"
    _write_model_config(
        tmp_path,
        model,
        [
            {
                "class_name": "InputLayer",
                "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"},
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "name": "patch_embedding",
                    "kernel_size": [4, 4],
                    "strides": [4, 4],
                },
            },
            {
                "class_name": "Reshape",
                "config": {"name": "flatten_patches", "target_shape": [64, 96]},
            },
        ],
    )

    spec = infer_tokenizer_spec_for_model(model, model_root=str(tmp_path / "model"))
    assert spec.kind == "patch_2d"
    assert spec.input_hw == (32, 32)
    assert spec.patch_size == 4
    assert spec.token_count_before == 64


def test_infer_tokenizer_spec_detects_sequence_pool_model(tmp_path: Path) -> None:
    model = "cifar10_cctlike_eight_mha"
    _write_model_config(
        tmp_path,
        model,
        [
            {
                "class_name": "InputLayer",
                "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"},
            },
            {
                "class_name": "Reshape",
                "config": {"name": "tokens", "target_shape": [1024, 3]},
            },
            {
                "class_name": "AveragePooling1D",
                "config": {"name": "token_pool", "pool_size": [4], "strides": [4]},
            },
        ],
    )

    spec = infer_tokenizer_spec_for_model(model, model_root=str(tmp_path / "model"))
    assert spec.kind == "sequence_pool_1d"
    assert spec.input_hw == (32, 32)
    assert spec.token_count_before == 1024
    assert spec.token_count_after == 256
    assert spec.pool_size == 4
    assert spec.stride == 4


def test_infer_tokenizer_spec_detects_sequence_model_without_pooling(tmp_path: Path) -> None:
    model = "cifar10_cctlike_single_mha"
    _write_model_config(
        tmp_path,
        model,
        [
            {
                "class_name": "InputLayer",
                "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"},
            },
            {
                "class_name": "Reshape",
                "config": {"name": "reshape_to_tokens", "target_shape": [1024, 3]},
            },
            {
                "class_name": "MultiHeadAttention",
                "config": {"name": "mha_stage1", "num_heads": 4, "key_dim": 16},
            },
        ],
    )

    spec = infer_tokenizer_spec_for_model(model, model_root=str(tmp_path / "model"))
    assert spec.kind == "sequence_pool_1d"
    assert spec.token_count_before == 1024
    assert spec.token_count_after == 1024
    assert spec.pool_size == 1
    assert spec.stride == 1


def test_infer_patch_size_rejects_mismatched_stride(tmp_path: Path) -> None:
    model = "bad"
    _write_model_config(
        tmp_path,
        model,
        [
            {
                "class_name": "InputLayer",
                "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"},
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "name": "patch_embedding",
                    "kernel_size": [4, 4],
                    "strides": [2, 2],
                },
            },
            {
                "class_name": "Reshape",
                "config": {"name": "flatten_patches", "target_shape": [64, 96]},
            },
        ],
    )

    with pytest.raises(ValueError, match="Unable to infer patch size"):
        infer_patch_size_for_model(model, model_root=str(tmp_path / "model"))


def test_infer_tokenizer_spec_rejects_ambiguous_or_unsupported_models(tmp_path: Path) -> None:
    ambiguous = "ambiguous"
    _write_model_config(
        tmp_path,
        ambiguous,
        [
            {"class_name": "InputLayer", "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"}},
            {"class_name": "Conv2D", "config": {"name": "patch_embedding", "kernel_size": [4, 4], "strides": [4, 4]}},
            {"class_name": "Reshape", "config": {"name": "flatten_patches", "target_shape": [64, 96]}},
            {"class_name": "Reshape", "config": {"name": "tokens", "target_shape": [1024, 3]}},
        ],
    )
    unsupported = "unsupported"
    _write_model_config(
        tmp_path,
        unsupported,
        [
            {"class_name": "InputLayer", "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"}},
            {"class_name": "Dense", "config": {"name": "dense"}},
        ],
    )

    with pytest.raises(ValueError, match="matched multiple tokenizer heuristics"):
        infer_tokenizer_spec_for_model(ambiguous, model_root=str(tmp_path / "model"))
    with pytest.raises(ValueError, match="Unsupported model tokenizer"):
        infer_tokenizer_spec_for_model(unsupported, model_root=str(tmp_path / "model"))


def test_patch_shap_selects_top_coordinate_within_top_patch(tmp_path: Path) -> None:
    model = "tiny"
    _write_model_config(
        tmp_path,
        model,
        [
            {
                "class_name": "InputLayer",
                "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"},
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "name": "patch_embedding",
                    "kernel_size": [4, 4],
                    "strides": [4, 4],
                },
            },
            {
                "class_name": "Reshape",
                "config": {"name": "flatten_patches", "target_shape": [64, 96]},
            },
        ],
    )
    _write_shap_json(
        tmp_path,
        model,
        0,
        {
            "-1_0_0_0": 0.90,
            "-1_4_4_1": 0.61,
            "-1_4_5_2": 0.60,
            "-1_7_7_0": 0.01,
        },
    )

    provider = JsonShapPixelProvider(
        model_name=model,
        shap_root=str(tmp_path / "shap_value_all_layer"),
        selector="patch-shap",
        coordinate_dims=3,
        coordinate_bounds=(32, 32, 3),
        model_root=str(tmp_path / "model"),
    )

    assert provider.top_pixels(0, ton=1) == [(4, 4, 1)]


def test_patch_shap_requires_patch_size_and_spatial_coords(monkeypatch, tmp_path: Path) -> None:
    model = "tiny"
    monkeypatch.setattr(
        "explainability.pixel_provider.infer_tokenizer_spec_for_model",
        lambda *args, **kwargs: TokenizerSpec(
            kind="patch_2d",
            input_hw=(32, 32),
            token_count_before=64,
            token_count_after=64,
            patch_size=4,
        ),
    )
    provider = JsonShapPixelProvider(
        model_name=model,
        shap_root=str(tmp_path / "shap_value_all_layer"),
        selector="patch-shap",
        coordinate_dims=1,
    )
    ranked = [((0,), 1.0)]

    provider.patch_size = None
    with pytest.raises(ValueError, match="patch_size is required"):
        provider._select_patch_shap_coordinates(ranked, Path("demo.json"))

    provider.patch_size = 4
    with pytest.raises(ValueError, match="requires at least 2 spatial dimensions"):
        provider._select_patch_shap_coordinates(ranked, Path("demo.json"))


def test_patch_shap_rejects_ton_greater_than_one(tmp_path: Path) -> None:
    model = "tiny"
    _write_model_config(
        tmp_path,
        model,
        [
            {
                "class_name": "InputLayer",
                "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"},
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "name": "patch_embedding",
                    "kernel_size": [4, 4],
                    "strides": [4, 4],
                },
            },
            {
                "class_name": "Reshape",
                "config": {"name": "flatten_patches", "target_shape": [64, 96]},
            },
        ],
    )
    _write_shap_json(tmp_path, model, 0, {"-1_0_0_0": 0.1})

    provider = JsonShapPixelProvider(
        model_name=model,
        shap_root=str(tmp_path / "shap_value_all_layer"),
        selector="patch-shap",
        coordinate_dims=3,
        coordinate_bounds=(32, 32, 3),
        model_root=str(tmp_path / "model"),
    )

    with pytest.raises(ValueError, match="patch-shap supports only ton=1"):
        provider.top_pixels(0, ton=2)


def test_patch_shap_rejects_sequence_pool_model(tmp_path: Path) -> None:
    model = "cifar10_cctlike_eight_mha"
    _write_model_config(
        tmp_path,
        model,
        [
            {
                "class_name": "InputLayer",
                "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"},
            },
            {
                "class_name": "Reshape",
                "config": {"name": "tokens", "target_shape": [1024, 3]},
            },
            {
                "class_name": "AveragePooling1D",
                "config": {"name": "token_pool", "pool_size": [4], "strides": [4]},
            },
        ],
    )
    _write_shap_json(tmp_path, model, 0, {"-1_0_0_0": 0.4})

    with pytest.raises(ValueError, match="patch-embedding tokenizers"):
        JsonShapPixelProvider(
            model_name=model,
            shap_root=str(tmp_path / "shap_value_all_layer"),
            selector="patch-shap",
            coordinate_dims=3,
            coordinate_bounds=(32, 32, 3),
            model_root=str(tmp_path / "model"),
        )


def test_token_shap_selects_top_coordinate_within_top_group(tmp_path: Path) -> None:
    model = "cifar10_cctlike_eight_mha"
    _write_model_config(
        tmp_path,
        model,
        [
            {
                "class_name": "InputLayer",
                "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"},
            },
            {
                "class_name": "Reshape",
                "config": {"name": "tokens", "target_shape": [1024, 3]},
            },
            {
                "class_name": "AveragePooling1D",
                "config": {"name": "token_pool", "pool_size": [4], "strides": [4]},
            },
        ],
    )
    _write_shap_json(
        tmp_path,
        model,
        0,
        {
            "-1_0_6_0": 0.90,
            "-1_0_5_1": 0.85,
            "-1_0_4_2": 0.84,
            "-1_0_7_0": 0.83,
            "-1_1_1_0": 0.99,
        },
    )

    provider = JsonShapPixelProvider(
        model_name=model,
        shap_root=str(tmp_path / "shap_value_all_layer"),
        selector="token-shap",
        coordinate_dims=3,
        coordinate_bounds=(32, 32, 3),
        model_root=str(tmp_path / "model"),
    )

    assert provider.top_pixels(0, ton=1) == [(0, 6, 0)]


def test_token_shap_rejects_patch_model(tmp_path: Path) -> None:
    model = "tiny"
    _write_model_config(
        tmp_path,
        model,
        [
            {
                "class_name": "InputLayer",
                "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"},
            },
            {
                "class_name": "Conv2D",
                "config": {
                    "name": "patch_embedding",
                    "kernel_size": [4, 4],
                    "strides": [4, 4],
                },
            },
            {
                "class_name": "Reshape",
                "config": {"name": "flatten_patches", "target_shape": [64, 96]},
            },
        ],
    )
    _write_shap_json(tmp_path, model, 0, {"-1_0_0_0": 0.4})

    with pytest.raises(ValueError, match="sequence tokenizers"):
        JsonShapPixelProvider(
            model_name=model,
            shap_root=str(tmp_path / "shap_value_all_layer"),
            selector="token-shap",
            coordinate_dims=3,
            coordinate_bounds=(32, 32, 3),
            model_root=str(tmp_path / "model"),
        )


def test_token_shap_rejects_overlapping_pooling(tmp_path: Path) -> None:
    model = "overlap"
    _write_model_config(
        tmp_path,
        model,
        [
            {
                "class_name": "InputLayer",
                "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"},
            },
            {
                "class_name": "Reshape",
                "config": {"name": "tokens", "target_shape": [1024, 3]},
            },
            {
                "class_name": "AveragePooling1D",
                "config": {"name": "token_pool", "pool_size": [4], "strides": [2]},
            },
        ],
    )
    _write_shap_json(tmp_path, model, 0, {"-1_0_0_0": 0.4})

    with pytest.raises(ValueError, match="non-overlapping sequence pooling"):
        JsonShapPixelProvider(
            model_name=model,
            shap_root=str(tmp_path / "shap_value_all_layer"),
            selector="token-shap",
            coordinate_dims=3,
            coordinate_bounds=(32, 32, 3),
            model_root=str(tmp_path / "model"),
        )


def test_token_shap_rejects_ton_greater_than_one(tmp_path: Path) -> None:
    model = "cifar10_cctlike_eight_mha"
    _write_model_config(
        tmp_path,
        model,
        [
            {
                "class_name": "InputLayer",
                "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"},
            },
            {
                "class_name": "Reshape",
                "config": {"name": "tokens", "target_shape": [1024, 3]},
            },
            {
                "class_name": "AveragePooling1D",
                "config": {"name": "token_pool", "pool_size": [4], "strides": [4]},
            },
        ],
    )
    _write_shap_json(tmp_path, model, 0, {"-1_0_0_0": 0.4})

    provider = JsonShapPixelProvider(
        model_name=model,
        shap_root=str(tmp_path / "shap_value_all_layer"),
        selector="token-shap",
        coordinate_dims=3,
        coordinate_bounds=(32, 32, 3),
        model_root=str(tmp_path / "model"),
    )

    with pytest.raises(ValueError, match="token-shap supports only ton=1"):
        provider.top_pixels(0, ton=2)


def test_token_shap_supports_sequence_model_without_pooling(tmp_path: Path) -> None:
    model = "cifar10_cctlike_single_mha"
    _write_model_config(
        tmp_path,
        model,
        [
            {
                "class_name": "InputLayer",
                "config": {"batch_input_shape": [None, 32, 32, 3], "name": "input_1"},
            },
            {
                "class_name": "Reshape",
                "config": {"name": "reshape_to_tokens", "target_shape": [1024, 3]},
            },
            {
                "class_name": "MultiHeadAttention",
                "config": {"name": "mha_stage1", "num_heads": 4, "key_dim": 16},
            },
        ],
    )
    _write_shap_json(
        tmp_path,
        model,
        0,
        {
            "-1_0_6_0": 0.90,
            "-1_0_5_1": 0.85,
            "-1_0_4_2": 0.84,
        },
    )

    provider = JsonShapPixelProvider(
        model_name=model,
        shap_root=str(tmp_path / "shap_value_all_layer"),
        selector="token-shap",
        coordinate_dims=3,
        coordinate_bounds=(32, 32, 3),
        model_root=str(tmp_path / "model"),
    )

    assert provider.top_pixels(0, ton=1) == [(0, 6, 0)]


def test_token_shap_validates_domain_and_group_bounds(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "explainability.pixel_provider.infer_tokenizer_spec_for_model",
        lambda *args, **kwargs: TokenizerSpec(
            kind="sequence_pool_1d",
            input_hw=(2, 2),
            token_count_before=4,
            token_count_after=1,
            pool_size=1,
            stride=1,
        ),
    )
    provider = JsonShapPixelProvider(
        model_name="demo",
        shap_root=str(tmp_path / "shap_value_all_layer"),
        selector="token-shap",
        coordinate_dims=3,
    )

    with pytest.raises(ValueError, match="maps outside the tokenizer domain"):
        provider._select_token_shap_coordinates([((5, 0, 0), 1.0)], Path("demo.json"))
    with pytest.raises(ValueError, match="maps outside pooled token groups"):
        provider._select_token_shap_coordinates([((1, 1, 0), 1.0)], Path("demo.json"))


def test_load_sorted_uses_cache_and_json_failures_are_reported(tmp_path: Path) -> None:
    model = "dummy"
    path = tmp_path / "shap_value_all_layer" / model / "shap_value_0.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _target_class_payload(0, {"-1_1_1": 0.9, "-1_0_0": 0.1})
        ),
        encoding="utf-8",
    )
    provider = JsonShapPixelProvider(
        model_name=model,
        shap_root=str(tmp_path / "shap_value_all_layer"),
        coordinate_dims=3,
    )

    first = provider.top_pixels(0, ton=1)
    path.write_text(
        json.dumps(_target_class_payload(0, {"-1_9_9": 9.9})),
        encoding="utf-8",
    )
    second = provider.top_pixels(0, ton=1)

    assert first == second == [(1, 1, 0)]

    missing = JsonShapPixelProvider(model_name="missing", shap_root=str(tmp_path / "shap_value_all_layer"))
    with pytest.raises(FileNotFoundError, match="Missing target-class SHAP cache"):
        missing.top_pixels(3)


def test_load_json_and_extract_ranked_items_validate_payloads(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    nested_bad = tmp_path / "nested_bad.json"
    nested_bad.write_text(json.dumps({"values": [1, 2, 3]}), encoding="utf-8")
    provider = JsonShapPixelProvider(model_name="demo", shap_root=str(tmp_path), coordinate_dims=3, coordinate_bounds=(4, 4, 2))

    with pytest.raises(ShapCacheContractError, match="not a JSON object"):
        provider._load_json(bad_json)
    with pytest.raises(ShapCacheContractError, match="no target-class metadata"):
        provider._load_json(nested_bad)
    with pytest.raises(ValueError, match="No pixel-level SHAP entries found"):
        provider._extract_ranked_pixel_items({"0_0": 1.0}, Path("demo.json"))


def test_normalize_coords_handles_padding_truncation_and_bounds(tmp_path: Path) -> None:
    provider = JsonShapPixelProvider(
        model_name="demo",
        shap_root=str(tmp_path),
        coordinate_dims=3,
        coordinate_bounds=(4, 4, 2),
    )

    assert provider._normalize_coords((1, 2)) == (1, 2, 0)
    assert provider._normalize_coords((1, 2, 1)) == (1, 2, 1)
    assert provider._normalize_coords((1, 2, 3)) is None
    assert provider._normalize_coords((1, 2, 3, 4)) is None


def test_build_tensor_rejects_empty_indices(tmp_path: Path) -> None:
    provider = JsonShapPixelProvider(model_name="demo", shap_root=str(tmp_path), coordinate_dims=3)

    with pytest.raises(ValueError, match="indices must be non-empty"):
        provider.build_tensor([])
