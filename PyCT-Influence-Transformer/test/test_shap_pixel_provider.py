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

from libct.shap_pixel_provider import JsonShapPixelProvider, infer_patch_size_for_model


def _write_shap_json(root: Path, model: str, idx: int, payload: dict[str, float]) -> None:
    path = root / "shap_value_all_layer" / model / f"shap_value_{idx}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


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
