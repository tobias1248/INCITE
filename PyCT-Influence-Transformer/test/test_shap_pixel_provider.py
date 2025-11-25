from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from libct.shap_pixel_provider import JsonShapPixelProvider


def _write_shap_json(root: Path, model: str, idx: int, payload: dict[str, float]) -> None:
    path = root / "shap_value_all_layer" / model / f"shap_value_{idx}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


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
