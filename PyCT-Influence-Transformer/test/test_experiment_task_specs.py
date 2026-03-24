from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.dataset as dataset_mod
import utils.experiment_task_specs as specs


class _DummyDataset:
    def __init__(self) -> None:
        self.x_test = np.zeros((2, 32, 32, 3), dtype=np.float32)

    def get_cifar10_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict = {"dummy": float(idx)}
        con_dict = {}
        for i, j, k in attack_pixels:
            con_dict[f"v_{i}_{j}_{k}"] = 1
        return in_dict, con_dict, np.zeros((32, 32, 3), dtype=np.float32), np.zeros((1, 32, 32, 3), dtype=np.float32)


class _DummyPixelProvider:
    last_init = None

    def __init__(self, **kwargs) -> None:
        type(self).last_init = kwargs

    def top_pixels(self, idx, ton):
        assert ton == 1
        return [(4, 5, 2)]


def test_cifar10_transformer_shap_patch_selector_builds_single_channel(monkeypatch) -> None:
    monkeypatch.setattr(dataset_mod, "Cifar10Dataset", _DummyDataset)
    monkeypatch.setattr(specs, "JsonShapPixelProvider", _DummyPixelProvider)

    inputs = specs.cifar10_transformer_shap(
        "cifar10_transformer_tiny",
        first_n_img=[0],
        force=True,
        ton_values=(1,),
        attack_mode="shap_patchshap_solver60s",
        pixel_selector="patch-shap",
    )

    assert len(inputs) == 1
    ton_plan = inputs[0]["ton_plans"][0]
    assert ton_plan["con_dict"] == {"v_4_5_2": 1}
    assert _DummyPixelProvider.last_init["selector"] == "patch-shap"


def test_cifar10_transformer_shap_patch_selector_rejects_multi_ton() -> None:
    with pytest.raises(ValueError, match="patch-shap supports only --pixel-search 1"):
        specs.cifar10_transformer_shap(
            "cifar10_transformer_tiny",
            first_n_img=[0],
            force=True,
            ton_values=(1, 2),
            attack_mode="shap_patchshap_solver60s",
            pixel_selector="patch-shap",
        )


def test_cifar10_transformer_shap_token_selector_builds_single_channel(monkeypatch) -> None:
    monkeypatch.setattr(dataset_mod, "Cifar10Dataset", _DummyDataset)
    monkeypatch.setattr(specs, "JsonShapPixelProvider", _DummyPixelProvider)

    inputs = specs.cifar10_transformer_shap(
        "cifar10_cctlike_eight_mha",
        first_n_img=[0],
        force=True,
        ton_values=(1,),
        attack_mode="shap_tokenshap_solver60s",
        pixel_selector="token-shap",
    )

    assert len(inputs) == 1
    ton_plan = inputs[0]["ton_plans"][0]
    assert ton_plan["con_dict"] == {"v_4_5_2": 1}
    assert _DummyPixelProvider.last_init["selector"] == "token-shap"


def test_cifar10_transformer_shap_token_selector_rejects_multi_ton() -> None:
    with pytest.raises(ValueError, match="token-shap supports only --pixel-search 1"):
        specs.cifar10_transformer_shap(
            "cifar10_cctlike_eight_mha",
            first_n_img=[0],
            force=True,
            ton_values=(1, 2),
            attack_mode="shap_tokenshap_solver60s",
            pixel_selector="token-shap",
        )
