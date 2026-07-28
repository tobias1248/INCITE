from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tasks.builders.fashion_mnist as builders


class _DummyFashionDataset:
    def __init__(self) -> None:
        self.x_test = np.zeros((3, 28, 28, 1), dtype=np.float32)

    def get_fashion_mnist_test_data_and_set_condict(self, idx, attack_pixels):
        con_dict = {'v_0_0_0': 0, 'v_2_2_0': 0}
        for i, j, k in attack_pixels:
            con_dict[f'v_{i}_{j}_{k}'] = 1
        return (
            {'value': float(idx)},
            con_dict,
            np.zeros((28, 28, 1), dtype=np.float32),
            np.zeros((10, 28, 28, 1), dtype=np.float32),
        )


class _DummyPixelProvider:
    last_kwargs = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        type(self).last_kwargs = kwargs

    def top_pixels(self, idx, ton):
        return [(0, 0, 0)] * ton


class _DummyCalculator:
    calls = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.cache_path = Path(kwargs['output_root']) / f"shap_value_{kwargs['idx']}.json"
        self.last_timing = {
            "computed": False,
            "was_cached": False,
            "compute_seconds": 0.0,
        }

    def ensure(self, assume_cached: bool, force_refresh: bool) -> None:
        type(self).calls.append((self.kwargs['idx'], assume_cached, force_refresh))
        self.last_timing = {
            "computed": not assume_cached,
            "was_cached": bool(assume_cached),
            "compute_seconds": float(self.kwargs['idx']) + 0.25,
        }


def test_fashion_mnist_transformer_shap_builds_ton_plans(monkeypatch) -> None:
    monkeypatch.setattr(builders, 'FashionMnistDataset', _DummyFashionDataset)
    monkeypatch.setattr(builders, 'JsonShapPixelProvider', _DummyPixelProvider)

    inputs = builders.fashion_mnist_transformer_shap('demo', [0], force=True, ton_values=(1, 2))

    assert len(inputs) == 1
    assert [plan['ton'] for plan in inputs[0]['ton_plans']] == [1, 2]
    assert inputs[0]['ton_plans'][0]['con_dict'] == {'v_0_0_0': 1}
    assert _DummyPixelProvider.last_kwargs['shap_root'] == 'shap_target_class'


def test_fashion_mnist_transformer_random_uses_coordinate_provider(monkeypatch) -> None:
    monkeypatch.setattr(builders, 'FashionMnistDataset', _DummyFashionDataset)
    monkeypatch.setattr(builders, 'make_coordinate_provider', lambda *args, **kwargs: (lambda idx, ton: [(2, 2, 0)] * ton))

    inputs = builders.fashion_mnist_transformer_random('demo', [1], ton_values=(1,), force=True)

    assert len(inputs) == 1
    assert inputs[0]['ton_plans'][0]['con_dict'] == {'v_2_2_0': 1}


def test_fashion_mnist_transformer_shap_calculate_all_runs_calculator(monkeypatch, tmp_path) -> None:
    _DummyCalculator.calls = []
    monkeypatch.setattr(builders, 'FashionMnistDataset', _DummyFashionDataset)
    monkeypatch.setattr(builders, 'ShapValuesCalculator', _DummyCalculator)

    artifacts = builders.fashion_mnist_transformer_shap_calculate_all(
        'demo',
        first_n_img=2,
        force_refresh=True,
        output_root=str(tmp_path),
    )

    assert [item['idx'] for item in artifacts] == [0, 1]
    assert [item['computed'] for item in artifacts] == [True, True]
    assert [item['compute_seconds'] for item in artifacts] == [0.25, 1.25]
    assert _DummyCalculator.calls == [(0, False, True), (1, False, True)]
