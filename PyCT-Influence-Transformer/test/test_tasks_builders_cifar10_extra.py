from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tasks.builders.cifar10 as builders


class _DummyCifarDataset:
    def __init__(self) -> None:
        self.x_test = np.zeros((3, 32, 32, 3), dtype=np.float32)

    def get_cifar10_test_data_and_set_condict(self, idx, attack_pixels):
        con_dict = {'v_0_0_0': 0, 'v_1_1_1': 0}
        for i, j, k in attack_pixels:
            con_dict[f'v_{i}_{j}_{k}'] = 1
        return (
            {'value': float(idx)},
            con_dict,
            np.zeros((32, 32, 3), dtype=np.float32),
            np.zeros((10, 32, 32, 3), dtype=np.float32),
        )


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
            "compute_seconds": float(self.kwargs['idx']) + 0.75,
        }


def test_cifar10_transformer_random_uses_coordinate_provider(monkeypatch) -> None:
    monkeypatch.setattr(builders, 'Cifar10Dataset', _DummyCifarDataset)
    monkeypatch.setattr(builders, 'make_coordinate_provider', lambda *args, **kwargs: (lambda idx, ton: [(1, 1, 1)] * ton))

    inputs = builders.cifar10_transformer_random('demo', [1], ton_values=(1,), force=True)

    assert len(inputs) == 1
    assert inputs[0]['ton_plans'][0]['con_dict'] == {'v_1_1_1': 1}


def test_cifar10_shap_calculate_all_runs_calculator(monkeypatch, tmp_path) -> None:
    _DummyCalculator.calls = []
    monkeypatch.setattr(builders, 'Cifar10Dataset', _DummyCifarDataset)
    monkeypatch.setattr(builders, 'ShapValuesCalculator', _DummyCalculator)

    artifacts = builders.cifar10_cal_shap_specs(
        'demo',
        first_n_img=2,
        force_refresh=True,
        output_root=str(tmp_path),
    )

    assert [item['idx'] for item in artifacts] == [0, 1]
    assert [item['computed'] for item in artifacts] == [True, True]
    assert [item['compute_seconds'] for item in artifacts] == [0.75, 1.75]
    assert _DummyCalculator.calls == [(0, False, True), (1, False, True)]
