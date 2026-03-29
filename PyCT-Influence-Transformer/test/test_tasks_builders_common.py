from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.builders.common import (
    TaskGenerationSpec,
    generate_inputs,
    make_coordinate_provider,
    make_payload_builder,
    normalize_indices,
    normalize_ton_sequence,
    queue_modes,
)


class _DummyDataset:
    def get_case(self, idx, attack_pixels):
        con_dict = {f"v_{i}_{j}": 1 for i, j in attack_pixels}
        return {"value": idx}, con_dict


def test_normalize_indices_and_ton_sequence() -> None:
    assert normalize_indices(3) == [0, 1, 2]
    assert normalize_indices(range(1, 3)) == [1, 2]
    assert normalize_ton_sequence([1, 1, 4, 2]) == (1, 4, 2)

    with pytest.raises(ValueError, match=">= 1"):
        normalize_ton_sequence([0])


def test_make_coordinate_provider_is_deterministic_and_unique() -> None:
    provider = make_coordinate_provider((4, 4), [1, 3], base_seed=99)

    first = provider(2, 3)
    second = provider(2, 3)

    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3


def test_generate_inputs_skips_existing_outputs(monkeypatch, tmp_path) -> None:
    def save_exp_builder(idx, ton, mode):
        return {"input_name": f"case_{idx}", "exp_name": f"demo_{ton}", "attack_mode": mode.identifier}

    def fake_get_save_dir(save_exp, model_name, attack_mode, only_first_forward=False, **kwargs):
        return str(tmp_path / f"{model_name}-{attack_mode}-{save_exp['idx']}-{save_exp['exp_name']}")

    monkeypatch.setattr('tasks.builders.common.get_save_dir_from_save_exp', fake_get_save_dir)

    existing = tmp_path / 'demo-queue-0-demo_1'
    existing.mkdir(parents=True)

    spec = TaskGenerationSpec(
        dataset_factory=_DummyDataset,
        attack_pixel_fn=lambda idx, ton: [(idx, ton)],
        queue_modes=queue_modes(include_queue=True, include_stack=False),
        ton_values=[1],
        save_exp_builder=save_exp_builder,
        payload_builder=make_payload_builder('get_case'),
    )

    result = generate_inputs('demo', [0, 1], spec)

    assert result.skipped == 1
    assert len(result.inputs) == 1
    assert result.inputs[0]['idx'] == 1
    assert result.inputs[0]['con_dict'] == {'v_1_1': 1}
