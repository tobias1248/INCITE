from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tasks.builders.legacy as legacy
from tasks.types import GenerationResult


def test_pyct_random_builder_uses_new_task_generation_spec(monkeypatch) -> None:
    captured = {}

    def fake_generate_inputs(model_name, first_n_img, spec, *, skip_existing_override=None):
        captured['model_name'] = model_name
        captured['first_n_img'] = first_n_img
        captured['spec'] = spec
        return GenerationResult(inputs=[{'ok': True}], skipped=0)

    monkeypatch.setattr(legacy, 'generate_inputs', fake_generate_inputs)
    monkeypatch.setattr(legacy, '_deterministic_random_provider', lambda *args, **kwargs: (lambda idx, ton: [(idx, ton)]))

    outputs = legacy.pyct_random_1_4_8_16_32('demo-model', 5)

    assert outputs == [{'ok': True}]
    assert captured['model_name'] == 'demo-model'
    assert captured['first_n_img'] == 5
    assert list(captured['spec'].ton_values) == [1, 4, 8, 16, 32]
    assert [mode.identifier for mode in captured['spec'].queue_modes] == ['queue', 'stack']
