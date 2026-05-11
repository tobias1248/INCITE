from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reporting import experiment_stats
from libct.random_assign_attack import RandomAssignResult, write_experiment_artifacts
from tasks.paths import get_save_dir_from_save_exp


def test_write_experiment_artifacts_omits_attack_label_on_failed_random_assign(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)

    result = RandomAssignResult(
        model_name="dummy_model",
        pixel_source="random",
        idx=0,
        ton=1,
        assignments=[],
        logits_before=np.array([0.1, 0.9], dtype=np.float32),
        logits_after=np.array([0.2, 0.8], dtype=np.float32),
        pred_before=1,
        pred_after=1,
        success=False,
        timestamp=datetime.now(timezone.utc),
        save_exp={"input_name": "case_0", "idx": 0, "attack_mode": "random-assign"},
        original_input=np.zeros((2, 2, 1), dtype=np.float32),
        modified_input=np.ones((2, 2, 1), dtype=np.float32),
        base_seed=314159,
        attempt=0,
    )

    write_experiment_artifacts(result)

    save_dir = Path(
        get_save_dir_from_save_exp(
            result.save_exp,
            result.model_name,
            result.save_exp["attack_mode"],
        )
    )
    payload = json.loads((save_dir / "stats.json").read_text(encoding="utf-8"))
    meta = payload["meta"]

    assert meta["success"] is False
    assert meta["attack_label"] is None
    assert meta["predicted_label_after"] == 1


def test_statistic_prefers_explicit_success_flag_over_attack_label() -> None:
    meta = {"success": False, "attack_label": 3}
    data = {}

    assert experiment_stats._derive_success(meta, data) is False
