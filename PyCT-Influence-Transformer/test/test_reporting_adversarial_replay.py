from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reporting import adversarial_replay


class FakeModel:
    def __call__(self, batch, training=False):
        outputs = []
        for sample in batch:
            marker = int(round(float(sample[0, 0, 0])))
            if marker == 10:
                outputs.append([0.9, 0.1, 0.0])
            elif marker == 20:
                outputs.append([0.1, 0.8, 0.1])
            elif marker == 30:
                outputs.append([0.05, 0.05, 0.9])
            else:
                outputs.append([0.6, 0.2, 0.2])
        return np.asarray(outputs, dtype=np.float32)


def _write_case(root: Path, case_name: str, *, idx: int, original_label: int, attack_label: int | None, ori_marker: float, adv_marker: float | None) -> None:
    case_dir = root / case_name
    case_dir.mkdir(parents=True)
    payload = {
        "meta": {
            "idx": idx,
            "model_name": "demo_model",
            "original_label": original_label,
            "attack_label": attack_label,
            "status": "success" if attack_label is not None else "timeout",
        }
    }
    (case_dir / "stats.json").write_text(json.dumps(payload), encoding="utf-8")
    np.save(case_dir / "ori_input.npy", np.full((2, 2, 1), ori_marker, dtype=np.float32))
    np.save(
        case_dir / "sat_inputs.npy",
        np.full((1, 2, 2, 1), ori_marker, dtype=np.float32),
    )
    if adv_marker is not None:
        np.save(case_dir / "adv_input.npy", np.full((2, 2, 1), adv_marker, dtype=np.float32))


def test_replay_adversarial_cases_filters_success_cases_and_flags_mismatches(tmp_path, monkeypatch) -> None:
    exp_root = tmp_path / "exp"
    _write_case(
        exp_root,
        "case_0",
        idx=0,
        original_label=0,
        attack_label=1,
        ori_marker=10,
        adv_marker=20,
    )
    _write_case(
        exp_root,
        "case_1",
        idx=1,
        original_label=0,
        attack_label=1,
        ori_marker=10,
        adv_marker=10,
    )
    _write_case(
        exp_root,
        "case_2",
        idx=2,
        original_label=0,
        attack_label=None,
        ori_marker=10,
        adv_marker=None,
    )
    model_path = tmp_path / "model" / "demo_model.h5"
    model_path.parent.mkdir()
    model_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(adversarial_replay, "_load_model", lambda *_args, **_kwargs: FakeModel())

    resolved_model_path, records = adversarial_replay.replay_adversarial_cases(
        exp_root,
        model_path=str(model_path),
        batch_size=1,
    )

    assert resolved_model_path == model_path
    assert [record.case_name for record in records] == ["case_0", "case_1"]
    assert records[0].still_adversarial is True
    assert records[0].stored_attack_label_matches_prediction is True
    assert records[1].still_adversarial is False
    assert records[1].stored_attack_label_matches_prediction is False


def test_main_json_output_and_fail_on_issues(tmp_path, monkeypatch, capsys) -> None:
    exp_root = tmp_path / "exp"
    _write_case(
        exp_root,
        "case_0",
        idx=0,
        original_label=0,
        attack_label=1,
        ori_marker=30,
        adv_marker=20,
    )
    model_path = tmp_path / "model" / "demo_model.h5"
    model_path.parent.mkdir()
    model_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(adversarial_replay, "_load_model", lambda *_args, **_kwargs: FakeModel())

    rc = adversarial_replay.main(
        [
            "--experiment-root",
            str(exp_root),
            "--model-path",
            str(model_path),
            "--json",
            "--fail-on-issues",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["total_replayed_cases"] == 1
    assert payload["still_adversarial_count"] == 1
    assert payload["original_prediction_mismatch_count"] == 1
