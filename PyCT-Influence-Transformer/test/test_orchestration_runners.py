from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import orchestration.runners as runners


def _write_stats(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": {}}), encoding="utf-8")


def test_shap_runner_advances_until_success(monkeypatch, tmp_path: Path) -> None:
    save_dir = tmp_path / "case_0"
    _write_stats(save_dir / "stats.json")

    calls = []
    updates = []
    results = [
        (
            0,
            SimpleNamespace(
                save_dir=str(save_dir),
                attack_label=None,
                solve_all_ctr=True,
                is_timeout=False,
            ),
        ),
        (
            0,
            SimpleNamespace(
                save_dir=str(save_dir),
                attack_label="adv",
                solve_all_ctr=False,
                is_timeout=False,
            ),
        ),
    ]

    runner = runners.ShapRunner(timeout=5, norm=False)
    monkeypatch.setattr(runner, "_execute_attack", lambda payload: calls.append(dict(payload)) or results.pop(0))
    monkeypatch.setattr(
        runners,
        "update_ton_progress_stats",
        lambda stats_path, **kwargs: updates.append((Path(stats_path), kwargs)) or True,
    )

    payload = {
        "idx": 0,
        "in_dict": {"v_0_0": 1.0},
        "ton_plans": [
            {"ton": 1, "con_dict": {"v_0_0": 1}, "save_exp": {"input_name": "case_0"}},
            {"ton": 2, "con_dict": {"v_0_1": 1}, "save_exp": {"input_name": "case_0"}},
        ],
    }

    result = runner._run_single(payload)

    assert result[1].attack_label == "adv"
    assert [call["con_dict"] for call in calls] == [{"v_0_0": 1}, {"v_0_1": 1}]
    assert updates[0][1]["current_ton"] == 1
    assert updates[0][1]["status"] == "continue"
    assert updates[0][1]["next_ton"] == 2
    assert updates[1][1]["current_ton"] == 2
    assert updates[1][1]["status"] == "stop"


def test_random_assign_runner_retries_until_success(monkeypatch) -> None:
    attempts = []
    logs = []
    artifact_results = []
    monotonic_values = iter([0.0, 0.2, 0.4, 0.7, 0.9])
    outcomes = [
        SimpleNamespace(success=False),
        SimpleNamespace(success=True),
    ]

    monkeypatch.setattr(runners.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        runners,
        "run_random_assign_step",
        lambda payload, pixel_source, base_seed, attempt: attempts.append((pixel_source, base_seed, attempt)) or outcomes.pop(0),
    )
    monkeypatch.setattr(runners, "write_combined_log", lambda result: logs.append(result.success))
    monkeypatch.setattr(runners, "write_experiment_artifacts", lambda result: artifact_results.append(result))

    runner = runners.RandomAssignRunner(
        timeout=2,
        norm=False,
        pixel_source="random",
        base_seed=2024,
    )

    result = runner._run_random_assign_for_plan({"idx": 3})

    assert result.success is True
    assert attempts == [("random", 2024, 0), ("random", 2024, 1)]
    assert logs == [False, True]
    assert artifact_results == [result]
    assert result.attack_wall_time == 0.9
