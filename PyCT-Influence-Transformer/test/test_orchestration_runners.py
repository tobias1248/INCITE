from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import orchestration.runners as runners


def _write_stats(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": {}}), encoding="utf-8")


def _result_recorder(
    save_dir: Path,
    *,
    attack_label=None,
    solve_all_ctr: bool = False,
    is_timeout: bool = False,
    status: str | None = None,
    error_type: str | None = None,
    input_name: str = "case_0",
):
    extra_meta = {}
    if status is not None:
        extra_meta["status"] = status
    if error_type is not None:
        extra_meta["error_type"] = error_type
    return (
        0,
        SimpleNamespace(
            save_dir=str(save_dir),
            input_name=input_name,
            attack_label=attack_label,
            solve_all_ctr=solve_all_ctr,
            is_timeout=is_timeout,
            extra_meta=extra_meta,
        ),
    )


class _DummyRunner(runners.BaseRunner):
    def __init__(self, result=None, exc: Exception | None = None) -> None:
        super().__init__(timeout=1, norm=False, collect_constraints_with="queue")
        self._result = result
        self._exc = exc

    def _run_single(self, payload):
        if self._exc is not None:
            raise self._exc
        return self._result


def test_base_runner_run_tasks_logs_and_cleans_payload(caplog) -> None:
    runner = _DummyRunner(result=(0, SimpleNamespace(is_timeout=False, input_name="case_4", save_dir="/tmp/case_4")))
    payload = {"idx": 4, "popped_log_attack_mode": "queue_solver1s", "solve_order_stack": False}

    with caplog.at_level(logging.INFO, logger="ct.runner"):
        runner.run_tasks([payload])

    assert payload == {}
    assert "[PAYLOAD-START]" in caplog.text
    assert "[PAYLOAD-END]" in caplog.text
    assert "input_name=case_4" in caplog.text


def test_base_runner_run_tasks_logs_terminal_error_without_reraising(caplog, tmp_path: Path) -> None:
    runner = _DummyRunner(
        result=_result_recorder(
            tmp_path / "case_error",
            status="error",
            error_type="constraint_transfer_failure",
        )
    )
    payload = {"idx": 4, "popped_log_attack_mode": "queue_solver1s", "solve_order_stack": False}

    with caplog.at_level(logging.ERROR, logger="ct.runner"):
        runner.run_tasks([payload])

    assert payload == {}
    assert "[PAYLOAD-ERROR]" in caplog.text
    assert "[PAYLOAD-END]" not in caplog.text
    assert "save_dir=" in caplog.text


def test_base_runner_run_tasks_reraises_after_logging(caplog) -> None:
    runner = _DummyRunner(exc=RuntimeError("boom"))
    payload = {"idx": 2, "popped_log_attack_mode": "random_solver1s", "solve_order_stack": False}

    with caplog.at_level(logging.ERROR, logger="ct.runner"):
        with pytest.raises(RuntimeError, match="boom"):
            runner.run_tasks([payload])

    assert payload == {}
    assert "[PAYLOAD-ERROR]" in caplog.text


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


def test_queue_runner_returns_execute_attack_when_no_ton_plans(monkeypatch) -> None:
    runner = runners.QueueRunner(timeout=5, norm=False)
    marker = object()
    monkeypatch.setattr(runner, "_execute_attack", lambda payload: marker)

    result = runner._run_single({"idx": 0, "con_dict": {"v_0_0": 1}})

    assert result is marker


def test_queue_runner_stops_after_adv_found(monkeypatch, tmp_path: Path) -> None:
    save_dir = tmp_path / "case_0"
    _write_stats(save_dir / "stats.json")
    runner = runners.QueueRunner(timeout=5, norm=False)
    calls = []
    updates = []
    monkeypatch.setattr(
        runner,
        "_execute_attack",
        lambda payload: calls.append(dict(payload))
        or (0, SimpleNamespace(save_dir=str(save_dir), attack_label="adv", solve_all_ctr=False, is_timeout=False)),
    )
    monkeypatch.setattr(runners, "derive_ton_outcome", lambda recorder: (False, "adv_found"))
    monkeypatch.setattr(
        runners,
        "update_ton_progress_stats",
        lambda stats_path, **kwargs: updates.append((Path(stats_path), kwargs)),
    )

    result = runner._run_single(
        {
            "idx": 0,
            "ton_plans": [
                {"ton": 1, "con_dict": {"v_0_0": 1}, "save_exp": {"input_name": "case_0"}},
                {"ton": 2, "con_dict": {"v_0_1": 1}, "save_exp": {"input_name": "case_0"}},
            ],
        }
    )

    assert result[1].attack_label == "adv"
    assert len(calls) == 1
    assert updates[0][1]["status"] == "stop"
    assert updates[0][1]["reason"] == "adv_found"


def test_queue_runner_stops_when_should_not_continue(monkeypatch, tmp_path: Path) -> None:
    save_dir = tmp_path / "case_0"
    _write_stats(save_dir / "stats.json")
    runner = runners.QueueRunner(timeout=5, norm=False)
    calls = []
    monkeypatch.setattr(
        runner,
        "_execute_attack",
        lambda payload: calls.append(dict(payload))
        or (0, SimpleNamespace(save_dir=str(save_dir), attack_label=None, solve_all_ctr=False, is_timeout=True)),
    )
    monkeypatch.setattr(runners, "derive_ton_outcome", lambda recorder: (False, "timeout"))
    monkeypatch.setattr(runners, "update_ton_progress_stats", lambda *_args, **_kwargs: True)

    runner._run_single(
        {
            "idx": 0,
            "ton_plans": [
                {"ton": 1, "con_dict": {"v_0_0": 1}, "save_exp": {"input_name": "case_0"}},
                {"ton": 2, "con_dict": {"v_0_1": 1}, "save_exp": {"input_name": "case_0"}},
            ],
        }
    )

    assert len(calls) == 1


def test_queue_runner_retries_transfer_failure_until_success(monkeypatch, tmp_path: Path) -> None:
    save_dir = tmp_path / "case_retry_success"
    _write_stats(save_dir / "stats.json")
    runner = runners.QueueRunner(timeout=5, norm=False, error_retry_limit=2)
    calls = []
    updates = []
    results = [
        _result_recorder(
            save_dir,
            status="error",
            error_type="constraint_transfer_failure",
        ),
        _result_recorder(
            save_dir,
            attack_label="adv",
        ),
    ]
    monkeypatch.setattr(
        runner,
        "_execute_attack",
        lambda payload: calls.append(dict(payload)) or results.pop(0),
    )
    monkeypatch.setattr(
        runners,
        "update_ton_progress_stats",
        lambda stats_path, **kwargs: updates.append((Path(stats_path), kwargs)) or True,
    )

    result = runner._run_single(
        {
            "idx": 0,
            "ton_plans": [
                {"ton": 1, "con_dict": {"v_0_0": 1}, "save_exp": {"input_name": "case_0"}},
                {"ton": 2, "con_dict": {"v_0_1": 1}, "save_exp": {"input_name": "case_0"}},
            ],
        }
    )

    assert result[1].attack_label == "adv"
    assert len(calls) == 2
    assert calls[0]["con_dict"] == {"v_0_0": 1}
    assert calls[1]["con_dict"] == {"v_0_0": 1}
    assert len(updates) == 1
    assert updates[0][1]["status"] == "stop"
    assert updates[0][1]["reason"] == "adv_found"


def test_queue_runner_stops_after_transfer_retry_limit(monkeypatch, tmp_path: Path) -> None:
    save_dir = tmp_path / "case_retry_limit"
    _write_stats(save_dir / "stats.json")
    runner = runners.QueueRunner(timeout=5, norm=False, error_retry_limit=2)
    calls = []
    updates = []
    monkeypatch.setattr(
        runner,
        "_execute_attack",
        lambda payload: calls.append(dict(payload))
        or _result_recorder(
            save_dir,
            status="error",
            error_type="constraint_transfer_failure",
        ),
    )
    monkeypatch.setattr(
        runners,
        "update_ton_progress_stats",
        lambda stats_path, **kwargs: updates.append((Path(stats_path), kwargs)) or True,
    )

    result = runner._run_single(
        {
            "idx": 0,
            "ton_plans": [
                {"ton": 1, "con_dict": {"v_0_0": 1}, "save_exp": {"input_name": "case_0"}},
                {"ton": 2, "con_dict": {"v_0_1": 1}, "save_exp": {"input_name": "case_0"}},
            ],
        }
    )

    assert result[1].extra_meta["status"] == "error"
    assert len(calls) == 3
    assert [call["con_dict"] for call in calls] == [{"v_0_0": 1}, {"v_0_0": 1}, {"v_0_0": 1}]
    assert len(updates) == 1
    assert updates[0][1]["status"] == "stop"
    assert updates[0][1]["reason"] == "error_constraint_transfer_failure"


def test_queue_runner_does_not_retry_non_transfer_errors(monkeypatch, tmp_path: Path) -> None:
    save_dir = tmp_path / "case_non_retryable_error"
    _write_stats(save_dir / "stats.json")
    runner = runners.QueueRunner(timeout=5, norm=False, error_retry_limit=2)
    calls = []
    updates = []
    monkeypatch.setattr(
        runner,
        "_execute_attack",
        lambda payload: calls.append(dict(payload))
        or _result_recorder(
            save_dir,
            status="error",
            error_type="solver_crash",
        ),
    )
    monkeypatch.setattr(
        runners,
        "update_ton_progress_stats",
        lambda stats_path, **kwargs: updates.append((Path(stats_path), kwargs)) or True,
    )

    result = runner._run_single(
        {
            "idx": 0,
            "ton_plans": [
                {"ton": 1, "con_dict": {"v_0_0": 1}, "save_exp": {"input_name": "case_0"}},
                {"ton": 2, "con_dict": {"v_0_1": 1}, "save_exp": {"input_name": "case_0"}},
            ],
        }
    )

    assert result[1].extra_meta["error_type"] == "solver_crash"
    assert len(calls) == 1
    assert len(updates) == 1
    assert updates[0][1]["reason"] == "error_solver_crash"


def test_queue_runner_zero_retry_limit_disables_transfer_retry(monkeypatch, tmp_path: Path) -> None:
    save_dir = tmp_path / "case_zero_retry"
    _write_stats(save_dir / "stats.json")
    runner = runners.QueueRunner(timeout=5, norm=False, error_retry_limit=0)
    calls = []
    updates = []
    monkeypatch.setattr(
        runner,
        "_execute_attack",
        lambda payload: calls.append(dict(payload))
        or _result_recorder(
            save_dir,
            status="error",
            error_type="constraint_transfer_failure",
        ),
    )
    monkeypatch.setattr(
        runners,
        "update_ton_progress_stats",
        lambda stats_path, **kwargs: updates.append((Path(stats_path), kwargs)) or True,
    )

    runner._run_single(
        {
            "idx": 0,
            "ton_plans": [
                {"ton": 1, "con_dict": {"v_0_0": 1}, "save_exp": {"input_name": "case_0"}},
            ],
        }
    )

    assert len(calls) == 1
    assert len(updates) == 1
    assert updates[0][1]["reason"] == "error_constraint_transfer_failure"


def test_queue_runner_runs_ternary_fallback_after_timeout(monkeypatch, tmp_path: Path) -> None:
    baseline_dir = tmp_path / "case_timeout"
    fallback_dir = tmp_path / "case_timeout_ternary"
    _write_stats(fallback_dir / "stats.json")
    runner = runners.QueueRunner(
        timeout=5,
        norm=False,
        ternary_fallback=True,
        ternary_fallback_threshold_scale=1.5,
    )
    calls = []
    results = [
        _result_recorder(baseline_dir, is_timeout=True),
        _result_recorder(fallback_dir, attack_label="adv"),
    ]
    monkeypatch.setattr(
        runner,
        "_execute_attack",
        lambda payload: calls.append(dict(payload)) or results.pop(0),
    )
    monkeypatch.setattr(runners, "update_ton_progress_stats", lambda *_args, **_kwargs: True)

    result = runner._run_single(
        {
            "idx": 0,
            "popped_log_attack_mode": "queue_solver1s",
            "ton_plans": [
                {
                    "ton": 1,
                    "con_dict": {"v_0_0": 1},
                    "save_exp": {
                        "input_name": "case_0",
                        "attack_mode": "queue_solver1s",
                        "ton": 1,
                        "ton_next": 2,
                    },
                },
            ],
        }
    )

    assert result[1].attack_label == "adv"
    assert len(calls) == 2
    assert calls[0]["con_dict"] == {"v_0_0": 1}
    assert calls[1]["con_dict"] == {"v_0_0": 1}
    assert calls[1]["ternary_simplification"] is True
    assert calls[1]["ternary_threshold_scale"] == 1.5
    assert calls[1]["popped_log_attack_mode"] == "queue_solver1s_ternaryfb"
    assert calls[1]["save_exp"]["attack_mode"] == "queue_solver1s_ternaryfb"
    assert calls[1]["save_exp"]["fallback"] is True
    assert calls[1]["save_exp"]["fallback_type"] == "ternary"
    assert calls[1]["save_exp"]["fallback_trigger"] == "timeout"
    assert calls[1]["save_exp"]["fallback_source_attack_mode"] == "queue_solver1s"
    assert calls[1]["save_exp"]["fallback_source_ton"] == 1
    assert calls[1]["save_exp"]["fallback_source_ton_next"] == 2


def test_queue_runner_does_not_fallback_after_success(monkeypatch, tmp_path: Path) -> None:
    save_dir = tmp_path / "case_success"
    _write_stats(save_dir / "stats.json")
    runner = runners.QueueRunner(timeout=5, norm=False, ternary_fallback=True)
    calls = []
    monkeypatch.setattr(
        runner,
        "_execute_attack",
        lambda payload: calls.append(dict(payload)) or _result_recorder(save_dir, attack_label="adv"),
    )
    monkeypatch.setattr(runners, "update_ton_progress_stats", lambda *_args, **_kwargs: True)

    runner._run_single(
        {
            "idx": 0,
            "ton_plans": [
                {"ton": 1, "con_dict": {"v_0_0": 1}, "save_exp": {"input_name": "case_0"}},
            ],
        }
    )

    assert len(calls) == 1


def test_queue_runner_does_not_fallback_for_existing_ternary_payload(monkeypatch, tmp_path: Path) -> None:
    save_dir = tmp_path / "case_ternary_timeout"
    _write_stats(save_dir / "stats.json")
    runner = runners.QueueRunner(timeout=5, norm=False, ternary_fallback=True)
    calls = []
    monkeypatch.setattr(
        runner,
        "_execute_attack",
        lambda payload: calls.append(dict(payload)) or _result_recorder(save_dir, is_timeout=True),
    )
    monkeypatch.setattr(runners, "update_ton_progress_stats", lambda *_args, **_kwargs: True)

    runner._run_single(
        {
            "idx": 0,
            "ternary_simplification": True,
            "ton_plans": [
                {"ton": 1, "con_dict": {"v_0_0": 1}, "save_exp": {"input_name": "case_0"}},
            ],
        }
    )

    assert len(calls) == 1


def test_write_ton_sequence_ignores_missing_stats_file(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(runners, "update_ton_progress_stats", lambda *_args, **_kwargs: calls.append(True))

    runners.BaseRunner._write_ton_sequence(
        SimpleNamespace(save_dir=str(tmp_path)),
        ton_sequence=[1, 2],
        current_ton=1,
    )

    assert calls == []


def test_random_assign_runner_stops_on_timeout(monkeypatch) -> None:
    attempts = []
    logs = []
    artifacts = []
    monotonic_values = iter([0.0, 0.1, 2.1, 2.2])

    monkeypatch.setattr(runners.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        runners,
        "run_random_assign_step",
        lambda payload, pixel_source, base_seed, attempt: attempts.append(attempt)
        or SimpleNamespace(success=False),
    )
    monkeypatch.setattr(runners, "write_combined_log", lambda result: logs.append(result.success))
    monkeypatch.setattr(runners, "write_experiment_artifacts", lambda result: artifacts.append(result))

    runner = runners.RandomAssignRunner(timeout=2, norm=False, pixel_source="random", base_seed=9)
    result = runner._run_random_assign_for_plan({"idx": 3})

    assert attempts == [0]
    assert logs == [False]
    assert artifacts == [result]
    assert result.success is False
    assert result.attack_wall_time == 2.2


def test_random_assign_runner_attaches_ton_sequence_to_last_result(monkeypatch) -> None:
    outcomes = [SimpleNamespace(success=False), SimpleNamespace(success=True)]
    runner = runners.RandomAssignRunner(timeout=2, norm=False, pixel_source="random", base_seed=9)
    monkeypatch.setattr(runner, "_run_random_assign_for_plan", lambda payload: outcomes.pop(0))

    result = runner._run_single(
        {
            "idx": 1,
            "ton_plans": [
                {"ton": 1, "con_dict": {"v_0_0": 1}, "save_exp": {"input_name": "case_1"}},
                {"ton": 2, "con_dict": {"v_0_1": 1}, "save_exp": {"input_name": "case_1"}},
            ],
        }
    )

    assert result.success is True
    assert result.ton_sequence == [1, 2]
