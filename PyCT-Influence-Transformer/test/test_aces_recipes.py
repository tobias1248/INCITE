from __future__ import annotations

import copy
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from libct.aces_like_multi import parameter_name
from orchestration import aces_recipes as recipes


CONTROLS = ("exposure", "contrast", "saturation")
PAIRS_THEN_TRIPLE = [
    ("exposure", "contrast"), ("exposure", "saturation"),
    ("contrast", "saturation"), CONTROLS,
]


def _payload():
    return {
        "model_name": "offline", "idx": 3, "score_alpha": 0.0,
        "popped_log_attack_mode": "global-real",
        "save_exp": {"input_name": "case_3", "idx": 3},
        "in_dict": {"v_0_0_0": 0.2, "v_0_0_1": 0.5, "v_0_0_2": 0.8,
                    **{parameter_name(c): 0.0 for c in CONTROLS}},
        "con_dict": {parameter_name(c): 1 for c in CONTROLS},
        "input_for_shap": np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float32),
        "background_dataset_for_shap": np.zeros((2, 1, 1, 3), dtype=np.float32),
        "shap_output_root": "cached-target-shap", "shap_value_pre_calculated": True,
        "global_real_config": {
            "global_shift_kind": "aces-multi", "shap_priority_enabled": True,
            "shap_sensitivity_enabled": False, "probe_enabled": False,
            "controls": list(CONTROLS), "control_bounds": [[-0.1, 0.1]] * 3,
        },
        "aces_recipe_plan": {
            "policy": "staged", "controls": ["saturation", "exposure", "contrast"],
            "control_bounds": [[-0.03, 0.04], [-0.01, 0.02], [-0.02, 0.03]],
            "region_budgets": [8, 16, 32], "pwl_build_timeout_seconds": 5.0,
        },
    }


@pytest.fixture
def case_clock(monkeypatch, tmp_path):
    clock = [100.0]
    monkeypatch.setattr(recipes.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(recipes, "get_save_dir_from_save_exp", lambda *args, **kwargs: str(tmp_path))
    return clock


def _summary(tmp_path):
    return json.loads((tmp_path / "recipe_summary.json").read_text())


def test_recipe_sequence_uses_canonical_pairs_then_triple():
    assert recipes.recipe_sequence(["saturation", "exposure", "contrast"]) == PAIRS_THEN_TRIPLE
    assert recipes.recipe_sequence(["curves", "exposure"]) == [("exposure", "curves")]


def test_shared_deadline_redistributes_unused_time_and_preserves_original_inputs(
    monkeypatch, tmp_path, case_clock,
):
    original = _payload()
    before = copy.deepcopy(original)
    seen, deadlines, progress = [], [], []
    durations = iter([2, 10, 4, 1])

    def attempt(payload, options, deadline, build_seconds, update):
        seen.append(copy.deepcopy(payload))
        deadlines.append(deadline)
        assert options == {"solver_run_timeout": 3}
        assert build_seconds == 5.0
        update({"pwl": {"pwl_region_count": 6}})
        progress.append(_summary(tmp_path)["attempts"][-1]["pwl"])
        case_clock[0] += next(durations)
        # Any search mutation must remain local to this attempt.
        payload["in_dict"]["v_0_0_0"] = 0.99
        payload["in_dict"].update({parameter_name(c): 0.07 for c in CONTROLS})
        payload["input_for_shap"].fill(9)
        payload["background_dataset_for_shap"].fill(9)
        payload["global_real_config"]["controls"].clear()
        return {"status": "exhausted", "reason": "queue empty", "original_label": 2}

    monkeypatch.setattr(recipes, "_run_isolated", attempt)
    _, recorder = recipes.run_recipes(original, timeout=120, solver_run_timeout=3)
    summary = _summary(tmp_path)
    assert [a["allocated_seconds"] for a in summary["attempts"]] == pytest.approx([30, 118 / 3, 54, 104])
    assert deadlines == pytest.approx([130, 102 + 118 / 3, 166, 220])
    assert summary["total_wall_time"] == 17
    assert summary["schema_version"] == "aces-recipes-v1"
    assert summary["status"] == recorder.extra_meta["status"] == "exhausted"
    assert recorder.solve_all_ctr is True
    assert summary["original_label"] == 2
    assert progress == [{"pwl_region_count": 6}] * 4
    expected_bounds = {"exposure": [-0.01, 0.02], "contrast": [-0.02, 0.03],
                       "saturation": [-0.03, 0.04]}
    for stage, selected in zip(seen, PAIRS_THEN_TRIPLE):
        assert tuple(stage["global_real_config"]["controls"]) == selected
        assert stage["global_real_config"]["control_bounds"] == [expected_bounds[c] for c in selected]
        assert stage["con_dict"] == {parameter_name(c): 1 for c in selected}
        assert {k: v for k, v in stage["in_dict"].items() if k.startswith("__pyct_")} == {
            parameter_name(c): 0.0 for c in selected}
        assert stage["in_dict"]["v_0_0_0"] == 0.2
        np.testing.assert_array_equal(stage["input_for_shap"], before["input_for_shap"])
        np.testing.assert_array_equal(stage["background_dataset_for_shap"], before["background_dataset_for_shap"])
        assert stage["shap_output_root"] == "cached-target-shap"
        assert stage["shap_value_pre_calculated"] is True
    assert original["in_dict"] == before["in_dict"]
    assert original["global_real_config"] == before["global_real_config"]
    np.testing.assert_array_equal(original["input_for_shap"], before["input_for_shap"])
    np.testing.assert_array_equal(original["background_dataset_for_shap"], before["background_dataset_for_shap"])


@pytest.mark.parametrize(("statuses", "durations", "expected", "count"), [
    (["pwl_budget_exhausted"] * 4, [1] * 4, "invalid", 4),
    (["pwl_build_timeout"] * 4, [1] * 4, "invalid", 4),
    (["pwl_build_timeout"] * 4, [30] * 4, "invalid", 4),
    (["pwl_budget_exhausted", "exhausted", "exhausted", "exhausted"], [1] * 4, "incomplete", 4),
    (["pwl_build_timeout", "exhausted", "exhausted", "exhausted"], [1] * 4, "incomplete", 4),
    (["exhausted", "incomplete", "exhausted", "exhausted"], [1] * 4, "incomplete", 4),
    (["timeout", "exhausted", "exhausted", "exhausted"], [1] * 4, "timeout", 4),
    (["exhausted"] * 4, [30, 30, 30, 30], "timeout", 4),
    (["exhausted", "success"], [1, 1], "success", 2),
    (["error"], [1], "error", 1),
])
def test_terminal_outcomes_and_early_stop(
    monkeypatch, tmp_path, case_clock, statuses, durations, expected, count,
):
    outcomes = iter(zip(statuses, durations))

    def attempt(payload, options, deadline, build_seconds, update):
        status, seconds = next(outcomes)
        case_clock[0] += seconds
        return {"status": status, "reason": "test result", "original_label": 2,
                "attack_label": 7 if status == "success" else None}

    monkeypatch.setattr(recipes, "_run_isolated", attempt)
    _, recorder = recipes.run_recipes(_payload(), timeout=120)
    summary = _summary(tmp_path)
    assert summary["status"] == recorder.extra_meta["status"] == expected
    assert len(summary["attempts"]) == count
    assert recorder.is_timeout is (expected == "timeout")
    assert summary["winning_recipe"] == (1 if expected == "success" else None)
    assert summary["attack_label"] == (7 if expected == "success" else None)
    assert json.loads((tmp_path / "stats.json").read_text())["meta"]["status"] == expected


def test_unexpected_parent_failure_persists_error_then_propagates(monkeypatch, tmp_path, case_clock):
    def fail(*args):
        raise RuntimeError("unexpected child supervisor failure")

    monkeypatch.setattr(recipes, "_run_isolated", fail)
    with pytest.raises(RuntimeError, match="supervisor failure"):
        recipes.run_recipes(_payload(), timeout=120)
    assert _summary(tmp_path)["status"] == "error"
    meta = json.loads((tmp_path / "stats.json").read_text())["meta"]
    assert meta["status"] == "error"
    assert "supervisor failure" in meta["error_reason"]


def test_truncated_stats_from_timed_out_stage_preserves_timeout_and_continues(
    monkeypatch, tmp_path, case_clock,
):
    calls = []

    def attempt(payload, options, deadline, build_seconds, update):
        calls.append(payload)
        case_clock[0] += 1
        if len(calls) == 1:
            directory = Path(payload["save_exp"]["input_name"])
            directory.mkdir(parents=True)
            (directory / "stats.json").write_text('{"meta":')
            return {"status": "timeout", "reason": "search deadline exceeded"}
        return {"status": "exhausted", "reason": "queue empty"}

    monkeypatch.setattr(recipes, "_run_isolated", attempt)
    _, recorder = recipes.run_recipes(_payload(), timeout=120)
    summary = _summary(tmp_path)
    assert len(calls) == len(summary["attempts"]) == 4
    assert summary["status"] == recorder.extra_meta["status"] == "timeout"
    first = summary["attempts"][0]
    assert first["status"] == "timeout"
    assert first["reason"] == "search deadline exceeded"
    assert first["stats_path"] is None
    assert first["stats_unavailable_reason"]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_runtime_rejects_invalid_case_timeout(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        recipes.run_recipes(_payload(), timeout=timeout)


@pytest.fixture
def child_harness(monkeypatch, tmp_path):
    import engine.executor as executor
    import libct.global_real as global_real

    # Exercise the child body in-process without changing the test session.
    monkeypatch.setattr(recipes.os, "setsid", lambda: None)
    monkeypatch.setattr(recipes.signal, "signal", lambda *args: None)
    clock = [100.0]
    monkeypatch.setattr(recipes.time, "monotonic", lambda: clock[0])
    messages = []
    pipe = SimpleNamespace(send=messages.append, closed=False)
    pipe.close = lambda: setattr(pipe, "closed", True)
    payload = _payload()
    payload.pop("aces_recipe_plan")
    payload["save_exp"]["input_name"] = str(tmp_path)
    return SimpleNamespace(executor=executor, global_real=global_real,
                           clock=clock, pipe=pipe, messages=messages, payload=payload,
                           directory=tmp_path)


@pytest.mark.parametrize(("build_seconds", "search_seconds"), [(5.25, 4), (9.25, None)])
def test_attempt_child_rounds_remaining_search_time_down_and_skips_subsecond_search(
    monkeypatch, child_harness, build_seconds, search_seconds,
):
    harness = child_harness
    builds, searches = [], []

    def build(rgb, config):
        builds.append((rgb.copy(), dict(config)))
        harness.clock[0] += build_seconds
        return dict(config, pwl_region_count=12)

    def search(**kwargs):
        searches.append(kwargs)
        for name in ("timeout", "total_timeout", "single_timeout"):
            assert type(kwargs[name]) is int
            assert kwargs[name] == search_seconds
            assert 0 < kwargs[name] <= 110.0 - harness.clock[0]
        (harness.directory / "stats.json").write_text(json.dumps({
            "meta": {"status": "exhausted", "original_label": 2},
        }))

    monkeypatch.setattr(harness.global_real, "build_aces_like_global_real_config", build)
    monkeypatch.setattr(harness.executor, "run", search)
    recipes._attempt_child(harness.pipe, harness.payload, {"solver_run_timeout": 1}, 110.0, 20.0)
    assert harness.pipe.closed
    assert len(builds) == 1
    assert builds[0][1]["pwl_build_deadline"] == 110.0
    np.testing.assert_array_equal(builds[0][0], [[[0.2, 0.5, 0.8]]])
    assert len(searches) == (1 if search_seconds else 0)
    assert harness.messages[-1]["status"] == ("exhausted" if search_seconds else "timeout")
    assert [message["event"] for message in harness.messages] == (
        ["building", "searching", "finished"] if search_seconds else ["building", "finished"]
    )
    if search_seconds is None:
        assert harness.messages[-1]["timeout_phase"] == "integer_search_budget"
        assert harness.messages[-1]["search_seconds_available"] == 0.75


@pytest.mark.parametrize("failure_phase", ["build", "search"])
def test_attempt_child_reports_unexpected_errors_and_closes_pipe(monkeypatch, child_harness, failure_phase):
    harness = child_harness

    def fail(*args, **kwargs):
        raise RuntimeError("unexpected {} failure".format(failure_phase))

    monkeypatch.setattr(harness.global_real, "build_aces_like_global_real_config",
                        fail if failure_phase == "build" else lambda rgb, config: config)
    monkeypatch.setattr(harness.executor, "run", fail)
    recipes._attempt_child(harness.pipe, harness.payload, {}, 110.0, 5.0)
    assert harness.pipe.closed
    assert {key: value for key, value in harness.messages[-1].items() if key != "sent_at"} == {
        "event": "finished", "status": "error",
        "reason": "RuntimeError: unexpected {} failure".format(failure_phase),
    }


@pytest.mark.parametrize("session_ready", [False, True])
def test_cleanup_kills_session_even_when_leader_exited_before_handshake(monkeypatch, session_ready):
    signals, joins = [], []
    process = SimpleNamespace(pid=12345, is_alive=lambda: False,
                              join=lambda timeout: joins.append(timeout))
    monkeypatch.setattr(recipes.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    recipes._stop_attempt(process, session_ready=session_ready)
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert joins == [0.2, 0.2]


def _controlled_attempt(connection, payload, options, deadline, build_seconds):
    """Fork-only fixture: leave a live descendant for the supervisor to stop."""
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], close_fds=True)
    Path(payload["pid_file"]).write_text(json.dumps([os.getpid(), child.pid]))
    connection.send({"event": "building", "deadline": min(deadline, time.monotonic() + build_seconds),
                     "sent_at": time.monotonic()})
    if payload["mode"] == "search":
        connection.send({"event": "searching", "pwl": {"pwl_region_count": 6},
                         "sent_at": time.monotonic()})
    if payload["mode"] == "exit":
        connection.close()
        return
    time.sleep(30)


def _is_running(pid):
    try:
        # An orphan reaped asynchronously by init may briefly remain a zombie.
        return Path("/proc/{}/stat".format(pid)).read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _timed_events_attempt(connection, payload, options, deadline, build_seconds):
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    now = time.monotonic()
    connection.send({"event": "building", "deadline": now + 0.1, "sent_at": now})
    if payload["mode"] == "late_search":
        time.sleep(0.15)
    connection.send({"event": "searching", "sent_at": time.monotonic(), "pwl": {"ready": True}})
    if payload["mode"] == "late_success":
        time.sleep(max(0, deadline - time.monotonic()) + 0.05)
        connection.send({"event": "finished", "sent_at": time.monotonic(), "status": "success"})
    time.sleep(30)


@pytest.mark.skipif(sys.platform != "linux" or "fork" not in multiprocessing.get_all_start_methods(),
                    reason="Recipe event deadlines use Linux fork")
@pytest.mark.parametrize(("mode", "expected"), [
    ("timely_search", "timeout"), ("late_search", "pwl_build_timeout"), ("late_success", "timeout"),
])
def test_supervisor_uses_child_event_time_despite_parent_progress_delay(monkeypatch, mode, expected):
    monkeypatch.setattr(recipes, "_attempt_child", _timed_events_attempt)
    updates = []

    def update(progress):
        updates.append(progress)
        if len(updates) == 1:
            # Simulate a slow summary write while later events queue in the pipe.
            time.sleep(0.7 if mode == "late_success" else 0.2)

    result = recipes._run_isolated({"mode": mode}, {}, time.monotonic() + 0.5, 0.1, update)
    assert result["status"] == expected
    assert all("sent_at" not in progress for progress in updates)
    if mode == "timely_search":
        assert {"pwl": {"ready": True}} in updates
    assert not any(progress.get("status") == "success" for progress in updates)


@pytest.mark.skipif(sys.platform != "linux" or "fork" not in multiprocessing.get_all_start_methods(),
                    reason="Recipe session cleanup uses Linux fork and /proc")
@pytest.mark.parametrize(("mode", "expected"), [("build", "pwl_build_timeout"),
                                                  ("search", "timeout"), ("exit", "error")])
def test_real_supervisor_stops_process_and_descendants(monkeypatch, tmp_path, mode, expected):
    monkeypatch.setattr(recipes, "_attempt_child", _controlled_attempt)
    pid_file = tmp_path / "pids.json"
    events = []
    started = time.monotonic()
    pids = []
    try:
        result = recipes._run_isolated({"pid_file": str(pid_file), "mode": mode}, {},
                                      started + 1.0, 0.2, events.append)
        assert result["status"] == expected
        assert time.monotonic() - started < 5
        pids = json.loads(pid_file.read_text())
        until = time.monotonic() + 2
        while any(_is_running(pid) for pid in pids) and time.monotonic() < until:
            time.sleep(0.01)
        assert not any(_is_running(pid) for pid in pids)
        if mode == "search":
            assert {"pwl": {"pwl_region_count": 6}} in events
        elif mode == "exit":
            assert "without a result" in result["reason"]
    finally:
        if not pids and pid_file.exists():
            pids = json.loads(pid_file.read_text())
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
