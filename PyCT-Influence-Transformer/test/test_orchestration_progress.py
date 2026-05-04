from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import orchestration.progress as progress
from libct.record import ConcolicTestRecorder


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_derive_ton_outcome_covers_all_main_statuses() -> None:
    assert progress.derive_ton_outcome(type("R", (), {"attack_label": "adv"})()) == (False, "adv_found")
    assert progress.derive_ton_outcome(type("R", (), {"attack_label": None, "solve_all_ctr": True})()) == (
        True,
        "solve_all_ctr",
    )
    assert progress.derive_ton_outcome(type("R", (), {"attack_label": None, "solve_all_ctr": False, "is_timeout": True})()) == (
        True,
        "timeout",
    )
    assert progress.derive_ton_outcome(type("R", (), {"attack_label": None, "solve_all_ctr": False, "is_timeout": False})()) == (
        False,
        "incomplete",
    )
    assert progress.derive_ton_outcome(
        type(
            "R",
            (),
            {"attack_label": None, "solve_all_ctr": False, "is_timeout": False, "extra_meta": {"status": "error", "error_type": "constraint_transfer_failure"}},
        )()
    ) == (False, "error_constraint_transfer_failure")


def test_load_stats_payload_reports_missing_and_invalid(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing" / "stats.json"
    invalid_path = tmp_path / "invalid" / "stats.json"
    invalid_path.parent.mkdir(parents=True)
    invalid_path.write_text("{bad json", encoding="utf-8")

    assert progress.load_stats_payload(missing_path) == (None, "missing_stats")
    assert progress.load_stats_payload(invalid_path) == (None, "invalid_stats")


def test_coerce_int_accepts_digit_strings_only() -> None:
    assert progress.coerce_int(3) == 3
    assert progress.coerce_int("12") == 12
    assert progress.coerce_int("01") == 1
    assert progress.coerce_int("-1") is None
    assert progress.coerce_int("abc") is None


def test_extract_last_ton_uses_meta_then_legacy_progress_then_ton_progress() -> None:
    assert progress.extract_last_ton({"meta": {"ton": "2"}}) == 2
    assert progress.extract_last_ton({"meta": {"progress": {"ton_current": "3"}}}) == 3
    assert progress.extract_last_ton({"meta": {"ton_progress": {"current": "4"}}}) == 4
    assert progress.extract_last_ton({}) is None


def test_stats_indicate_completion_recognizes_success_finish_and_timeout() -> None:
    assert progress.stats_indicate_completion({"meta": {"attack_label": "adv"}}) is True
    assert progress.stats_indicate_completion({"meta": {"is_finish": True}}) is True
    assert progress.stats_indicate_completion({"meta": {"is_timeout": True}}) is True
    assert progress.stats_indicate_completion({"meta": {}}) is False


def test_error_stats_are_terminal_and_keep_error_reason() -> None:
    recorder = ConcolicTestRecorder(None, "case_0")
    recorder.mark_error("constraint_transfer_failure", "could not transfer constraints")

    stats = recorder.output_stats_dict()

    assert stats["meta"]["status"] == "error"
    assert stats["meta"]["error_type"] == "constraint_transfer_failure"
    assert stats["meta"]["error_reason"] == "could not transfer constraints"
    assert stats["meta"]["attack_label"] is None
    assert stats["meta"]["is_finish"] is False
    assert stats["meta"]["is_timeout"] is False
    assert stats["meta"]["solve_all_ctr"] is False
    assert progress.stats_indicate_completion(stats) is True
    assert progress.derive_stage_outcome_payload(stats) == (False, "error_constraint_transfer_failure")


def test_update_ton_progress_stats_rewrites_legacy_keys(tmp_path: Path) -> None:
    stats_path = tmp_path / "case_0" / "stats.json"
    _write_json(
        stats_path,
        {
            "progress": {"ton_current": 1},
            "ton_progress": {"current": 1},
            "ton_sequence": [1, 2],
            "meta": {
                "progress": {"ton_current": 1},
                "ton_sequence": [1, 2],
                "finished": True,
            },
        },
    )

    ok = progress.update_ton_progress_stats(
        stats_path,
        current_ton=1,
        status="continue",
        reason="timeout",
        next_ton=2,
    )

    assert ok is True
    written = json.loads(stats_path.read_text(encoding="utf-8"))
    assert written["meta"]["ton_progress"] == {
        "current": 1,
        "next": 2,
        "stop_at": None,
        "status": "continue",
        "reason": "timeout",
    }
    assert "progress" not in written
    assert "ton_progress" not in written
    assert "ton_sequence" not in written
    assert "progress" not in written["meta"]
    assert "ton_sequence" not in written["meta"]
    assert "finished" not in written["meta"]

    history_path = stats_path.with_name("stats_history.jsonl")
    assert history_path.is_file()
    history_lines = history_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(history_lines) == 1
    assert json.loads(history_lines[0])["meta"]["ton_progress"]["next"] == 2


def test_should_run_ton_advances_only_after_continue_state(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_1"
    _write_json(
        case_dir / "stats.json",
        {
            "meta": {
                "ton_progress": {"current": 1},
                "solve_all_ctr": True,
            }
        },
    )
    case = {"save_dir": str(case_dir)}

    assert progress.should_run_ton(case, 2, (1, 2), force_refresh=False) is True
    assert progress.should_run_ton(case, 1, (1, 2), force_refresh=False) is False


def test_should_run_ton_handles_missing_stats_and_force_refresh(tmp_path: Path) -> None:
    case = {"save_dir": str(tmp_path / "case_missing")}

    assert progress.should_run_ton(case, 1, (1, 2), force_refresh=False) is True
    assert progress.should_run_ton(case, 2, (1, 2), force_refresh=False) is False
    assert progress.should_run_ton(case, 2, (1, 2), force_refresh=True) is True


def test_should_run_ton_stops_after_success_or_future_ton(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_done"
    _write_json(case_dir / "stats.json", {"meta": {"attack_label": "adv", "ton": 1}})
    case = {"save_dir": str(case_dir)}

    assert progress.should_run_ton(case, 2, (1, 2), force_refresh=False) is False

    _write_json(case_dir / "stats.json", {"meta": {"ton": 3}})
    assert progress.should_run_ton(case, 2, (1, 2, 3), force_refresh=False) is False


def test_should_run_ton_retries_same_ton_when_incomplete(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_retry"
    _write_json(case_dir / "stats.json", {"meta": {"ton_progress": {"current": 2}}})
    case = {"save_dir": str(case_dir)}

    assert progress.should_run_ton(case, 2, (1, 2, 3), force_refresh=False) is True


def test_should_run_ton_stops_after_terminal_error(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_error"
    _write_json(
        case_dir / "stats.json",
        {
            "meta": {
                "ton_progress": {"current": 2},
                "status": "error",
                "error_type": "constraint_transfer_failure",
            }
        },
    )
    case = {"save_dir": str(case_dir)}

    assert progress.should_run_ton(case, 2, (1, 2, 3), force_refresh=False) is False
    assert progress.should_run_ton(case, 3, (1, 2, 3), force_refresh=False) is False


def test_should_run_payload_skips_finished_stats(monkeypatch, tmp_path: Path) -> None:
    case_dir = tmp_path / "exp_case"
    _write_json(case_dir / "stats.json", {"meta": {"attack_label": "adv"}})
    monkeypatch.setattr(progress, "get_save_dir_from_save_exp", lambda *args, **kwargs: str(case_dir))

    should_run = progress.should_run_payload(
        {
            "model_name": "demo",
            "save_exp": {"input_name": "case_0", "attack_mode": "queue"},
        },
        force_refresh=False,
    )

    assert should_run is False


def test_should_run_payload_runs_when_stats_missing_or_force_refresh(monkeypatch, tmp_path: Path) -> None:
    case_dir = tmp_path / "exp_case_missing"
    monkeypatch.setattr(progress, "get_save_dir_from_save_exp", lambda *args, **kwargs: str(case_dir))

    should_run_missing = progress.should_run_payload(
        {
            "model_name": "demo",
            "save_exp": {"input_name": "case_1", "attack_mode": "queue"},
        },
        force_refresh=False,
    )
    should_run_forced = progress.should_run_payload(
        {
            "model_name": "demo",
            "save_exp": {"input_name": "case_1", "attack_mode": "queue"},
        },
        force_refresh=True,
    )

    assert should_run_missing is True
    assert should_run_forced is True


def test_collect_stage_cases_groups_ton_plans(monkeypatch, tmp_path: Path) -> None:
    expected_dir = tmp_path / "save_dir"
    monkeypatch.setattr(
        progress,
        "get_save_dir_from_save_exp",
        lambda *args, **kwargs: str(expected_dir),
    )

    cases = progress.collect_stage_cases(
        [
            {
                "model_name": "demo",
                "idx": 7,
                "in_dict": {"v_0_0": 1.0},
                "ton_plans": [
                    {"ton": 1, "con_dict": {"v_0_0": 1}, "save_exp": {"input_name": "case_7"}},
                    {"ton": 2, "con_dict": {"v_0_1": 1}, "save_exp": {"input_name": "case_7"}},
                ],
            }
        ]
    )

    assert len(cases) == 1
    case = cases[0]
    assert case["idx"] == 7
    assert case["input_name"] == "case_7"
    assert case["save_dir"] == str(expected_dir)
    assert case["plans"][1]["con_dict"] == {"v_0_0": 1}
    assert "ton_plans" not in case["base_payload"]
