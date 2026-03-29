from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import orchestration.progress as progress


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
