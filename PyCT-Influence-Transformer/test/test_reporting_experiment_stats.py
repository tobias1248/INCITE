from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reporting import experiment_stats


def test_collect_files_returns_single_file_path(tmp_path) -> None:
    stats_path = tmp_path / "stats.json"
    stats_path.write_text("{}", encoding="utf-8")

    assert experiment_stats._collect_files(str(stats_path), "stats.json") == [str(stats_path)]


def test_main_returns_error_when_no_stats_match(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_stats", "--path", str(tmp_path)],
    )

    rc = experiment_stats.main()

    captured = capsys.readouterr()
    assert rc == 1
    assert "No files matched." in captured.out


def test_main_summarizes_single_stats_file(tmp_path, monkeypatch, capsys) -> None:
    case_dir = tmp_path / "exp" / "case_0"
    case_dir.mkdir(parents=True)
    payload = {
        "meta": {"status": "success", "is_timeout": False, "solve_all_ctr": False, "is_finish": True},
        "summary": {"total_wall_time": 1.25},
    }
    (case_dir / "stats.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_stats", "--path", str(tmp_path / "exp")],
    )

    rc = experiment_stats.main()

    captured = capsys.readouterr()
    assert rc == 0
    assert "Path:" in captured.out
    assert "Cases: total=1" in captured.out


def test_collect_files_recurses_for_stats_json(tmp_path) -> None:
    nested = tmp_path / "exp" / "case_0"
    nested.mkdir(parents=True)
    stats_path = nested / "stats.json"
    stats_path.write_text("{}", encoding="utf-8")

    assert experiment_stats._collect_files(str(tmp_path), "stats.json") == [str(stats_path)]


def test_main_reports_parse_errors_for_bad_json(tmp_path, monkeypatch, capsys) -> None:
    stats_path = tmp_path / "stats.json"
    stats_path.write_text("{bad json", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["experiment_stats", "--path", str(tmp_path)])

    rc = experiment_stats.main()

    captured = capsys.readouterr()
    assert rc == 0
    assert "parse_errors=1" in captured.out


def test_main_split_by_status_outputs_groups(tmp_path, monkeypatch, capsys) -> None:
    success_dir = tmp_path / "exp" / "case_success"
    timeout_dir = tmp_path / "exp" / "case_timeout"
    success_dir.mkdir(parents=True)
    timeout_dir.mkdir(parents=True)
    payloads = [
        (
            success_dir / "stats.json",
            {
                "meta": {"status": "success", "attack_label": "adv", "is_finish": True},
                "summary": {"total_wall_time": 1.0, "total_iter": 0},
                "constraint_complexity": {"entries": [{"status": "sat", "assert_num": 10, "byte": 20, "path_len": 1, "total_time": 0.1}]},
            },
        ),
        (
            timeout_dir / "stats.json",
            {
                "meta": {"status": "timeout", "is_timeout": True, "is_finish": False},
                "summary": {"total_wall_time": 2.0, "total_iter": 1},
                "constraint_complexity": {"entries": [{"status": "unsat", "assert_num": 20, "byte": 30, "path_len": 2, "total_time": 0.2}]},
            },
        ),
    ]
    for path, payload in payloads:
        path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_stats", "--path", str(tmp_path / "exp"), "--split-by-status"],
    )

    rc = experiment_stats.main()

    captured = capsys.readouterr()
    assert rc == 0
    assert "By status:" in captured.out
    assert "[success]" in captured.out
    assert "[timeout]" in captured.out


def test_main_include_history_outputs_history_section(tmp_path, monkeypatch, capsys) -> None:
    case_dir = tmp_path / "exp" / "case_0"
    case_dir.mkdir(parents=True)
    (case_dir / "stats.json").write_text(
        json.dumps(
            {
                "meta": {"status": "success", "attack_label": "adv", "is_finish": True},
                "summary": {"total_wall_time": 1.25},
                "constraint_complexity": {"entries": []},
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "stats_history.jsonl").write_text(
        json.dumps(
            {
                "meta": {"status": "success", "attack_label": "adv", "ton": 1},
                "summary": {"total_wall_time": 0.5},
                "ton_progress": {"reason": "continue"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_stats", "--path", str(tmp_path / "exp"), "--include-history"],
    )

    rc = experiment_stats.main()

    captured = capsys.readouterr()
    assert rc == 0
    assert "History (ton stages):" in captured.out
    assert "ton_counts: 1=1" in captured.out
    assert "reason_counts: continue=1" in captured.out


def test_main_summarizes_multiple_cases_with_percentiles(tmp_path, monkeypatch, capsys) -> None:
    for idx, wall_time in enumerate((1.0, 5.0), start=1):
        case_dir = tmp_path / "exp" / f"case_{idx}"
        case_dir.mkdir(parents=True)
        (case_dir / "stats.json").write_text(
            json.dumps(
                {
                    "meta": {"status": "success", "attack_label": f"adv-{idx}", "is_finish": True},
                    "summary": {"total_wall_time": wall_time, "total_iter": idx},
                    "constraint_complexity": {
                        "entries": [
                            {"status": "sat", "assert_num": 10 * idx, "byte": 20 * idx, "path_len": idx, "total_time": 0.1 * idx}
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(sys, "argv", ["experiment_stats", "--path", str(tmp_path / "exp")])

    rc = experiment_stats.main()

    captured = capsys.readouterr()
    assert rc == 0
    assert "Cases: total=2" in captured.out
    assert "n=2" in captured.out
