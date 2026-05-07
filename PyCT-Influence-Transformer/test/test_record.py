from __future__ import annotations

import json
from pathlib import Path
import sys

try:
    import cv2 as _cv2  # type: ignore
except ModuleNotFoundError:
    class _StubCV2:
        @staticmethod
        def imwrite(_path, _img):
            return True

    sys.modules["cv2"] = _StubCV2()
else:
    del _cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libct.record import ConcolicTestRecorder


def _constraint_complexity(detail):
    return {
        "type": [entry.get("status") for entry in detail],
        "time": [
            entry.get("solver_subprocess_time_s")
            for entry in detail
            if isinstance(entry.get("solver_subprocess_time_s"), (int, float))
        ],
        "byte": [
            entry.get("byte")
            for entry in detail
            if isinstance(entry.get("byte"), (int, float))
        ],
        "assert_num": [
            entry.get("assert_num")
            for entry in detail
            if isinstance(entry.get("assert_num"), (int, float))
        ],
        "assert_len": [entry.get("assert_len", []) for entry in detail],
        "path_len": [
            entry.get("path_len")
            for entry in detail
            if isinstance(entry.get("path_len"), (int, float))
        ],
        "build_time": [
            entry.get("formula_build_time_s")
            for entry in detail
            if isinstance(entry.get("formula_build_time_s"), (int, float))
        ],
        "total_time": [
            entry.get("solve_total_time_s")
            for entry in detail
            if isinstance(entry.get("solve_total_time_s"), (int, float))
        ],
        "detail": detail,
    }


def test_output_stats_dict_builds_attempt_summary_and_strips_formula() -> None:
    recorder = ConcolicTestRecorder(None, "case_0")
    detail = [
        {
            "iter": 1,
            "attempt_index": 1,
            "status": "sat",
            "path_len": 1,
            "assert_num": 2,
            "byte": 100,
            "assert_len": [10, 20],
            "formula_build_time_s": 1.0,
            "solver_subprocess_time_s": 4.0,
            "solve_total_time_s": 5.0,
            "total_time": 5.0,
            "smt_formula": "(assert true)\n(check-sat)\n",
        },
        {
            "iter": 1,
            "attempt_index": 2,
            "status": "unsat",
            "path_len": 2,
            "assert_num": 4,
            "byte": 200,
            "assert_len": [30, 40, 50, 60],
            "formula_build_time_s": 2.0,
            "solver_subprocess_time_s": 8.0,
            "solve_total_time_s": 10.0,
            "total_time": 10.0,
            "smt_formula": "(assert false)\n(check-sat)\n",
        },
        {
            "iter": 1,
            "attempt_index": 3,
            "status": "timeout",
            "path_len": 3,
            "assert_num": 6,
            "byte": 300,
            "assert_len": [70],
            "formula_build_time_s": 3.0,
            "solver_subprocess_time_s": 12.0,
            "solve_total_time_s": 15.0,
            "total_time": 15.0,
            "smt_formula": "(assert maybe)\n(check-sat)\n",
        },
    ]

    stats = recorder.output_stats_dict(_constraint_complexity(detail))

    attempt_summary = stats["constraint_complexity"]["attempt_summary"]
    assert attempt_summary["all"]["formula_build_time_s"]["mean"] == 2.0
    assert attempt_summary["all"]["formula_byte"]["sum"] == 600
    assert attempt_summary["all"]["assert_len"]["sum"] == 280
    assert attempt_summary["sat"]["solve_total_time_s"]["sum"] == 5.0
    assert attempt_summary["unsat"]["assert_count"]["mean"] == 4.0
    assert "smt_formula" not in stats["constraint_complexity"]["entries"][0]


def test_save_stats_dict_writes_solver_iter1_top3_artifacts(tmp_path: Path) -> None:
    save_dir = tmp_path / "case_0"
    recorder = ConcolicTestRecorder(str(save_dir), "case_0")
    recorder.input_shape = (1,)

    detail = [
        {
            "iter": 1,
            "attempt_index": 1,
            "status": "sat",
            "assert_num": 1,
            "byte": 101,
            "assert_len": [11],
            "formula_build_time_s": 1.0,
            "solver_subprocess_time_s": 2.0,
            "solve_total_time_s": 3.0,
            "total_time": 3.0,
            "smt_formula": "(assert a)\n(check-sat)\n",
        },
        {
            "iter": 1,
            "attempt_index": 2,
            "status": "unsat",
            "assert_num": 2,
            "byte": 202,
            "assert_len": [22, 23],
            "formula_build_time_s": 4.0,
            "solver_subprocess_time_s": 5.0,
            "solve_total_time_s": 9.0,
            "total_time": 9.0,
            "smt_formula": "(assert b)\n(check-sat)\n",
        },
        {
            "iter": 1,
            "attempt_index": 3,
            "status": "timeout",
            "assert_num": 3,
            "byte": 303,
            "assert_len": [33],
            "formula_build_time_s": 6.0,
            "solver_subprocess_time_s": 7.0,
            "solve_total_time_s": 13.0,
            "total_time": 13.0,
            "smt_formula": "(assert c)\n(check-sat)\n",
        },
        {
            "iter": 1,
            "attempt_index": 4,
            "status": "sat",
            "assert_num": 4,
            "byte": 404,
            "assert_len": [44, 45],
            "formula_build_time_s": 8.0,
            "solver_subprocess_time_s": 9.0,
            "solve_total_time_s": 17.0,
            "total_time": 17.0,
            "smt_formula": "(assert d)\n(check-sat)\n",
        },
        {
            "iter": 2,
            "attempt_index": 1,
            "status": "sat",
            "assert_num": 5,
            "byte": 505,
            "assert_len": [55],
            "formula_build_time_s": 10.0,
            "solver_subprocess_time_s": 11.0,
            "solve_total_time_s": 21.0,
            "total_time": 21.0,
            "smt_formula": "(assert e)\n(check-sat)\n",
        },
    ]

    recorder.save_stats_dict(_constraint_complexity(detail))

    stats = json.loads((save_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["constraint_complexity"]["attempt_summary"]["all"]["formula_byte"]["sum"] == 1515
    assert "smt_formula" not in stats["constraint_complexity"]["entries"][0]

    jsonl_path = save_dir / "solver_iter1_top3.jsonl"
    assert jsonl_path.is_file()
    lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert [line["attempt_index"] for line in lines] == [1, 2, 4]
    assert [line["status"] for line in lines] == ["sat", "unsat", "sat"]
    assert [line["smt_path"] for line in lines] == [
        "solver_iter1_top3_smt/01_sat.smt2",
        "solver_iter1_top3_smt/02_unsat.smt2",
        "solver_iter1_top3_smt/03_sat.smt2",
    ]

    smt_dir = save_dir / "solver_iter1_top3_smt"
    assert (smt_dir / "01_sat.smt2").read_text(encoding="utf-8") == "(assert a)\n(check-sat)\n"
    assert (smt_dir / "02_unsat.smt2").read_text(encoding="utf-8") == "(assert b)\n(check-sat)\n"
    assert (smt_dir / "03_sat.smt2").read_text(encoding="utf-8") == "(assert d)\n(check-sat)\n"
