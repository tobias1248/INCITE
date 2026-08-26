from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

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

import libct.record as record_module
from libct.record import ConcolicTestRecorder


def test_recorder_rejects_non_finite_sat_and_adversarial_inputs() -> None:
    recorder = ConcolicTestRecorder(None, "case_0")
    recorder.input_shape = (1,)

    with pytest.raises(ValueError, match="NaN or Inf"):
        recorder.save_sat_input({"v_0": np.nan})
    with pytest.raises(ValueError, match="NaN or Inf"):
        recorder.find_adversarial_input({"v_0": np.inf}, attack_label=1)

    assert recorder.sat_inputs == []
    assert recorder.adversarial_input is None
    assert recorder.attack_label is None


def test_save_stats_dict_writes_original_jpg_without_adversarial(
    tmp_path: Path,
) -> None:
    if not hasattr(record_module.cv2, "imread"):
        pytest.skip("OpenCV image decoding is unavailable")
    save_dir = tmp_path / "case_original_only"
    recorder = ConcolicTestRecorder(str(save_dir), "case_original_only")
    recorder.input_shape = (8, 8, 1)
    recorder.original_label = 3
    recorder.original_input = np.full((8, 8, 1), 0.25, dtype=np.float32)

    recorder.save_stats_dict()

    original_npy = np.load(save_dir / "ori_input.npy")
    original_jpg = record_module.cv2.imread(
        str(save_dir / "ori_input.jpg"),
        record_module.cv2.IMREAD_GRAYSCALE,
    )
    assert original_npy.dtype == np.float32
    np.testing.assert_array_equal(original_npy, recorder.original_input)
    assert original_jpg.shape == (8, 8)
    assert np.max(np.abs(original_jpg.astype(int) - 63)) <= 2
    assert list(save_dir.glob("adv_*.jpg")) == []


def test_save_stats_dict_writes_rgb_ori_and_adv_with_correct_color_order(
    tmp_path: Path,
) -> None:
    if not hasattr(record_module.cv2, "imread"):
        pytest.skip("OpenCV image decoding is unavailable")
    save_dir = tmp_path / "case_rgb"
    recorder = ConcolicTestRecorder(str(save_dir), "case_rgb")
    recorder.input_shape = (8, 8, 3)
    recorder.original_label = 1
    recorder.attack_label = 2
    recorder.original_input = np.zeros((8, 8, 3), dtype=np.float32)
    recorder.original_input[..., 0] = 1.0
    recorder.adversarial_input = np.zeros((8, 8, 3), dtype=np.float32)
    recorder.adversarial_input[..., 2] = 1.0

    recorder.save_stats_dict()

    original_jpg = record_module.cv2.imread(str(save_dir / "ori_input.jpg"))
    adversarial_jpg = record_module.cv2.imread(
        str(save_dir / "adv_1_to_2.jpg")
    )
    assert original_jpg.shape == (8, 8, 3)
    assert adversarial_jpg.shape == (8, 8, 3)
    np.testing.assert_allclose(
        original_jpg.mean(axis=(0, 1)),
        [0, 0, 255],
        atol=3,
    )
    np.testing.assert_allclose(
        adversarial_jpg.mean(axis=(0, 1)),
        [255, 0, 0],
        atol=3,
    )


def test_output_stats_dict_reports_reference_prediction_timing() -> None:
    recorder = ConcolicTestRecorder(None, "case_0")
    recorder.record_reference_prediction(0.1, phase="original_reference")
    recorder.record_reference_prediction(0.2, phase="candidate_reference")

    stats = recorder.output_stats_dict()

    assert stats["summary"]["reference_prediction_count"] == 2
    assert stats["summary"]["reference_prediction_wall_time_total"] == pytest.approx(0.3)
    assert stats["summary"]["reference_prediction_phase_counts"] == {
        "original_reference": 1,
        "candidate_reference": 1,
    }


def test_save_stats_dict_preserves_invalid_model_diagnostic(tmp_path: Path) -> None:
    save_dir = tmp_path / "case_invalid_model"
    recorder = ConcolicTestRecorder(str(save_dir), "case_invalid_model")
    recorder.input_shape = (1,)
    detail = [
        {
            "iter": 1,
            "attempt_index": 1,
            "status": "invalid_model",
            "assert_num": 1,
            "byte": 20,
            "assert_len": [10],
            "formula_build_time_s": 0.1,
            "solver_subprocess_time_s": 0.2,
            "solve_total_time_s": 0.3,
            "total_time": 0.3,
            "model_error": "SMT Real decoded to a non-finite float",
            "smt_formula": "(check-sat)\n",
        }
    ]

    recorder.save_stats_dict(_constraint_complexity(detail))

    line = json.loads(
        (save_dir / "solver_iter1_top3.jsonl").read_text(encoding="utf-8").strip()
    )
    assert line["status"] == "invalid_model"
    assert line["model_error"] == "SMT Real decoded to a non-finite float"
    assert (save_dir / "solver_iter1_top3_smt" / "01_invalid_model.smt2").is_file()


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
            "solver_stdout": "sat\n((x_VAR (/ 1 2)))\n",
            "solver_stderr": "",
            "solver_returncode": 0,
            "solver_model_diagnostics": [
                {
                    "name": "x_VAR",
                    "value_type": "Real",
                    "raw_value": "(/ 1 2)",
                    "real": {
                        "is_rational": True,
                        "numerator_digits": 1,
                        "denominator_digits": 1,
                    },
                }
            ],
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
        {
            "iter": 2,
            "attempt_index": 2,
            "status": "invalid_model",
            "assert_num": 6,
            "byte": 606,
            "assert_len": [66],
            "formula_build_time_s": 12.0,
            "solver_subprocess_time_s": 13.0,
            "solve_total_time_s": 25.0,
            "total_time": 25.0,
            "model_error": "unsupported SMT Real atom",
            "smt_formula": "(assert f)\n(check-sat)\n",
            "solver_stdout": "sat\n((x_VAR NaN))\n",
            "solver_model_diagnostics": [
                {
                    "name": "x_VAR",
                    "value_type": "Real",
                    "raw_value": "NaN",
                    "real": {"parse_error": "unsupported SMT Real atom"},
                }
            ],
        },
    ]

    recorder.save_stats_dict(_constraint_complexity(detail))

    stats = json.loads((save_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["constraint_complexity"]["attempt_summary"]["all"]["formula_byte"]["sum"] == 2121
    assert "smt_formula" not in stats["constraint_complexity"]["entries"][0]
    assert "solver_stdout" not in stats["constraint_complexity"]["entries"][0]
    assert "solver_model_diagnostics" not in stats["constraint_complexity"]["entries"][0]

    jsonl_path = save_dir / "solver_iter1_top3.jsonl"
    assert jsonl_path.is_file()
    lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert [line["attempt_index"] for line in lines] == [1, 2, 4, 2]
    assert [line["status"] for line in lines] == ["sat", "unsat", "sat", "invalid_model"]
    assert [line["smt_path"] for line in lines] == [
        "solver_iter1_top3_smt/01_sat.smt2",
        "solver_iter1_top3_smt/02_unsat.smt2",
        "solver_iter1_top3_smt/03_sat.smt2",
        "solver_iter1_top3_smt/04_invalid_model.smt2",
    ]
    assert lines[0]["solver_stdout_path"] == "solver_iter1_top3_model/01_sat.stdout.txt"
    assert (
        lines[0]["solver_model_diagnostics_path"]
        == "solver_iter1_top3_model/01_sat.model.json"
    )

    smt_dir = save_dir / "solver_iter1_top3_smt"
    assert (smt_dir / "01_sat.smt2").read_text(encoding="utf-8") == "(assert a)\n(check-sat)\n"
    assert (smt_dir / "02_unsat.smt2").read_text(encoding="utf-8") == "(assert b)\n(check-sat)\n"
    assert (smt_dir / "03_sat.smt2").read_text(encoding="utf-8") == "(assert d)\n(check-sat)\n"
    assert (smt_dir / "04_invalid_model.smt2").read_text(encoding="utf-8") == "(assert f)\n(check-sat)\n"
    model_dir = save_dir / "solver_iter1_top3_model"
    assert (model_dir / "01_sat.stdout.txt").read_text(encoding="utf-8") == "sat\n((x_VAR (/ 1 2)))\n"
    assert (model_dir / "04_invalid_model.stdout.txt").read_text(encoding="utf-8") == "sat\n((x_VAR NaN))\n"
    model_diagnostics = json.loads((model_dir / "01_sat.model.json").read_text(encoding="utf-8"))
    assert model_diagnostics[0]["raw_value"] == "(/ 1 2)"
    invalid_diagnostics = json.loads(
        (model_dir / "04_invalid_model.model.json").read_text(encoding="utf-8")
    )
    assert invalid_diagnostics[0]["real"]["parse_error"] == "unsupported SMT Real atom"
