from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from libct import solver


def _engine() -> SimpleNamespace:
    return SimpleNamespace(
        concolic_flag_dict={"x": 1},
        shap_value_pre_calculated=True,
        model_path="model/mock.h5",
        var_to_types={"x_VAR": "Real"},
        popped_log_attack_mode="shap",
        concolic_name_list=["x_VAR"],
    )


def _configure_solver() -> None:
    solver.Solver.stats = {
        "sat_number": 0,
        "sat_time": 0,
        "unsat_number": 0,
        "unsat_time": 0,
        "otherwise_number": 0,
        "otherwise_time": 0,
        "invalid_model_number": 0,
    }
    solver.Solver.smtdir = None
    solver.Solver.store = None
    solver.Solver.iter = 1
    solver.Solver.iter_count = 1
    solver.Solver.cnt = 1
    solver.Solver.cmd = ["cvc5"]
    solver.Solver.limit_change_range = None
    solver.Solver.norm = True
    solver.Solver.build_timeout_enabled = True
    solver.Solver.build_timeout_seconds = 30
    solver.Solver.run_timeout = None
    solver.Solver.ctr_size = {
        "type": [],
        "time": [],
        "byte": [],
        "assert_num": [],
        "assert_len": [],
        "path_len": [],
        "build_time": [],
        "total_time": [],
        "detail": [],
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.5", 0.5),
        ("(/ 1 4)", 0.25),
        ("(- (/ 1 4))", -0.25),
        ("(/ (- 1) 4)", -0.25),
    ],
)
def test_parse_real_model_value_supports_cvc5_forms(raw: str, expected: float) -> None:
    assert solver._parse_real_model_value(raw) == expected


def test_parse_real_model_value_avoids_large_rational_overflow() -> None:
    numerator = "1" + ("0" * 400)
    denominator = "2" + ("0" * 400)

    parsed = solver._parse_real_model_value(f"(/ {numerator} {denominator})")

    assert parsed == 0.5
    assert math.isfinite(parsed)


def test_describe_real_model_value_records_legacy_overflow_risk() -> None:
    numerator = "1" + ("0" * 400)
    denominator = "2" + ("0" * 400)

    diagnostic = solver._describe_real_model_value(f"(/ {numerator} {denominator})")

    assert diagnostic["is_rational"] is True
    assert diagnostic["numerator_digits"] == 401
    assert diagnostic["denominator_digits"] == 401
    assert diagnostic["legacy_numerator_float"]["finite"] is False
    assert diagnostic["legacy_denominator_float"]["finite"] is False
    assert diagnostic["legacy_division"]["class"] == "nan"
    assert diagnostic["exact_float"] == {"finite": True, "class": "finite", "value": 0.5}
    assert diagnostic["exact_in_norm_range"] is True


def test_describe_real_model_value_preserves_huge_rational_metadata_when_int_parse_fails() -> None:
    get_digit_limit = getattr(sys, "get_int_max_str_digits", None)
    if get_digit_limit is None:
        pytest.skip("Python runtime does not enforce integer string digit limits")
    digit_limit = get_digit_limit()
    if digit_limit <= 0:
        pytest.skip("Python integer string digit limit is disabled")

    digit_count = digit_limit + 1
    numerator = "1" * digit_count
    denominator = "2" * digit_count
    raw_value = f"(/ {numerator} {denominator})"

    diagnostic = solver._describe_real_model_value(raw_value)

    assert diagnostic["is_rational"] is True
    assert diagnostic["raw_value_length"] == len(raw_value)
    assert diagnostic["numerator_digits"] == digit_count
    assert diagnostic["denominator_digits"] == digit_count
    assert diagnostic["legacy_numerator_float"]["finite"] is False
    assert diagnostic["legacy_denominator_float"]["finite"] is False
    assert diagnostic["legacy_division"]["class"] == "nan"
    assert diagnostic["parse_error"] == "integer string exceeds Python int_max_str_digits"
    assert "Exceeds the limit" in diagnostic["parse_error_detail"]


@pytest.mark.parametrize("raw", ["NaN", "(/ 1 0)", "(+ 1 2)", "("])
def test_parse_real_model_value_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(solver.InvalidSolverModelError):
        solver._parse_real_model_value(raw)


@pytest.mark.parametrize(
    ("raw_value", "expected_error"),
    [
        ("NaN", "unsupported SMT Real atom"),
        ("2.0", "SMT Real is outside the normalized [0, 1] range"),
    ],
)
def test_find_model_discards_invalid_sat_binding_and_records_diagnostic(
    raw_value: str,
    expected_error: str,
) -> None:
    _configure_solver()
    fake_subprocess = SimpleNamespace(stdout=f"sat\n((x_VAR {raw_value}))\n".encode())

    with mock.patch.object(
        solver.Solver,
        "_build_formulas_from_constraint",
        return_value="(check-sat)\n",
    ):
        with mock.patch("libct.solver.subprocess.run", return_value=fake_subprocess):
            with mock.patch.object(solver.Solver, "_resolve_constraint_log_path", return_value=Path("inline.log")):
                with mock.patch.object(solver.Solver, "_append_constraint_log"):
                    result = solver.Solver.find_model_from_constraint(
                        _engine(),
                        constraint=SimpleNamespace(height=1),
                        shap_value=0.5,
                        position=(1,),
                        idx=0,
                        ori_args={"x": 0.0},
                    )

    assert result is None
    assert solver.Solver.stats["sat_number"] == 1
    assert solver.Solver.stats["invalid_model_number"] == 1
    assert solver.Solver.ctr_size["detail"][0]["status"] == "invalid_model"
    assert solver.Solver.ctr_size["detail"][0]["model_error"] == expected_error
    assert solver.Solver.ctr_size["detail"][0]["solver_stdout"] == f"sat\n((x_VAR {raw_value}))\n"
    assert solver.Solver.ctr_size["detail"][0]["solver_model_diagnostics"][0]["raw_value"] == raw_value
