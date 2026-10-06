from __future__ import annotations

from dataclasses import replace
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from libct.concolic import Concolic
from libct.executor import CandidateExecutionRunner, ConcolicArgumentBuilder
from libct.global_real import (
    GLOBAL_BRIGHTNESS_INPUT_NAME,
    GLOBAL_BRIGHTNESS_SMT_NAME,
    GLOBAL_CONTRAST_INPUT_NAME,
    GLOBAL_CONTRAST_SMT_NAME,
    GLOBAL_X_INPUT_NAME,
    GLOBAL_X_SMT_NAME,
    TRANSFORM_MODE_AFFINE_BC,
    build_concolic_global_real_kwargs,
    materialize_global_real_details,
    materialize_global_real_arguments,
    solver_variable_bounds,
    validate_global_real_config,
)
import libct.global_real as global_real
from libct.aces_like import (
    ACES_LIKE_COLOR_SPACE,
    ACES_LIKE_CURVE_VERSION,
    ACES_LIKE_GAMUT_MAPPER,
    ACES_LIKE_PWL_ERROR_METRIC,
    ACES_LIKE_PWL_VALIDATOR_VERSION,
    PiecewiseLinearApproximation,
    build_adaptive_pwl_approximation,
)
from libct.predicate import Predicate
from libct.record import ConcolicTestRecorder
from libct.solver import Solver
from tasks.builders import global_real as global_real_builder


def _config(*, bounds_mode="clip"):
    return {
        "variable_name": GLOBAL_X_INPUT_NAME,
        "requested_min": -0.1,
        "requested_max": 0.1,
        "effective_min": -0.1,
        "effective_max": 0.1,
        "bounds_mode": bounds_mode,
        "sign_by_input": {"v_0": 1, "v_1": -1, "v_2": 0},
    }


def _bc_config():
    return {
        "transform_mode": TRANSFORM_MODE_AFFINE_BC,
        "effective_min": -0.1,
        "effective_max": 0.1,
        "bounds_mode": "clip",
        "coefficient_by_input": {"v_0": -0.2, "v_1": 0.2},
    }


class _Engine:
    class LazyLoading:
        pass

    def __init__(self, config):
        self.idx = 0
        self.reference_execute = None
        self.constraints_collection_type = "priority_queue"
        self.global_real_config = config
        self.concolic_name_list = []
        self.concolic_flag_dict = {}
        self.var_to_types = {}
        self.symbolic_enabled = True


def test_global_real_argument_builder_uses_one_shared_real() -> None:
    def predict(**kwargs):
        return kwargs

    engine = _Engine(_config())
    primitive = {
        "v_0": 0.95,
        "v_1": 0.2,
        "v_2": 0.4,
        GLOBAL_X_INPUT_NAME: 0.0,
    }

    args, kwargs = ConcolicArgumentBuilder(engine).build(
        predict,
        primitive,
        {GLOBAL_X_INPUT_NAME: 1},
    )

    assert args == []
    assert set(kwargs) == {"v_0", "v_1", "v_2"}
    assert isinstance(kwargs["v_0"], Concolic)
    assert isinstance(kwargs["v_1"], Concolic)
    assert kwargs["v_2"] == pytest.approx(0.4)
    assert GLOBAL_X_SMT_NAME in Predicate.get_formula_deep(kwargs["v_0"])
    assert engine.concolic_name_list == [GLOBAL_X_SMT_NAME]
    assert engine.var_to_types == {GLOBAL_X_SMT_NAME: "Real"}


def test_hybrid_argument_builder_uses_two_bounded_reals() -> None:
    engine = _Engine(_bc_config())
    primitive = {
        "v_0": 0.2,
        "v_1": 0.8,
        GLOBAL_BRIGHTNESS_INPUT_NAME: 0.05,
        GLOBAL_CONTRAST_INPUT_NAME: -0.1,
    }

    args, kwargs = ConcolicArgumentBuilder(engine).build(
        lambda **values: values,
        primitive,
        {GLOBAL_BRIGHTNESS_INPUT_NAME: 1, GLOBAL_CONTRAST_INPUT_NAME: 1},
    )

    assert args == []
    assert set(kwargs) == {"v_0", "v_1"}
    assert engine.concolic_name_list == [GLOBAL_BRIGHTNESS_SMT_NAME, GLOBAL_CONTRAST_SMT_NAME]
    assert engine.var_to_types == {
        GLOBAL_BRIGHTNESS_SMT_NAME: "Real",
        GLOBAL_CONTRAST_SMT_NAME: "Real",
    }
    formula = Predicate.get_formula_deep(kwargs["v_0"])
    assert GLOBAL_BRIGHTNESS_SMT_NAME in formula
    assert GLOBAL_CONTRAST_SMT_NAME in formula
    assert float(kwargs["v_0"]) == pytest.approx(0.27)
    assert float(kwargs["v_1"]) == pytest.approx(0.83)


def test_hybrid_materialization_and_solver_bounds() -> None:
    inputs = {
        "v_0": 0.02,
        "v_1": 0.98,
        GLOBAL_BRIGHTNESS_INPUT_NAME: 0.05,
        GLOBAL_CONTRAST_INPUT_NAME: 0.1,
    }
    materialized, params, clipped_count = materialize_global_real_arguments(
        inputs, _bc_config()
    )

    assert params == pytest.approx((0.05, 0.1))
    assert clipped_count == 1
    assert materialized == pytest.approx({"v_0": 0.05, "v_1": 1.0})
    assert solver_variable_bounds(_bc_config()) == {
        GLOBAL_BRIGHTNESS_SMT_NAME: (-0.1, 0.1),
        GLOBAL_CONTRAST_SMT_NAME: (-0.1, 0.1),
    }


def test_solver_declares_and_bounds_both_hybrid_variables(monkeypatch) -> None:
    class _Constraint:
        @staticmethod
        def get_all_asserts():
            return []

    monkeypatch.setattr(Solver, "norm", True)
    monkeypatch.setattr(Solver, "limit_change_range", None)
    bounds = solver_variable_bounds(_bc_config())
    engine = SimpleNamespace(
        concolic_name_list=[GLOBAL_BRIGHTNESS_SMT_NAME, GLOBAL_CONTRAST_SMT_NAME],
        var_to_types={name: "Real" for name in bounds},
        solver_variable_bounds=bounds,
    )
    formula = Solver._build_formulas_from_constraint(engine, _Constraint(), {})

    for name in bounds:
        assert f"(declare-const {name} Real)" in formula
        assert f"(<= {name} 0.100000000000000)" in formula
        assert f"(>= {name} (- 0.100000000000000))" in formula


def test_hybrid_rejects_missing_or_out_of_bounds_parameter() -> None:
    inputs = {
        "v_0": 0.2,
        "v_1": 0.8,
        GLOBAL_BRIGHTNESS_INPUT_NAME: 0.0,
        GLOBAL_CONTRAST_INPUT_NAME: 0.0,
    }
    with pytest.raises(ValueError, match="missing"):
        materialize_global_real_arguments(
            {name: value for name, value in inputs.items() if name != GLOBAL_CONTRAST_INPUT_NAME},
            _bc_config(),
        )
    with pytest.raises(ValueError, match="outside configured bounds"):
        materialize_global_real_arguments(
            {**inputs, GLOBAL_CONTRAST_INPUT_NAME: 0.2}, _bc_config()
        )


def test_materialize_global_real_clip_matches_reference_semantics() -> None:
    materialized, shift, clipped_count = materialize_global_real_arguments(
        {
            "v_0": 0.95,
            "v_1": 0.2,
            "v_2": 0.4,
            GLOBAL_X_INPUT_NAME: 0.1,
        },
        _config(),
    )

    assert shift == pytest.approx(0.1)
    assert clipped_count == 1
    assert materialized == pytest.approx({"v_0": 1.0, "v_1": 0.1, "v_2": 0.4})


def test_materialize_global_real_strict_rejects_saturation() -> None:
    with pytest.raises(ValueError, match="strict-mode"):
        materialize_global_real_arguments(
            {
                "v_0": 0.95,
                "v_1": 0.2,
                "v_2": 0.4,
                GLOBAL_X_INPUT_NAME: 0.1,
            },
            _config(bounds_mode="strict"),
        )


def test_global_real_primitive_runner_removes_control_argument() -> None:
    def predict(**kwargs):
        return kwargs

    engine = _Engine(_config())
    args, kwargs = CandidateExecutionRunner(engine).complete_primitive_arguments(
        predict,
        {
            "v_0": 0.2,
            "v_1": 0.8,
            "v_2": 0.4,
            GLOBAL_X_INPUT_NAME: 0.05,
        },
    )

    assert args == []
    assert GLOBAL_X_INPUT_NAME not in kwargs
    assert kwargs == pytest.approx({"v_0": 0.25, "v_1": 0.75, "v_2": 0.4})


def test_solver_uses_explicit_global_real_bounds(monkeypatch) -> None:
    class _Constraint:
        @staticmethod
        def get_all_asserts():
            return []

    monkeypatch.setattr(Solver, "norm", True)
    monkeypatch.setattr(Solver, "limit_change_range", None)
    engine = SimpleNamespace(
        concolic_name_list=[GLOBAL_X_SMT_NAME],
        var_to_types={GLOBAL_X_SMT_NAME: "Real"},
        solver_variable_bounds={GLOBAL_X_SMT_NAME: (-0.1, 0.1)},
    )

    formula = Solver._build_formulas_from_constraint(
        engine,
        _Constraint(),
        {GLOBAL_X_INPUT_NAME: 0.0},
    )

    assert f"(declare-const {GLOBAL_X_SMT_NAME} Real)" in formula
    assert (
        f"(<= {GLOBAL_X_SMT_NAME} 0.100000000000000)" in formula
    )
    assert (
        f"(>= {GLOBAL_X_SMT_NAME} (- 0.100000000000000))" in formula
    )
    assert f"(>= {GLOBAL_X_SMT_NAME} 0)" not in formula


def test_solver_accepts_negative_global_real_model_and_materializes_image(monkeypatch) -> None:
    monkeypatch.setattr(Solver, "norm", True)
    engine = SimpleNamespace(
        var_to_types={GLOBAL_X_SMT_NAME: "Real"},
        solver_variable_bounds={GLOBAL_X_SMT_NAME: (-0.1, 0.1)},
    )

    model = Solver._get_model(
        engine,
        [f"(({GLOBAL_X_SMT_NAME} (- (/ 1 20))))"],
    )
    materialized, shift, clipped_count = materialize_global_real_arguments(
        {
            "v_0": 0.2,
            "v_1": 0.8,
            "v_2": 0.4,
            **model,
        },
        _config(),
    )

    assert shift == pytest.approx(-0.05)
    assert clipped_count == 0
    assert materialized == pytest.approx({"v_0": 0.15, "v_1": 0.85, "v_2": 0.4})
    assert all(0.0 <= value <= 1.0 for value in materialized.values())


def test_recorder_saves_materialized_adversarial_input_and_x() -> None:
    recorder = ConcolicTestRecorder(None, "case_0")
    recorder.input_shape = (3,)
    recorder.global_real_config = _config()

    recorder.find_adversarial_input(
        {
            "v_0": 0.95,
            "v_1": 0.2,
            "v_2": 0.4,
            GLOBAL_X_INPUT_NAME: 0.1,
        },
        attack_label=2,
    )

    np.testing.assert_allclose(recorder.adversarial_input, [1.0, 0.1, 0.4])
    assert recorder.extra_meta["global_real_solved_x"] == pytest.approx(0.1)
    assert recorder.extra_meta["global_real_solved_clipped_count"] == 1


def test_recorder_tracks_each_sat_global_x_candidate() -> None:
    recorder = ConcolicTestRecorder(None, "case_0")
    recorder.input_shape = (3,)
    recorder.global_real_config = _config()

    recorder.save_sat_input(
        {
            "v_0": 0.2,
            "v_1": 0.8,
            "v_2": 0.4,
            GLOBAL_X_INPUT_NAME: 0.05,
        }
    )

    assert recorder.global_real_sat_x == pytest.approx([0.05])
    assert recorder.global_real_sat_clipped_count == [0]
    np.testing.assert_allclose(recorder.sat_inputs[0], [0.25, 0.75, 0.4])


def test_validate_global_real_config_requires_matching_sign_values() -> None:
    config = _config()
    config["sign_by_input"] = {"v_0": 2}

    with pytest.raises(ValueError, match="must be -1, 0, or 1"):
        validate_global_real_config(config)


def test_cifar10_global_real_builder_creates_shared_x_payload(monkeypatch) -> None:
    sample = np.array([[[0.2], [0.8]]], dtype=np.float32)
    background = np.stack([sample, sample])

    class _Dataset:
        x_test = np.stack([sample])

        def get_cifar10_test_data(self, idx):
            return {"v_0_0_0": 0.2, "v_0_1_0": 0.8}, {
                "v_0_0_0": 0,
                "v_0_1_0": 0,
            }

        def get_cifar10_test_data_and_set_condict(self, idx, pixels):
            in_dict, con_dict = self.get_cifar10_test_data(idx)
            return in_dict, con_dict, sample, background

    class _Provider:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def load_cached(**kwargs):
            return SimpleNamespace(
                values=np.array([[[0.3], [-0.4]]]),
                target_class=1,
                cache_path="cache.json",
            )

        @staticmethod
        def ensure(**kwargs):
            pytest.fail("builder must not generate SHAP or initialize Keras")

    monkeypatch.setattr(global_real_builder, "Cifar10Dataset", _Dataset)
    monkeypatch.setattr(global_real_builder, "TargetClassInputShapProvider", _Provider)
    monkeypatch.setattr(global_real_builder, "get_save_dir_from_save_exp", lambda *a, **k: "unused")

    payloads = global_real_builder.cifar10_global_real(
        "demo",
        [0],
        force=True,
        bounds_mode="clip",
    )

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["con_dict"] == {GLOBAL_X_INPUT_NAME: 1}
    assert payload["in_dict"][GLOBAL_X_INPUT_NAME] == 0.0
    config = payload["global_real_config"]
    assert config["shap_target_class"] == 1
    assert config["coefficient_by_input"] == pytest.approx(
        {"v_0_0_0": 1.0, "v_0_1_0": -1.0}
    )


def test_materialize_global_real_accepts_affine_coefficients() -> None:
    config = {
        "variable_name": GLOBAL_X_INPUT_NAME,
        "effective_min": -0.5,
        "effective_max": 0.5,
        "bounds_mode": "strict",
        "coefficient_by_input": {"v_0": 1.0, "v_1": -0.5, "v_2": 0.0},
    }

    materialized, shift, clipped_count = materialize_global_real_arguments(
        {"v_0": 0.2, "v_1": 0.8, "v_2": 0.4, GLOBAL_X_INPUT_NAME: 0.2},
        config,
    )

    assert shift == pytest.approx(0.2)
    assert clipped_count == 0
    assert materialized == pytest.approx({"v_0": 0.4, "v_1": 0.7, "v_2": 0.4})


def test_cifar10_global_real_brightness_and_contrast_do_not_load_shap(monkeypatch) -> None:
    sample = np.array([[[0.2], [0.8]]], dtype=np.float32)

    class _Dataset:
        x_test = np.stack([sample])

        def get_cifar10_test_data(self, idx):
            return {"v_0_0_0": 0.2, "v_0_1_0": 0.8}, {}

    class _Provider:
        def __init__(self, **kwargs):
            pytest.fail("appearance shifts must not initialize a SHAP provider")

    monkeypatch.setattr(global_real_builder, "Cifar10Dataset", _Dataset)
    monkeypatch.setattr(global_real_builder, "TargetClassInputShapProvider", _Provider)
    monkeypatch.setattr(global_real_builder, "get_save_dir_from_save_exp", lambda *a, **k: "unused")

    brightness = global_real_builder.cifar10_global_real(
        "demo", [0], force=True, shift_kind="brightness"
    )[0]["global_real_config"]
    contrast = global_real_builder.cifar10_global_real(
        "demo", [0], force=True, shift_kind="contrast"
    )[0]["global_real_config"]

    assert brightness["coefficient_by_input"] == pytest.approx(
        {"v_0_0_0": 1.0, "v_0_1_0": 1.0}
    )
    assert contrast["coefficient_by_input"] == pytest.approx(
        {"v_0_0_0": -0.3, "v_0_1_0": 0.3}
    )
    assert contrast["contrast_channel_means"] == pytest.approx([0.5])


def test_validate_global_real_config_normalizes_legacy_sign_mapping_once() -> None:
    normalized = validate_global_real_config(_config())
    revalidated = validate_global_real_config(normalized)

    assert "sign_by_input" not in normalized
    assert revalidated["coefficient_by_input"] == pytest.approx(
        {"v_0": 1.0, "v_1": -1.0, "v_2": 0.0}
    )

def _aces_config_for_rgb(rgb, *, kind="aces-brightness"):
    approximation = build_adaptive_pwl_approximation(
        rgb,
        kind=kind,
        x_min=-0.1,
        x_max=0.1,
        max_segments=8,
        error_tolerance=1.0 / 255.0,
    )
    return {
        "variable_name": GLOBAL_X_INPUT_NAME,
        "effective_min": -0.1,
        "effective_max": 0.1,
        "bounds_mode": "clip",
        "transform_mode": "aces-like-pwl",
        "global_shift_kind": kind,
        "pwl_knots": approximation.knots.tolist(),
        "pwl_max_segments": 8,
        "pwl_segment_count": approximation.segment_count,
        "pwl_error_tolerance": 1.0 / 255.0,
        "pwl_max_abs_error": approximation.max_abs_error,
        "pwl_error_metric": ACES_LIKE_PWL_ERROR_METRIC,
        "pwl_validator_version": ACES_LIKE_PWL_VALIDATOR_VERSION,
        "aces_like_color_space": ACES_LIKE_COLOR_SPACE,
        "aces_like_curve_version": ACES_LIKE_CURVE_VERSION,
        "aces_like_gamut_mapper": ACES_LIKE_GAMUT_MAPPER,
    }


def _rgb_inputs(rgb, shift):
    inputs = {
        "v_" + "_".join(str(part) for part in index): float(rgb[index])
        for index in np.ndindex(rgb.shape)
    }
    inputs[GLOBAL_X_INPUT_NAME] = shift
    return inputs


def test_aces_like_materialization_preserves_rgb_layout_and_reports_diagnostics() -> None:
    rgb = np.array([[[0.2, 0.5, 0.8], [0.9, 0.1, 0.4]]], dtype=np.float64)
    config = _aces_config_for_rgb(rgb)
    materialized, shift, clipped_count = materialize_global_real_arguments(
        _rgb_inputs(rgb, 0.05), config
    )
    details = materialize_global_real_details(_rgb_inputs(rgb, 0.05), config)
    assert shift == pytest.approx(0.05)
    assert clipped_count == 0
    assert details[3]["transform_mode"] == "aces-like-pwl"
    assert details[3]["pwl_error_at_x"] <= config["pwl_max_abs_error"] + 1e-12
    output = np.array(
        [[[materialized[f"v_{i}_{j}_{c}"] for c in range(3)] for j in range(2)] for i in range(1)]
    )
    assert output.shape == rgb.shape
    assert np.all((output >= 0.0) & (output <= 1.0))


def test_aces_like_symbolic_arguments_use_one_shared_x_and_piecewise_ites() -> None:
    rgb = np.array([[[0.2, 0.5, 0.8]]], dtype=np.float64)
    config = _aces_config_for_rgb(rgb, kind="aces-contrast")
    engine = _Engine(config)
    kwargs = build_concolic_global_real_kwargs(engine, _rgb_inputs(rgb, 0.05))
    assert isinstance(kwargs["v_0_0_0"], Concolic)
    assert isinstance(kwargs["v_0_0_1"], Concolic)
    assert engine.concolic_name_list == [GLOBAL_X_SMT_NAME]
    formula = Predicate.get_formula_deep(kwargs["v_0_0_0"])
    assert "ite" in repr(formula)
    assert GLOBAL_X_SMT_NAME in repr(formula)


def test_aces_like_pwl_symbolic_formula_accepts_negative_literals_in_cvc5() -> None:
    cvc5 = shutil.which("cvc5")
    if cvc5 is None:
        pytest.skip("cvc5 is required for the runtime SMT serialization regression")

    approximation = PiecewiseLinearApproximation(
        knots=np.asarray((-0.1, -0.05, 0.1), dtype=np.float64),
        rgb_at_knots=np.asarray(
            [
                [[[1.0, 0.5, 0.5]]],
                [[[0.0, 0.5, 0.5]]],
                [[[0.5, 0.5, 0.5]]],
            ],
            dtype=np.float64,
        ),
        max_abs_error=0.0,
        error_tolerance=1.0 / 255.0,
    )
    expression = global_real._pwl_symbolic_expression(
        GLOBAL_X_SMT_NAME, approximation, (0, 0, 0)
    )
    formula = Predicate.get_formula_deep(expression)
    assert "(- 0.050000000000000)" in formula
    assert "(- 1.000000000000000)" in formula

    smt2 = "\n".join(
        (
            "(set-logic QF_LRA)",
            f"(declare-const {GLOBAL_X_SMT_NAME} Real)",
            f"(assert (and (<= {GLOBAL_X_SMT_NAME} 0.1) "
            f"(>= {GLOBAL_X_SMT_NAME} (- 0.1))))",
            f"(assert (= {GLOBAL_X_SMT_NAME} {formula}))",
            "(check-sat)",
        )
    )
    result = subprocess.run(
        [cvc5, "--produce-models", "--lang", "smt2", "--quiet"],
        input=smt2,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[0] in {"sat", "unsat"}


@pytest.mark.parametrize("defer_pwl", [False, True])
def test_cifar10_global_real_builder_creates_aces_like_pwl_payload(monkeypatch, defer_pwl) -> None:
    sample = np.array([[[0.2, 0.5, 0.8], [0.9, 0.1, 0.4]]], dtype=np.float32)
    class _Dataset:
        x_test = np.stack([sample])
        def get_cifar10_test_data(self, idx):
            return {
                "v_0_0_0": 0.2, "v_0_0_1": 0.5, "v_0_0_2": 0.8,
                "v_0_1_0": 0.9, "v_0_1_1": 0.1, "v_0_1_2": 0.4,
            }, {}
    monkeypatch.setattr(global_real_builder, "Cifar10Dataset", _Dataset)
    monkeypatch.setattr(global_real_builder, "get_save_dir_from_save_exp", lambda *a, **k: "unused")
    if defer_pwl:
        monkeypatch.setattr(
            global_real_builder, "build_aces_like_global_real_config",
            lambda *_args: pytest.fail("hybrid must build PWL after DE chooses a seed"),
        )
    config = global_real_builder.cifar10_global_real(
        "demo", [0], force=True, shift_kind="aces-brightness", defer_pwl=defer_pwl
    )[0]["global_real_config"]
    assert config["transform_mode"] == "aces-like-pwl"
    assert "coefficient_by_input" not in config
    if defer_pwl:
        assert config["pwl_deferred"] is True
        assert "pwl_knots" not in config
        assert "pwl_segment_count" not in config
        assert "pwl_max_abs_error" not in config
        assert config["pwl_max_segments"] == 32
        return
    assert config["pwl_knots"][0] == pytest.approx(-0.1)
    assert config["pwl_knots"][-1] == pytest.approx(0.1)
    assert config["pwl_segment_count"] == len(config["pwl_knots"]) - 1
    assert config["pwl_error_metric"] == ACES_LIKE_PWL_ERROR_METRIC
    assert config["pwl_validator_version"] == ACES_LIKE_PWL_VALIDATOR_VERSION
    assert config["aces_like_color_space"] == ACES_LIKE_COLOR_SPACE
    assert config["aces_like_curve_version"] == ACES_LIKE_CURVE_VERSION
    assert config["aces_like_gamut_mapper"] == ACES_LIKE_GAMUT_MAPPER
    assert config["probe_enabled"] is True
    assert config["probe_initial_points"] == 17
    assert config["probe_max_refinements"] == 8
    assert config["probe_tolerance_fraction"] == pytest.approx(1.0 / 1024.0)


@pytest.mark.parametrize("kind", ["aces-brightness", "aces-contrast"])
def test_aces_hybrid_seed_builds_other_axis_and_preserves_zero_identity(kind) -> None:
    source = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float64)
    seed = global_real.apply_aces_like_transform(source, 0.08, kind=kind).rgb
    other_kind = "aces-contrast" if kind == "aces-brightness" else "aces-brightness"
    deferred = {
        "variable_name": GLOBAL_X_INPUT_NAME,
        "effective_min": -0.1, "effective_max": 0.1,
        "bounds_mode": "clip", "global_shift_kind": other_kind,
        "pwl_deferred": True, "hybrid_de_enabled": True,
        "pwl_max_segments": 8, "pwl_error_tolerance": 1.0 / 255.0,
    }

    config = global_real.build_aces_like_global_real_config(seed, deferred)
    inputs = _rgb_inputs(seed, 0.0)
    materialized, shift, clipped = materialize_global_real_arguments(inputs, config)

    assert shift == 0.0
    assert clipped == 0
    assert config["hybrid_de_enabled"] is True
    assert config["global_shift_kind"] == other_kind
    assert 0.0 in config["pwl_knots"]
    assert config["pwl_max_abs_error"] <= config["pwl_error_tolerance"]
    assert "pwl_deferred" not in config
    assert deferred["pwl_deferred"] is True
    for name, value in inputs.items():
        if name != GLOBAL_X_INPUT_NAME:
            assert materialized[name] == value


@pytest.mark.parametrize("kind", ["aces-brightness", "aces-contrast"])
def test_hybrid_aces_reference_and_recorded_image_use_exact_transform_between_knots(kind):
    seed = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float64)
    config = _aces_config_for_rgb(seed, kind=kind)
    shift = (config["pwl_knots"][0] + config["pwl_knots"][1]) / 2.0
    inputs = _rgb_inputs(seed, shift)
    approximate, _, _ = materialize_global_real_arguments(inputs, config)
    exact = global_real.apply_aces_like_transform(seed, shift, kind=kind).rgb
    approximation = np.asarray([[[approximate[f"v_0_0_{i}"] for i in range(3)]]])
    assert np.max(np.abs(approximation - exact)) > 1e-7
    hybrid_config = {**config, "hybrid_de_enabled": True}

    materialized, _, _, diagnostics = materialize_global_real_details(inputs, hybrid_config)

    result = np.asarray([[[materialized[f"v_0_0_{i}"] for i in range(3)]]])
    np.testing.assert_array_equal(result, exact)
    assert diagnostics["pwl_error_at_x"] <= config["pwl_max_abs_error"] + 1e-12
    recorder = ConcolicTestRecorder(None, "case_0")
    recorder.global_real_config = hybrid_config
    recorder.input_shape = seed.shape
    recorder.save_sat_input(inputs)
    recorder.find_adversarial_input(inputs, attack_label=1)
    np.testing.assert_array_equal(recorder.sat_inputs[0], exact.astype(np.float32))
    np.testing.assert_array_equal(recorder.adversarial_input, exact.astype(np.float32))
    channel = int(np.argmax(np.abs(approximation - exact)))
    midpoint = (approximation[0, 0, channel] + exact[0, 0, channel]) / 2.0
    approximation_is_higher = approximation[0, 0, channel] > exact[0, 0, channel]

    def predict_scores(images):
        value = images[0, 0, 0, channel]
        changed = value > midpoint if approximation_is_higher else value < midpoint
        return np.asarray([[0.1, 0.9] if changed else [0.9, 0.1]])

    assert np.argmax(predict_scores(approximation[np.newaxis, ...])[0]) == 1
    hybrid_config["probe_enabled"] = False
    recorder.original_label = 0
    recorder.attack_label = None
    engine = SimpleNamespace(
        global_real_config=hybrid_config,
        reference_score_predictor=predict_scores,
        _get_recorder=lambda: recorder,
    )
    assert CandidateExecutionRunner(engine).validate_sat_candidate(inputs) is False
    assert recorder.attack_label is None
    assert engine.current_reference_margin == pytest.approx(0.8)

    def reference(**kwargs):
        return kwargs

    def search(**kwargs):
        return kwargs

    engine.reference_execute = reference
    runner = CandidateExecutionRunner(engine)
    _, reference_inputs = runner.complete_primitive_arguments(reference, inputs)
    _, search_inputs = runner.complete_primitive_arguments(search, inputs)
    assert reference_inputs == materialized
    assert search_inputs == approximate


def test_aces_like_validation_rejects_strict_mode() -> None:
    rgb = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float64)
    config = _aces_config_for_rgb(rgb)
    config["bounds_mode"] = "strict"

    with pytest.raises(ValueError, match="bounds_mode='clip'"):
        validate_global_real_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("pwl_segment_count", 0, "segment_count"),
        ("aces_like_gamut_mapper", "old-mapper", "gamut_mapper"),
    ),
)
def test_aces_like_validation_rejects_segment_or_mapper_mismatch(
    field, value, message
) -> None:
    rgb = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float64)
    config = _aces_config_for_rgb(rgb)
    config[field] = value

    with pytest.raises(ValueError, match=message):
        validate_global_real_config(config)


def test_aces_like_runtime_exact_error_check_fails_closed(monkeypatch) -> None:
    rgb = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float64)
    config = _aces_config_for_rgb(rgb)
    original_builder = global_real.build_adaptive_pwl_approximation

    def build_tampered_approximation(*args, **kwargs) -> PiecewiseLinearApproximation:
        approximation = original_builder(*args, **kwargs)
        values = np.array(approximation.rgb_at_knots, copy=True)
        values[1] += 0.2
        return replace(approximation, rgb_at_knots=values)

    monkeypatch.setattr(
        global_real,
        "build_adaptive_pwl_approximation",
        build_tampered_approximation,
    )

    with pytest.raises(ValueError, match="exceeds tolerance"):
        materialize_global_real_details(_rgb_inputs(rgb, 0.05), config)


def _joint_aces_config(rgb, *, order="brightness-contrast", hybrid=False):
    return global_real.build_aces_like_global_real_config(rgb, {
        "search_axes": "both", "global_shift_kind": "aces-brightness",
        "effective_min": -0.1, "effective_max": 0.1, "bounds_mode": "clip",
        "transform_order": order, "pwl_max_triangles": 128,
        "pwl_error_tolerance": 1.0 / 255.0, "hybrid_de_enabled": hybrid,
    })


def _joint_rgb_inputs(rgb, brightness, contrast):
    inputs = _rgb_inputs(rgb, 0.0)
    inputs.pop(GLOBAL_X_INPUT_NAME)
    inputs[GLOBAL_BRIGHTNESS_INPUT_NAME] = brightness
    inputs[GLOBAL_CONTRAST_INPUT_NAME] = contrast
    return inputs


@pytest.mark.parametrize("order", ["brightness-contrast", "contrast-brightness"])
def test_joint_aces_seed_zero_preserves_pixels_and_declares_two_bounded_reals(order):
    seed = np.asarray([[[0.2, 0.5, 0.8]]])
    config = _joint_aces_config(seed, order=order, hybrid=True)
    primitive = _joint_rgb_inputs(seed, 0.0, 0.0)
    engine = _Engine(config)
    args, kwargs = ConcolicArgumentBuilder(engine).build(
        lambda **values: values, primitive,
        {GLOBAL_BRIGHTNESS_INPUT_NAME: 1, GLOBAL_CONTRAST_INPUT_NAME: 1},
    )
    assert args == []
    assert config["transform_mode"] == "aces-like-pwl-2d"
    assert engine.concolic_name_list == [GLOBAL_BRIGHTNESS_SMT_NAME, GLOBAL_CONTRAST_SMT_NAME]
    assert engine.var_to_types == {
        GLOBAL_BRIGHTNESS_SMT_NAME: "Real", GLOBAL_CONTRAST_SMT_NAME: "Real",
    }
    assert solver_variable_bounds(config) == {
        GLOBAL_BRIGHTNESS_SMT_NAME: (-0.1, 0.1), GLOBAL_CONTRAST_SMT_NAME: (-0.1, 0.1),
    }
    materialized, parameters, clipped = materialize_global_real_arguments(primitive, config)
    assert parameters == (0.0, 0.0)
    assert clipped == 0
    assert set(kwargs) == set(materialized) == {"v_0_0_0", "v_0_0_1", "v_0_0_2"}
    for channel in range(3):
        assert materialized[f"v_0_0_{channel}"] == seed[0, 0, channel]
        assert float(kwargs[f"v_0_0_{channel}"]) == pytest.approx(seed[0, 0, channel])


def test_joint_aces_symbolic_formulas_execute_as_linear_real_arithmetic_in_cvc5():
    cvc5 = shutil.which("cvc5")
    if cvc5 is None:
        pytest.skip("cvc5 is required for the runtime SMT serialization regression")
    from libct.utils import py2smt

    seed = np.asarray([[[0.2, 0.5, 0.8]]])
    config = _joint_aces_config(seed)
    engine = _Engine(config)
    _, kwargs = ConcolicArgumentBuilder(engine).build(
        lambda **values: values, _joint_rgb_inputs(seed, 0.0, 0.0),
        {GLOBAL_BRIGHTNESS_INPUT_NAME: 1, GLOBAL_CONTRAST_INPUT_NAME: 1},
    )
    formulas = [Predicate.get_formula_deep(kwargs[f"v_0_0_{channel}"]) for channel in range(3)]
    assert all(GLOBAL_BRIGHTNESS_SMT_NAME in formula and GLOBAL_CONTRAST_SMT_NAME in formula
               and "ite" in formula for formula in formulas)
    for brightness, contrast in ((0.033, -0.046), (-0.047, 0.024), (0.075, 0.083), (0.0, 0.0)):
        materialized, _, _ = materialize_global_real_arguments(
            _joint_rgb_inputs(seed, brightness, contrast), config
        )
        statements = [
            "(set-logic QF_LRA)",
            f"(declare-const {GLOBAL_BRIGHTNESS_SMT_NAME} Real)",
            f"(declare-const {GLOBAL_CONTRAST_SMT_NAME} Real)",
            f"(assert (= {GLOBAL_BRIGHTNESS_SMT_NAME} {py2smt(brightness)}))",
            f"(assert (= {GLOBAL_CONTRAST_SMT_NAME} {py2smt(contrast)}))",
        ]
        for channel, formula in enumerate(formulas):
            value = materialized[f"v_0_0_{channel}"]
            statements.append(
                f"(assert (and (>= {formula} {py2smt(value - 1e-10)}) "
                f"(<= {formula} {py2smt(value + 1e-10)})))"
            )
        statements.append("(check-sat)")
        result = subprocess.run(
            [cvc5, "--lang", "smt2", "--quiet"], input="\n".join(statements),
            text=True, capture_output=True, check=False, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "sat", result.stdout + result.stderr


@pytest.mark.parametrize("order", ["brightness-contrast", "contrast-brightness"])
def test_joint_aces_hybrid_reference_and_recorded_images_use_exact_transform(order):
    from libct.aces_like import apply_aces_like_joint_transform

    seed = np.asarray([[[0.2, 0.5, 0.8]]])
    config = _joint_aces_config(seed, order=order)
    brightness, contrast = 0.033, -0.046
    primitive = _joint_rgb_inputs(seed, brightness, contrast)
    approximate, _, _ = materialize_global_real_arguments(primitive, config)
    approximation = np.asarray([[[approximate[f"v_0_0_{channel}"] for channel in range(3)]]])
    exact = apply_aces_like_joint_transform(seed, brightness, contrast, order=order).rgb
    assert np.max(np.abs(approximation - exact)) > 1e-7
    hybrid = {**config, "hybrid_de_enabled": True, "probe_enabled": False}
    materialized, parameters, _, diagnostics = materialize_global_real_details(primitive, hybrid)
    assert parameters == (brightness, contrast)
    result = np.asarray([[[materialized[f"v_0_0_{channel}"] for channel in range(3)]]])
    np.testing.assert_array_equal(result, exact)
    assert diagnostics["pwl_error_at_x"] <= hybrid["pwl_error_tolerance"]

    recorder = ConcolicTestRecorder(None, "case_joint")
    recorder.global_real_config = hybrid
    recorder.input_shape = seed.shape
    recorder.save_sat_input(primitive)
    recorder.find_adversarial_input(primitive, attack_label=1)
    np.testing.assert_array_equal(recorder.sat_inputs[0], exact.astype(np.float32))
    np.testing.assert_array_equal(recorder.adversarial_input, exact.astype(np.float32))

    channel = int(np.argmax(np.abs(approximation - exact)))
    midpoint = (approximation[0, 0, channel] + exact[0, 0, channel]) / 2.0
    approximate_is_higher = approximation[0, 0, channel] > exact[0, 0, channel]

    def predict_scores(images):
        value = images[0, 0, 0, channel]
        changed = value > midpoint if approximate_is_higher else value < midpoint
        return np.asarray([[0.1, 0.9] if changed else [0.9, 0.1]])

    assert np.argmax(predict_scores(approximation[np.newaxis, ...])[0]) == 1
    recorder.original_label = 0
    recorder.attack_label = None
    engine = SimpleNamespace(
        global_real_config=hybrid, reference_score_predictor=predict_scores,
        _get_recorder=lambda: recorder,
    )
    runner = CandidateExecutionRunner(engine)
    assert runner.validate_sat_candidate(primitive) is False
    assert recorder.attack_label is None
    assert engine.current_reference_margin == pytest.approx(0.8)

    def reference(**kwargs):
        return kwargs

    engine.reference_execute = reference
    _, reference_inputs = runner.complete_primitive_arguments(reference, primitive)
    _, search_inputs = runner.complete_primitive_arguments(lambda **kwargs: kwargs, primitive)
    assert reference_inputs == materialized
    assert search_inputs == approximate


@pytest.mark.parametrize("field,value", [
    ("pwl_triangle_count", 0), ("pwl_max_triangles", 1),
    ("pwl_max_abs_error", 1.0), ("pwl_validator_version", "wrong-version"),
    ("aces_like_gamut_mapper", "wrong-mapper"), ("transform_order", "wrong-order"),
])
def test_joint_aces_config_rejects_invalid_error_geometry_and_transform_metadata(field, value):
    config = _joint_aces_config(np.asarray([[[0.2, 0.5, 0.8]]]))
    config[field] = value
    with pytest.raises(ValueError):
        validate_global_real_config(config)


@pytest.mark.parametrize("values", [
    {GLOBAL_BRIGHTNESS_INPUT_NAME: 0.0},
    {GLOBAL_CONTRAST_INPUT_NAME: 0.0},
    {GLOBAL_BRIGHTNESS_INPUT_NAME: 0.0, GLOBAL_CONTRAST_INPUT_NAME: 0.11},
    {GLOBAL_BRIGHTNESS_INPUT_NAME: float("nan"), GLOBAL_CONTRAST_INPUT_NAME: 0.0},
    {GLOBAL_BRIGHTNESS_INPUT_NAME: 0.0, GLOBAL_CONTRAST_INPUT_NAME: 0.0, GLOBAL_X_INPUT_NAME: 0.0},
])
def test_joint_aces_materialization_rejects_missing_invalid_or_unbounded_parameters(values):
    seed = np.asarray([[[0.2, 0.5, 0.8]]])
    config = _joint_aces_config(seed)
    primitive = {f"v_0_0_{channel}": float(seed[0, 0, channel]) for channel in range(3)}
    with pytest.raises(ValueError):
        materialize_global_real_arguments({**primitive, **values}, config)


def test_joint_aces_runtime_error_check_rejects_tampered_rgb_table(monkeypatch):
    seed = np.asarray([[[0.2, 0.5, 0.8]]])
    config = _joint_aces_config(seed, hybrid=True)
    cached_builder = global_real._cached_planar

    def tampered_builder(*args):
        approximation = cached_builder(*args)
        return replace(
            approximation, rgb_at_knots=np.clip(approximation.rgb_at_knots + 0.1, 0.0, 1.0)
        )

    monkeypatch.setattr(global_real, "_cached_planar", tampered_builder)
    with pytest.raises(ValueError, match="exceeds tolerance"):
        materialize_global_real_details(_joint_rgb_inputs(seed, 0.033, -0.046), config)


def test_joint_aces_approximation_cache_does_not_reuse_different_seed_pixels():
    from libct.aces_like import apply_aces_like_joint_transform

    first = np.asarray([[[0.2, 0.5, 0.8]]])
    second = np.asarray([[[0.7, 0.3, 0.4]]])
    first_config = _joint_aces_config(first, hybrid=True)
    second_config = _joint_aces_config(second, hybrid=True)
    for seed, config in ((first, first_config), (second, second_config), (first, first_config)):
        inputs = _joint_rgb_inputs(seed, 0.033, -0.046)
        materialized, _, _ = materialize_global_real_arguments(inputs, config)
        exact = apply_aces_like_joint_transform(seed, 0.033, -0.046).rgb
        actual = np.asarray([[[materialized[f"v_0_0_{channel}"] for channel in range(3)]]])
        np.testing.assert_array_equal(actual, exact)
