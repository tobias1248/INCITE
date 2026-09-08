from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from libct.concolic import Concolic
from libct.executor import CandidateExecutionRunner, ConcolicArgumentBuilder
from libct.global_real import (
    GLOBAL_X_INPUT_NAME,
    GLOBAL_X_SMT_NAME,
    build_concolic_global_real_kwargs,
    materialize_global_real_details,
    materialize_global_real_arguments,
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


class _Engine:
    class LazyLoading:
        pass

    def __init__(self, config):
        self.idx = 0
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


def test_cifar10_global_real_builder_creates_aces_like_pwl_payload(monkeypatch) -> None:
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
    config = global_real_builder.cifar10_global_real(
        "demo", [0], force=True, shift_kind="aces-brightness"
    )[0]["global_real_config"]
    assert config["transform_mode"] == "aces-like-pwl"
    assert "coefficient_by_input" not in config
    assert config["pwl_knots"][0] == pytest.approx(-0.1)
    assert config["pwl_knots"][-1] == pytest.approx(0.1)
    assert config["pwl_segment_count"] == len(config["pwl_knots"]) - 1
    assert config["pwl_error_metric"] == ACES_LIKE_PWL_ERROR_METRIC
    assert config["pwl_validator_version"] == ACES_LIKE_PWL_VALIDATOR_VERSION
    assert config["aces_like_color_space"] == ACES_LIKE_COLOR_SPACE
    assert config["aces_like_curve_version"] == ACES_LIKE_CURVE_VERSION
    assert config["aces_like_gamut_mapper"] == ACES_LIKE_GAMUT_MAPPER


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
