from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from libct.concolic import Concolic
from libct.executor import CandidateExecutionRunner, ConcolicArgumentBuilder
from libct.global_real import (
    GLOBAL_X_INPUT_NAME,
    GLOBAL_X_SMT_NAME,
    materialize_global_real_arguments,
    validate_global_real_config,
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

    class _Model:
        @staticmethod
        def predict(inputs, verbose=0):
            return np.array([[0.1, 0.9]])

    class _Provider:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def ensure(**kwargs):
            return SimpleNamespace(
                values=np.array([[[0.3], [-0.4]]]),
                cache_path="cache.json",
            )

    monkeypatch.setattr(global_real_builder, "Cifar10Dataset", _Dataset)
    monkeypatch.setattr(global_real_builder, "load_model_with_compat", lambda _path: _Model())
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
    assert config["sign_by_input"] == {"v_0_0_0": 1, "v_0_1_0": -1}
