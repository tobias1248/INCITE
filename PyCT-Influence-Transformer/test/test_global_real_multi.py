from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

import libct.global_real as global_real
from libct.aces_like_multi import apply_aces_like_multi_transform, parameter_name
from libct.concolic import Concolic
from libct.executor import CandidateExecutionRunner, ConcolicArgumentBuilder
from libct.global_real import (
    GLOBAL_X_INPUT_NAME, GLOBAL_X_SMT_NAME,
    build_aces_like_global_real_config, build_concolic_global_real_kwargs,
    materialize_global_real_details, solver_variable_bounds, validate_global_real_config,
)
from libct.predicate import Predicate


RGB = np.asarray([[[0.2, 0.5, 0.8]]])


def _config(controls=("exposure", "contrast", "saturation")):
    return build_aces_like_global_real_config(RGB, {
        "global_shift_kind": "aces-multi", "controls": list(controls),
        "control_bounds": [[-0.02, 0.03], [-0.03, 0.04], [-0.04, 0.05]][:len(controls)],
        "effective_min": -0.2, "effective_max": 0.2,
        "requested_min": -0.2, "requested_max": 0.2,
        "pwl_max_regions": 512, "pwl_error_tolerance": 1 / 255,
        "bounds_mode": "clip", "probe_enabled": False,
        "shap_priority_enabled": True, "shap_sensitivity_enabled": False,
    })


def _inputs(config, params=None):
    inputs = {"v_0_0_" + str(channel): float(value) for channel, value in enumerate(RGB[0, 0])}
    inputs.update(zip(config["variable_names"], params or (0.0,) * len(config["controls"])))
    return inputs


def _engine(config):
    class Engine:
        class LazyLoading:
            pass

        def __init__(self):
            self.global_real_config = config
            self.idx = 0
            self.counter = 20
            self.reference_execute = None
            self.constraints_collection_type = "priority_queue"
            self.concolic_name_list = []
            self.concolic_flag_dict = {}
            self.var_to_types = {}
            self.symbolic_enabled = True

    return Engine()


@pytest.mark.parametrize("controls", [("exposure", "contrast"), ("exposure", "contrast", "saturation")])
def test_multi_config_records_complete_mesh_and_all_independent_solver_bounds(controls):
    config = _config(controls)
    assert config["transform_mode"] == "aces-like-pwl-nd"
    assert config["controls"] == config["control_order"] == list(controls)
    names = [parameter_name(control) for control in controls]
    assert config["variable_names"] == names
    assert solver_variable_bounds(config) == {
        name + "_VAR": tuple(bounds) for name, bounds in zip(names, config["control_bounds"])
    }
    assert GLOBAL_X_SMT_NAME not in solver_variable_bounds(config)
    assert len(config["pwl_nd_knots"]) == len(controls)
    assert config["pwl_region_count"] <= config["pwl_max_regions"]
    assert config["pwl_max_abs_error"] <= config["pwl_error_tolerance"]


@pytest.mark.parametrize("controls", [("exposure", "contrast"), ("exposure", "contrast", "saturation")])
def test_argument_builder_registers_every_axis_and_only_pixels_reach_predictor(controls):
    config = _config(controls)
    engine = _engine(config)
    primitive = _inputs(config, (0.013, -0.017, 0.021)[:len(controls)])
    args, kwargs = ConcolicArgumentBuilder(engine).build(
        lambda **values: values, primitive, {name: 1 for name in config["variable_names"]}
    )
    assert args == []
    assert set(kwargs) == {"v_0_0_0", "v_0_0_1", "v_0_0_2"}
    symbols = [name + "_VAR" for name in config["variable_names"]]
    assert engine.concolic_name_list == symbols
    assert engine.var_to_types == dict.fromkeys(symbols, "Real")
    assert engine._global_x_sensitivity is None
    expected, _, _, diagnostics = materialize_global_real_details(primitive, config, exact_transform=False)
    assert diagnostics["reference_transform"] == "pwl"
    for name, value in kwargs.items():
        assert isinstance(value, Concolic)
        assert float(value) == pytest.approx(expected[name])
        formula = repr(Predicate.get_formula_deep(value))
        assert "ite" in formula
        assert all(symbol in formula for symbol in symbols)
        assert GLOBAL_X_SMT_NAME not in formula


def test_primitive_candidate_materializes_complete_vector_with_exact_renderer():
    config = _config()
    params = (0.013, -0.017, 0.021)
    primitive = _inputs(config, params)
    output, actual_params, clipped, diagnostics = materialize_global_real_details(primitive, config)
    expected = apply_aces_like_multi_transform(RGB, params, controls=config["controls"])
    np.testing.assert_array_equal([output["v_0_0_" + str(i)] for i in range(3)], expected.rgb[0, 0])
    assert actual_params == params
    assert clipped == expected.diagnostics.hard_clipped_channel_count
    assert diagnostics["reference_transform"] == "exact"
    assert diagnostics["pwl_error_at_x"] <= config["pwl_error_tolerance"]
    reference = lambda **values: values
    engine = _engine(config)
    engine.reference_execute = reference
    runner = CandidateExecutionRunner(engine)
    args, kwargs = runner.complete_primitive_arguments(reference, primitive)
    assert args == []
    assert kwargs == output
    _, search = runner.complete_primitive_arguments(lambda **values: values, primitive)
    expected_search = materialize_global_real_details(primitive, config, exact_transform=False)[0]
    assert search == expected_search


@pytest.mark.parametrize("damage", ["out_of_bounds", "nan", "missing", "scalar"])
def test_candidate_rejects_invalid_or_incomplete_parameter_vector(damage):
    config = _config()
    inputs = _inputs(config)
    if damage == "out_of_bounds":
        inputs[config["variable_names"][-1]] = 0.051
    elif damage == "nan":
        inputs[config["variable_names"][1]] = float("nan")
    elif damage == "missing":
        del inputs[config["variable_names"][-1]]
    else:
        inputs[GLOBAL_X_INPUT_NAME] = 0.0
    with pytest.raises(ValueError):
        materialize_global_real_details(inputs, config)


@pytest.mark.parametrize("damage", [
    {"probe_enabled": True}, {"shap_priority_enabled": False},
    {"shap_sensitivity_enabled": True}, {"hybrid_de_enabled": True},
    {"bounds_mode": "strict"}, {"reference_transform": "pwl"},
    {"pwl_region_count": 1}, {"pwl_max_regions": 1},
    {"pwl_max_abs_error": float("nan")}, {"pwl_max_abs_error": 0.1},
    {"variable_names": [GLOBAL_X_INPUT_NAME]}, {"control_order": ["saturation", "contrast", "exposure"]},
])
def test_config_rejects_incompatible_mode_flags_and_mesh_metadata(damage):
    config = _config()
    with pytest.raises(ValueError):
        validate_global_real_config(dict(config, **damage))


def test_candidate_fails_closed_when_runtime_interpolation_error_exceeds_tolerance(monkeypatch):
    config = _config()
    inputs = _inputs(config, (0.013, -0.017, 0.021))
    original_cache = global_real._cached_multi

    def corrupted(*args):
        approximation = original_cache(*args)
        return replace(approximation, rgb_at_knots=approximation.rgb_at_knots + 0.02)

    monkeypatch.setattr(global_real, "_cached_multi", corrupted)
    with pytest.raises(ValueError, match="error tolerance"):
        materialize_global_real_details(inputs, config)
