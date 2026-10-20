from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from libct.aces_like_multi import apply_aces_like_multi_transform
from libct.constraint import Constraint
from libct.executor import ConcolicArgumentBuilder
from libct.global_real import (
    GLOBAL_X_INPUT_NAME, GLOBAL_X_SMT_NAME, build_aces_like_global_real_config,
    solver_variable_bounds,
)
from libct.predicate import Predicate
from libct.record import ConcolicTestRecorder
from libct.solver import Solver
import libct.solver as solver_module
from pyct.args import parse_args
from tasks.builders import global_real as builder


RGB = np.asarray([[[0.2, 0.5, 0.8]]])
BASE_CLI = ["--dataset", "cifar10", "--attack-mode", "global-real",
            "--global-shift-kind", "aces-multi"]


def _runtime(controls=("exposure", "contrast", "saturation")):
    config = build_aces_like_global_real_config(RGB, {
        "global_shift_kind": "aces-multi", "controls": list(controls),
        "control_bounds": [[-0.02, 0.03], [-0.03, 0.04], [-0.04, 0.05]][:len(controls)],
        "effective_min": -0.2, "effective_max": 0.2,
        "requested_min": -0.2, "requested_max": 0.2,
        "pwl_max_regions": 512, "pwl_error_tolerance": 1 / 255,
        "bounds_mode": "clip", "probe_enabled": False,
        "shap_priority_enabled": True, "shap_sensitivity_enabled": False,
    })
    inputs = {"v_0_0_" + str(i): float(value) for i, value in enumerate(RGB[0, 0])}
    inputs.update(dict.fromkeys(config["variable_names"], 0.0))
    return config, inputs


def test_cli_multi_defaults_select_shap_only_no_probe_and_independent_ranges():
    args = parse_args(BASE_CLI + ["--aces-controls", "saturation", "exposure", "contrast"])
    assert args.aces_controls == ("exposure", "contrast", "saturation")
    assert args.aces_control_bounds == ((-0.2, 0.2),) * 3
    assert args.aces_pwl_max_regions == 512
    assert args.global_real_shap_priority is True
    assert args.global_real_shap_sensitivity is False
    assert args.global_real_probe is False
    assert args.score_alpha == 0.0


def test_cli_per_control_overrides_follow_names_after_canonicalization():
    args = parse_args(BASE_CLI + [
        "--aces-controls", "curves", "exposure", "--aces-control-bounds=curves:-0.1:0.3",
        "--aces-control-bounds=exposure:-0.4:0.2", "--aces-pwl-max-regions", "128",
    ])
    assert args.aces_controls == ("exposure", "curves")
    assert args.aces_control_bounds == ((-0.4, 0.2), (-0.1, 0.3))
    assert args.aces_pwl_max_regions == 128


@pytest.mark.parametrize("options", [
    [], ["--aces-controls", "exposure"], ["--aces-controls", "exposure", "exposure"],
    ["--aces-controls", "exposure", "brightness"],
    ["--aces-controls", "exposure", "contrast", "saturation", "curves"],
    ["--aces-controls", "exposure", "contrast", "--global-real-shap-sensitivity"],
    ["--aces-controls", "exposure", "contrast", "--score-alpha", "0.1"],
    ["--aces-controls", "exposure", "contrast", "--attack-mode", "hybrid-de"],
    ["--aces-controls", "exposure", "contrast", "--global-x-bounds-mode", "strict"],
    ["--aces-controls", "exposure", "contrast", "--aces-control-bounds=curves:-0.1:0.1"],
    ["--aces-controls", "exposure", "contrast", "--aces-control-bounds=exposure:0.1:0.2"],
    ["--aces-controls", "exposure", "contrast", "--aces-control-bounds=exposure:nan:0.2"],
    ["--aces-controls", "exposure", "contrast", "--aces-control-bounds=exposure:-0.1:0.1",
     "--aces-control-bounds=exposure:-0.2:0.2"],
    ["--aces-controls", "exposure", "curves", "--aces-control-bounds=curves:-1:0.2"],
    ["--aces-controls", "exposure", "contrast", "--aces-pwl-max-regions", "0"],
])
def test_cli_rejects_invalid_multi_modes_controls_and_domains(options):
    with pytest.raises(SystemExit) as error:
        parse_args(BASE_CLI + options)
    assert error.value.code == 2


def test_scalar_cli_rejects_multi_only_options():
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "cifar10", "--attack-mode", "global-real",
                    "--global-shift-kind", "aces-exposure", "--score-alpha", "0",
                    "--aces-controls", "exposure", "contrast"])


def _patch_dataset(monkeypatch):
    background = np.stack([RGB, RGB]).astype(np.float32)

    class Dataset:
        x_test = np.stack([RGB])

        def get_cifar10_test_data(self, index):
            return {"v_0_0_" + str(i): float(v) for i, v in enumerate(RGB[0, 0])}, {}

        def get_cifar10_test_data_and_set_condict(self, index, pixels):
            return {}, {}, RGB.astype(np.float32), background

    monkeypatch.setattr(builder, "Cifar10Dataset", Dataset)
    monkeypatch.setattr(builder, "get_save_dir_from_save_exp", lambda *args, **kwargs: "unused")
    monkeypatch.setattr(builder, "TargetClassInputShapProvider",
                        lambda *args, **kwargs: pytest.fail("builder must use cached SHAP downstream"))
    return background


def test_builder_creates_only_selected_axes_and_preserves_named_bounds(monkeypatch):
    background = _patch_dataset(monkeypatch)
    payloads = builder.cifar10_global_real(
        "offline", [0], force=True, shift_kind="aces-multi", controls=("saturation", "exposure"),
        control_bounds=((-0.03, 0.04), (-0.01, 0.02)),
        probe_enabled=False, shap_priority_enabled=True,
    )
    assert len(payloads) == 1
    payload = payloads[0]
    config = payload["global_real_config"]
    assert config["controls"] == ["exposure", "saturation"]
    assert config["control_bounds"] == [[-0.01, 0.02], [-0.03, 0.04]]
    assert config["probe_enabled"] is False
    assert config["shap_priority_enabled"] is True
    assert config["shap_sensitivity_enabled"] is False
    assert payload["con_dict"] == {name: 1 for name in config["variable_names"]}
    assert all(payload["in_dict"][name] == 0.0 for name in config["variable_names"])
    assert GLOBAL_X_INPUT_NAME not in payload["in_dict"]
    assert payload["input_for_shap"].dtype == np.float32
    np.testing.assert_array_equal(payload["background_dataset_for_shap"], background)


@pytest.mark.parametrize("settings", [
    {"probe_enabled": True}, {"shap_priority_enabled": False},
    {"shap_sensitivity_enabled": True}, {"defer_pwl": True}, {"bounds_mode": "strict"},
])
def test_builder_rejects_incompatible_pipeline_flags_before_loading_data(monkeypatch, settings):
    monkeypatch.setattr(builder, "Cifar10Dataset", lambda: pytest.fail("invalid settings must fail early"))
    arguments = dict(force=True, shift_kind="aces-multi", controls=("exposure", "contrast"),
                     control_bounds=((-0.02, 0.02),) * 2,
                     probe_enabled=False, shap_priority_enabled=True)
    arguments.update(settings)
    with pytest.raises(ValueError):
        builder.cifar10_global_real("offline", [0], **arguments)


def test_last_solver_query_keeps_every_axis_bound_and_model_binding(monkeypatch, tmp_path):
    config, inputs = _runtime()
    engine = SimpleNamespace(global_real_config=config, counter=20, var_to_types={}, idx=0,
                             constraints_collection_type="priority_queue",
                             concolic_name_list=[], concolic_flag_dict={})
    ConcolicArgumentBuilder(engine).build(
        lambda **values: values, inputs, dict.fromkeys(config["variable_names"], 1)
    )
    engine.solver_variable_bounds = solver_variable_bounds(config)
    monkeypatch.setattr(Constraint, "global_constraints", [])
    first_name, _, last_name = engine.concolic_name_list
    first = Predicate([">", first_name, "0"], True)
    last = Predicate(["<", last_name, "0"], True)
    path = Constraint(None, None).add_child(first).add_child(last)
    monkeypatch.setenv("PYCT_SMT_PATH_MODE", "last")
    monkeypatch.setenv("PYCT_SMT_EXPERIMENT_MODE", "raw")
    for name, value in dict(norm=True, limit_change_range=None, build_timeout_enabled=False,
                            run_timeout=1, smtdir=None, store=None, iter=1, iter_count=1, cnt=1,
                            cmd=["offline-solver"], ctr_size={},
                            stats=dict(sat_number=0, sat_time=0, unsat_number=0, unsat_time=0,
                                       otherwise_number=0, otherwise_time=0)).items():
        monkeypatch.setattr(Solver, name, value, raising=False)
    monkeypatch.setattr(Solver, "_resolve_constraint_log_path", lambda *args: tmp_path / "solver.log")
    monkeypatch.setattr(Solver, "_append_constraint_log", lambda *args: None)
    captured = []

    def solve(command, *, input, **kwargs):
        captured.append(input.decode())
        bindings = "\n".join("(({} {}))".format(name, "(- (/ 1 100))" if i == 2 else "(/ 1 100)")
                            for i, name in enumerate(engine.concolic_name_list))
        return SimpleNamespace(stdout=("sat\n" + bindings + "\n").encode(), stderr=b"", returncode=0)

    monkeypatch.setattr(solver_module.subprocess, "run", solve)
    model = Solver.find_model_from_constraint(engine, path, 0.0, (0, 0), 0, inputs)
    assert model == dict(zip(config["variable_names"], (0.01, 0.01, -0.01)))
    assert len(captured) == 1
    formula = captured[0]
    assert first.get_formula() not in formula
    assert last.get_formula() in formula
    assert GLOBAL_X_SMT_NAME not in formula
    for name, (lo, hi) in engine.solver_variable_bounds.items():
        assert "(declare-const {} Real)".format(name) in formula
        assert "(<= {} {:.15f})".format(name, hi) in formula
        assert "(>= {} (- {:.15f}))".format(name, -lo) in formula
    assert Solver.ctr_size["detail"][0]["smt_transform"]["retained_assertion_count"] == 1


@pytest.mark.parametrize("controls", [("exposure", "contrast"), ("exposure", "contrast", "saturation")])
def test_recorder_saves_all_named_params_exact_images_and_dimensioned_sat_artifacts(tmp_path, controls):
    config, inputs = _runtime(controls)
    params = (0.013, -0.017, 0.021)[:len(controls)]
    inputs.update(zip(config["variable_names"], params))
    recorder = ConcolicTestRecorder(str(tmp_path), "case_0")
    recorder.input_shape = RGB.shape
    recorder.global_real_config = config
    recorder.save_original_input({**inputs, **dict.fromkeys(config["variable_names"], 0.0)})
    recorder.save_sat_input(inputs)
    recorder.find_adversarial_input(inputs, attack_label=2)
    recorder.save_stats_dict()
    expected = apply_aces_like_multi_transform(RGB, params, controls=controls).rgb.astype(np.float32)
    np.testing.assert_array_equal(np.load(tmp_path / "adv_input.npy"), expected)
    np.testing.assert_array_equal(np.load(tmp_path / "sat_inputs.npy"), expected[None])
    np.testing.assert_array_equal(np.load(tmp_path / "sat_aces_params.npy"), [params])
    metadata = json.loads((tmp_path / "stats.json").read_text())["meta"]
    assert metadata["global_real_solved_params"] == dict(zip(controls, params))
    assert metadata["global_real_last_sat_params"] == dict(zip(controls, params))
    assert "global_real_solved_x" not in metadata
    assert "global_real_solved_brightness" not in metadata
    assert not (tmp_path / "sat_global_x.npy").exists()
    assert not (tmp_path / "sat_hybrid_bc.npy").exists()


def test_recipe_cli_preserves_fixed_defaults_and_supports_staged_defaults():
    cli = BASE_CLI + ["--aces-controls", "exposure", "contrast", "saturation"]
    fixed = parse_args(cli)
    assert fixed.aces_recipe_policy == "fixed"
    assert fixed.aces_pwl_max_regions == 512
    assert fixed.aces_pwl_region_budgets is None
    staged = parse_args(cli + ["--aces-recipe-policy", "staged"])
    assert staged.aces_recipe_policy == "staged"
    assert staged.aces_pwl_region_budgets == (512, 1024, 2048)
    assert staged.aces_pwl_max_regions == 2048
    assert staged.aces_pwl_build_timeout == 30.0
    assert staged.score_alpha == 0.0
    assert staged.global_real_shap_priority is True
    assert staged.global_real_probe is False


def test_recipe_cli_custom_budget_ladder_accepts_matching_explicit_cap():
    args = parse_args(BASE_CLI + [
        "--aces-controls", "exposure", "contrast", "--aces-recipe-policy", "staged",
        "--aces-pwl-region-budgets", "12", "24", "48", "--aces-pwl-max-regions", "48",
        "--aces-pwl-build-timeout", "2.5",
    ])
    assert args.aces_pwl_region_budgets == (12, 24, 48)
    assert args.aces_pwl_max_regions == 48
    assert args.aces_pwl_build_timeout == 2.5


@pytest.mark.parametrize("options", [
    ["--aces-pwl-region-budgets", "16", "16"],
    ["--aces-pwl-region-budgets", "32", "16"],
    ["--aces-pwl-region-budgets", "0", "16"],
    ["--aces-pwl-region-budgets", "-1", "16"],
    ["--aces-pwl-region-budgets", "1.5", "16"],
    ["--aces-pwl-max-regions", "512"],
    ["--aces-pwl-region-budgets", "16", "32", "--aces-pwl-max-regions", "64"],
    ["--aces-pwl-build-timeout", "0"],
    ["--aces-pwl-build-timeout", "inf"],
    ["--aces-pwl-build-timeout", "nan"],
    ["--global-real-shap-sensitivity"],
    ["--score-alpha", "0.1"],
    ["--attack-mode", "hybrid-de"],
    ["--global-x-bounds-mode", "strict"],
])
def test_recipe_cli_rejects_invalid_budgets_and_incompatible_modes(options):
    with pytest.raises(SystemExit) as error:
        parse_args(BASE_CLI + ["--aces-controls", "exposure", "contrast",
                               "--aces-recipe-policy", "staged"] + options)
    assert error.value.code == 2


@pytest.mark.parametrize("options", [
    ["--aces-pwl-region-budgets", "16", "32"],
    ["--aces-pwl-build-timeout", "2"],
])
def test_recipe_cli_rejects_staged_budgets_for_fixed_policy(options):
    with pytest.raises(SystemExit):
        parse_args(BASE_CLI + ["--aces-controls", "exposure", "contrast"] + options)


def test_staged_builder_preserves_target_shap_and_defers_every_pwl_build(monkeypatch):
    background = _patch_dataset(monkeypatch)
    monkeypatch.setattr(builder, "build_aces_like_global_real_config",
                        lambda *args, **kwargs: pytest.fail("PWL must build inside the bounded recipe child"))
    payload, = builder.cifar10_global_real(
        "offline", [0], force=True, shift_kind="aces-multi", recipe_policy="staged",
        controls=("saturation", "exposure", "contrast"),
        control_bounds=((-0.03, 0.04), (-0.01, 0.02), (-0.02, 0.03)),
        probe_enabled=False, shap_priority_enabled=True,
        pwl_region_budgets=(8, 16, 32), pwl_build_timeout_seconds=2.5,
    )
    assert payload["aces_recipe_plan"] == {
        "policy": "staged", "controls": ["exposure", "contrast", "saturation"],
        "control_bounds": [[-0.01, 0.02], [-0.02, 0.03], [-0.03, 0.04]],
        "region_budgets": [8, 16, 32], "pwl_build_timeout_seconds": 2.5,
    }
    assert "pwl_table" not in payload["global_real_config"]
    assert payload["shap_value_pre_calculated"] is True
    assert payload["input_for_shap"].dtype == np.float32
    np.testing.assert_array_equal(payload["input_for_shap"], RGB.astype(np.float32))
    np.testing.assert_array_equal(payload["background_dataset_for_shap"], background)


@pytest.mark.parametrize("settings", [
    {"pwl_region_budgets": []}, {"pwl_region_budgets": [8, 8]},
    {"pwl_region_budgets": [16, 8]}, {"pwl_region_budgets": [0, 8]},
    {"pwl_region_budgets": [True, 8]}, {"pwl_region_budgets": [1.5, 8]},
    {"pwl_build_timeout_seconds": 0}, {"pwl_build_timeout_seconds": float("inf")},
    {"pwl_build_timeout_seconds": float("nan")},
])
def test_staged_builder_rejects_bad_budgets_before_dataset_loading(monkeypatch, settings):
    monkeypatch.setattr(builder, "Cifar10Dataset", lambda: pytest.fail("bad budgets must fail before data loading"))
    with pytest.raises(ValueError):
        builder.cifar10_global_real(
            "offline", [0], force=True, shift_kind="aces-multi", recipe_policy="staged",
            controls=("exposure", "contrast"), control_bounds=((-0.02, 0.02),) * 2,
            probe_enabled=False, shap_priority_enabled=True, **settings,
        )


def test_runner_routes_staged_payload_to_recipe_supervisor(monkeypatch):
    from orchestration import aces_recipes
    from orchestration.runners import BaseRunner

    calls = []
    expected = (0, SimpleNamespace(extra_meta={"status": "invalid"}))
    monkeypatch.setattr(aces_recipes, "run_recipes",
                        lambda payload, **options: calls.append((payload, options)) or expected)
    runner = BaseRunner(timeout=17, norm=True, collect_constraints_with="priority_queue",
                        constraint_build_timeout_seconds=4, solver_run_timeout=2)
    payload = {"aces_recipe_plan": {"policy": "staged"}}
    assert runner._execute_attack(payload) is expected
    assert calls == [(payload, {
        "timeout": 17, "norm": True, "constraint_build_timeout": True,
        "constraint_build_timeout_seconds": 4, "solver_run_timeout": 2,
        "collect_constraints_with": "priority_queue",
    })]
