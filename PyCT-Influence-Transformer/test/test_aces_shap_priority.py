"""Opt-in contracts for standalone ACES SHAP ranking."""
import pytest
import numpy as np
from types import SimpleNamespace
import multiprocessing
import os
import hashlib

from pyct.args import parse_args
from tasks.builders import global_real as builder
from libct.global_real import build_aces_like_global_real_config, build_concolic_global_real_kwargs, GLOBAL_X_INPUT_NAME
from libct.executor.child_protocol import ChildProtocol
from libct.searcher.constraint_sensitivity import SensitivityEstimator
from libct.explore import _requires_shap_comparator
import libct.explore as explore_module


def test_aces_shap_sensitivity_is_explicit_and_uses_alpha_zero():
    args = parse_args([
        "--dataset", "cifar10", "--attack-mode", "global-real",
        "--global-shift-kind", "aces-exposure", "--score-alpha", "0",
        "--global-real-shap-priority", "--global-real-shap-sensitivity",
        "--no-global-real-probe",
    ])
    assert args.global_real_shap_priority is True
    assert args.global_real_shap_sensitivity is True
    assert args.score_alpha == 0.0
    assert args.global_real_probe is False


@pytest.mark.parametrize("additional", [
    ["--global-real-shap-sensitivity"],
    ["--attack-mode", "hybrid-de", "--global-shift-kind", "aces-contrast", "--global-real-shap-priority"],
    ["--global-shift-kind", "brightness", "--global-real-shap-priority"],
    ["--attack-mode", "shap", "--global-real-shap-priority"],
])
def test_shap_options_reject_incompatible_pipeline(additional):
    with pytest.raises(SystemExit):
        parse_args([
            "--dataset", "cifar10", "--attack-mode", "global-real",
            "--global-shift-kind", "aces-exposure", "--score-alpha", "0",
            *additional,
        ])


def test_existing_aces_commands_leave_shap_ranking_disabled():
    args = parse_args([
        "--dataset", "cifar10", "--attack-mode", "global-real",
        "--global-shift-kind", "aces-exposure", "--score-alpha", "0.8",
    ])
    assert args.global_real_shap_priority is False
    assert args.global_real_shap_sensitivity is False


def test_aces_runtime_loads_comparator_only_when_priority_is_enabled():
    config = {"global_shift_kind": "aces-exposure", "transform_mode": "aces-like-pwl"}
    assert _requires_shap_comparator(config) is False
    assert _requires_shap_comparator(dict(config, shap_priority_enabled=True)) is True


@pytest.mark.parametrize("enabled", [False, True])
def test_builder_requests_float32_background_only_for_opt_in(monkeypatch, enabled):
    sample = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float32)
    background = np.stack([sample, sample])
    requested = []

    class Dataset:
        x_test = np.stack([sample])

        def get_cifar10_test_data(self, index):
            return {"v_0_0_0": 0.2, "v_0_0_1": 0.5, "v_0_0_2": 0.8}, {}

        def get_cifar10_test_data_and_set_condict(self, index, pixels):
            requested.append((index, pixels))
            return {}, {}, sample, background

    monkeypatch.setattr(builder, "Cifar10Dataset", Dataset)
    monkeypatch.setattr(builder, "get_save_dir_from_save_exp", lambda *a, **k: "unused")
    payload = builder.cifar10_global_real("demo", [0], force=True, shift_kind="aces-exposure",
        shap_priority_enabled=enabled, shap_sensitivity_enabled=enabled)[0]
    config = payload["global_real_config"]
    assert config["shap_priority_enabled"] is enabled
    assert config["shap_sensitivity_enabled"] is enabled
    if enabled:
        assert requested == [(0, [])]
        assert payload["input_for_shap"].dtype == np.float32
        np.testing.assert_array_equal(payload["input_for_shap"], sample)
        np.testing.assert_array_equal(payload["background_dataset_for_shap"], background)
    else:
        assert requested == []
        assert payload["background_dataset_for_shap"] is None


def test_each_concolic_forward_resets_sensitivity_at_current_candidate():
    sample = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float64)
    config = build_aces_like_global_real_config(sample, {
        "variable_name": GLOBAL_X_INPUT_NAME, "effective_min": -0.2, "effective_max": 0.2,
        "bounds_mode": "clip", "global_shift_kind": "aces-exposure", "pwl_max_segments": 64,
        "pwl_error_tolerance": 1 / 255, "shap_priority_enabled": True, "shap_sensitivity_enabled": True,
    })
    engine = SimpleNamespace(global_real_config=config, concolic_name_list=[], concolic_flag_dict={},
                             var_to_types={}, symbolic_enabled=True)
    inputs = {"v_0_0_0": 0.2, "v_0_0_1": 0.5, "v_0_0_2": 0.8, GLOBAL_X_INPUT_NAME: 0.0}
    build_concolic_global_real_kwargs(engine, inputs)
    old = engine._global_x_sensitivity
    old.estimate([">", old.variable, "0"])
    engine._shap_priority_stats["lookups"] = 9
    inputs[GLOBAL_X_INPUT_NAME] = 0.1
    build_concolic_global_real_kwargs(engine, inputs)
    assert engine._global_x_sensitivity is not old
    assert engine._global_x_sensitivity.x == 0.1
    assert engine._global_x_sensitivity.stats["attempted"] == 0
    assert engine._shap_priority_stats == {"lookups": 0, "nonzero": 0}


def _send_rank_stats(connection, soft_timeout):
    instance = SensitivityEstimator("X", 0.0, -0.2, 0.2)
    instance.estimate([">", "X", "0"])
    instance.estimate([">", "unknown", "0"])
    if soft_timeout:
        def interrupted(*args):
            raise TimeoutError("forward interrupted")
        instance._evaluate = interrupted
        try:
            instance.estimate([">", "X", "0"])
        except TimeoutError:
            pass
    engine = SimpleNamespace(var_to_types={}, concolic_name_list=[], concolic_flag_dict={},
        _global_x_sensitivity=instance, _shap_priority_stats={"lookups": 3, "nonzero": 2})
    protocol = ChildProtocol(engine)
    if soft_timeout:
        envelope = protocol.build_child_event_envelope(pid=os.getpid(), updated_args={}, result=None,
            event_type="soft_timeout", message="interrupted forward")
    else:
        envelope = protocol.build_child_ok_envelope(pid=os.getpid(), updated_args={}, result=0, constraint_payload=[])
    connection.send(envelope)
    connection.close()


@pytest.mark.parametrize("soft_timeout", [False, True])
def test_rank_telemetry_survives_real_child_process_and_aggregates(soft_timeout):
    context = multiprocessing.get_context("fork")
    recorder = SimpleNamespace(extra_meta={})
    engine = SimpleNamespace(var_to_types={}, concolic_name_list=[], concolic_flag_dict={}, _get_recorder=lambda: recorder)
    protocol = ChildProtocol(engine)
    for _ in range(2):
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_send_rank_stats, args=(sender, soft_timeout))
        process.start()
        sender.close()
        try:
            assert receiver.poll(5), "child did not send telemetry"
            envelope = receiver.recv()
            protocol.apply_child_shared_state({}, envelope)
            process.join(5)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            receiver.close()
    meta = recorder.extra_meta
    assert meta["global_real_shap_lookup_total"] == 6
    assert meta["global_real_shap_nonzero_total"] == 4
    stats = meta["global_real_sensitivity_stats"]
    assert stats["attempted"] == (6 if soft_timeout else 4)
    assert stats["evaluated"] == 2
    assert stats["fallback"] == 2
    assert stats["interrupted"] == (2 if soft_timeout else 0)
    assert stats["fallback_reasons"] == {"unknown_symbol": 2}
    assert stats["forwards"] == 2


def test_explore_initialization_preserves_required_cache_provenance(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(explore_module.Solver, "ctr_size", 0, raising=False)
    monkeypatch.setattr(explore_module, "recorder", explore_module.recorder)
    cache = tmp_path / "cache.json"
    cache.write_text("{}")
    cache_meta = {"target_class": 0, "case_index": 0}
    comparator = SimpleNamespace(cache_path=cache, cache_metadata=cache_meta, compare=lambda *a: 0)
    monkeypatch.setattr(explore_module, "ShapValuesComparator", lambda **kw: comparator)
    monkeypatch.setattr(explore_module, "solver_variable_bounds", lambda config: {})
    monkeypatch.setattr(explore_module, "ConcolicTestRecorder", lambda *a: SimpleNamespace(
        extra_meta={}, total_iter=0, end=lambda **kw: None))
    engine = explore_module.ExplorationEngine.__new__(explore_module.ExplorationEngine)
    engine.save_dir, engine.input_name, engine.statsdir = None, "case_0", None
    engine.symbolic_path_threshold = 2000
    monkeypatch.setattr(engine, "_can_use_concolic_wrapper", lambda *a: False)
    observed = []
    monkeypatch.setattr(engine, "_execution_loop", lambda *a, **k: observed.append(dict(engine._get_recorder().extra_meta)) or False)
    _, recorder = engine.explore("predict", {"v_0_0_0": 0.2}, root=str(tmp_path),
        input_for_shap=np.zeros((1, 1, 3), dtype=np.float32), background_dataset_for_shap=np.zeros((1, 1, 1, 3)),
        global_real_config={"global_shift_kind": "aces-exposure", "shap_priority_enabled": True,
            "effective_min": -0.2, "effective_max": 0.2, "variable_name": GLOBAL_X_INPUT_NAME}, idx=0)
    assert observed[0]["global_real_shap_cache_metadata"] == cache_meta
    assert recorder.extra_meta["global_real_shap_cache_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
