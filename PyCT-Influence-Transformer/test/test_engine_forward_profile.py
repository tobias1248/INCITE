"""Regression coverage for the single-case forward diagnostic."""
from __future__ import annotations

from types import SimpleNamespace
import cProfile
import json
import math
import pstats
import signal
import sys
import time

import numpy as np
import pytest

from libct.constraint import Constraint
from libct.concolic import Concolic
from libct.utils import ConcolicObject


@pytest.fixture
def isolated_constraints(monkeypatch):
    """Profiling must not leak its diagnostic path tree into other tests."""
    previous = Constraint.global_constraints
    Constraint.global_constraints = []
    import libct.position as position
    monkeypatch.setattr(position, "current_layer_number", 0, raising=False)
    monkeypatch.setattr(position, "current_indices_in_current_layer", (0,), raising=False)
    try:
        yield
    finally:
        Constraint.global_constraints = previous


@pytest.mark.parametrize("record_path", [True, False])
def test_actual_symbolic_branches_stop_at_the_same_threshold(
    isolated_constraints, record_path
):
    from engine.forward_profile import ProfileBranchEngine

    engine = ProfileBranchEngine(threshold=3, record_path=record_path)
    value = ConcolicObject(0.0, "X_VAR", engine)
    outcomes = [bool(value < bound) for bound in (1.0, -1.0, 2.0, 3.0)]

    assert outcomes == [True, False, True, True]
    assert engine.branch_count == 3
    assert engine.symbolic_enabled is False
    assert engine.symbolic_disabled_at_path_len == 3
    assert not isinstance(value + 1.0, Concolic)
    if record_path:
        assert engine.path.current_constraint.height == 3
        assert len(engine.constraints_to_solve) == 3
        assert len(Constraint.global_constraints) == 7
    else:
        assert len(engine.constraints_to_solve) == 0
        assert len(Constraint.global_constraints) == 0


def test_no_path_creates_no_constraint_nodes_or_work_items(
    isolated_constraints, monkeypatch
):
    from engine.forward_profile import ProfileBranchEngine

    def forbidden(*args, **kwargs):
        pytest.fail("no-path diagnostic created a path node or solver work item")

    monkeypatch.setattr(Constraint, "__init__", forbidden)
    engine = ProfileBranchEngine(threshold=5, record_path=False)
    monkeypatch.setattr(engine, "push_constraint", forbidden)
    value = ConcolicObject(0.0, "X_VAR", engine)
    assert bool(value >= -1.0)
    assert not bool(value > 1.0)
    assert engine.branch_count == 2
    assert engine.symbolic_enabled is True


class _ToyLayer:
    def forward(self, value):
        return np.asarray(value) + 1.0


def _toy_model():
    layers = [_ToyLayer(), _ToyLayer()]

    def forward(value):
        for layer in layers:
            value = layer.forward(value)
        return value

    return SimpleNamespace(layers=layers, forward=forward)


def test_profile_callable_records_layers_and_restores_methods(tmp_path):
    from engine.forward_profile import profile_callable

    model = _toy_model()
    before = [layer.forward for layer in model.layers]
    events = []
    result = profile_callable(
        lambda: int(np.argmax(model.forward([0.0, 3.0]))), model=model,
        timeout=1.0, emit=events.append, profile_path=tmp_path / "forward.prof",
    )

    assert result["status"] == "completed"
    assert result["label"] == 1
    assert result["wall_time"] >= 0.0
    assert result["cpu_time"] >= 0.0
    assert [layer["index"] for layer in result["layers"]] == [0, 1]
    assert all(layer["status"] == "completed" for layer in result["layers"])
    assert all(layer["wall_time"] >= 0.0 for layer in result["layers"])
    assert [layer.forward for layer in model.layers] == before
    assert all("forward" not in vars(layer) for layer in model.layers)
    assert (tmp_path / "forward.prof").is_file()
    assert events


def test_profile_callable_error_preserves_partial_layers_and_restores_methods():
    from engine.forward_profile import profile_callable

    class FailingLayer:
        def forward(self, value):
            raise ValueError("broken layer")

    model = SimpleNamespace(layers=[FailingLayer()])
    previous = model.layers[0].forward
    result = profile_callable(
        lambda: model.layers[0].forward([0.0]), model=model, timeout=1.0
    )

    assert result["status"] == "error"
    assert "broken layer" in result["error"]
    assert len(result["layers"]) == 1
    assert result["layers"][0]["status"] == "error"
    assert model.layers[0].forward == previous
    assert "forward" not in vars(model.layers[0])


def test_timeout_preserves_completed_and_active_layers_and_signal_handler():
    from engine.forward_profile import profile_callable

    class WaitingLayer:
        def forward(self, value):
            time.sleep(1.0)
            return value

    model = _toy_model()
    model.layers[1] = WaitingLayer()
    before = [layer.forward for layer in model.layers]
    previous_handler = signal.getsignal(signal.SIGALRM)
    result = profile_callable(
        lambda: int(np.argmax(model.forward([0.0, 3.0]))),
        model=model, timeout=0.03,
    )

    assert result["status"] == "timeout"
    assert result["label"] is None
    assert [row["status"] for row in result["layers"]] == ["completed", "timeout"]
    assert result["layers"][1]["wall_time"] > 0.0
    assert [layer.forward for layer in model.layers] == before
    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0.0


def test_swallowed_deadline_cannot_return_a_completed_label():
    from engine.forward_profile import profile_callable

    def legacy_forward():
        try:
            time.sleep(1.0)
        except BaseException:
            # Legacy arithmetic can swallow even the profiler's deadline.
            return 4
        pytest.fail("deadline was never delivered")

    result = profile_callable(legacy_forward, timeout=0.02)
    assert result["status"] == "timeout"
    assert result["label"] is None
    assert result["wall_time"] >= 0.02
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0.0


def test_existing_instance_forward_is_restored_after_failure():
    from engine.forward_profile import profile_callable

    layer = _ToyLayer()

    def overridden(value):
        raise RuntimeError("instance override")

    layer.forward = overridden
    result = profile_callable(lambda: layer.forward([0.0]), model=SimpleNamespace(layers=[layer]))
    assert result["status"] == "error"
    assert vars(layer)["forward"] is overridden


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_profile_rejects_invalid_timeout(timeout):
    from engine.forward_profile import profile_callable

    with pytest.raises(ValueError, match="finite and positive"):
        profile_callable(lambda: 0, timeout=timeout)


def test_self_time_groups_partition_nested_profile_without_double_counting():
    from engine.forward_profile import summarize_profile

    namespace = {}
    exec(compile(
        "def symbolic_work():\n    return sum(i * i for i in range(1000))\n",
        "/diagnostic/libct/concolic/float.py", "exec"), namespace)
    exec(compile(
        "def model_forward():\n    return symbolic_work() + symbolic_work()\n",
        "/diagnostic/dnnct/myDNN.py", "exec"), namespace)
    profiler = cProfile.Profile()
    profiler.runcall(namespace["model_forward"])
    summary = summarize_profile(profiler)
    own_total = sum(entry[2] for entry in pstats.Stats(profiler).stats.values())
    grouped_total = sum(group["self_time"] for group in summary["groups"].values())
    assert math.isclose(grouped_total, own_total, rel_tol=1e-9, abs_tol=1e-12)
    assert summary["groups"]["symbolic_arithmetic_and_objects"]["calls"] > 0
    assert summary["groups"]["model_python"]["calls"] > 0
    assert all(row["self_time"] <= row["cumulative_time"] for row in summary["top_functions"])


def _unfinished_worker(connection, mode, options, directory):
    """A worker exits after sending progress, without a final label/profile."""
    connection.send({"event": "phase_start", "phase": "forward"})
    connection.send({"event": "layer_start", "index": 0, "type": "WaitingLayer", "status": "running"})
    connection.close()


def test_parent_persists_partial_progress_when_worker_exits(tmp_path):
    from engine.forward_profile import run_mode

    output = tmp_path / "mode"
    result = run_mode(
        "symbolic_path", {"timeout": 1.0, "setup_timeout": 5.0}, output,
        worker=_unfinished_worker,
    )
    assert result["status"] == "error"
    assert result["hard_timeout"] is False
    assert result["label"] is None
    assert result["profile_groups"] is None
    assert result["layers"][0]["status"] == "interrupted"
    assert result["phase"] == "forward"
    assert json.loads((output / "result.json").read_text())["status"] == "error"
    assert "WaitingLayer" in (output / "events.jsonl").read_text()


def _stalled_worker(connection, mode, options, directory):
    connection.send({"event": "phase_end", "phase": "dataset_load", "wall_time": 0.01})
    connection.send({"event": "phase_start", "phase": "forward"})
    connection.send({"event": "layer_start", "index": 0, "type": "StalledLayer", "status": "running"})
    # Simulate a native forward that ignores the child's Python deadline.
    time.sleep(30)


def test_parent_watchdog_terminates_stalled_forward_and_keeps_partial_events(tmp_path):
    from engine.forward_profile import run_mode

    result = run_mode(
        "symbolic_no_path", {"timeout": 0.01, "setup_timeout": 5.0},
        tmp_path / "stalled", worker=_stalled_worker,
    )
    assert result["status"] == "timeout"
    assert result["hard_timeout"] is True
    assert result["label"] is None
    assert result["wall_time"] is None
    assert result["profile_groups"] is None
    assert result["layers"][0]["status"] == "timeout"
    assert result["phases"][0]["phase"] == "dataset_load"


@pytest.mark.parametrize("mismatch", [None, "timeout", "label", "branches", "disabled"])
def test_summary_only_compares_finished_equivalent_symbolic_forwards(
    tmp_path, monkeypatch, mismatch
):
    import engine.forward_profile as profile

    def fake_run(mode, options, directory):
        row = {
            "mode": mode, "status": "completed", "label": 2,
            "branch_count": 3, "symbolic_disabled_at_path_len": 3,
            "wall_time": 10.0 if mode == "symbolic_path" else 2.0,
        }
        if mode == "symbolic_no_path":
            if mismatch == "timeout":
                row.update(status="timeout", label=None)
            elif mismatch == "label":
                row["label"] = 1
            elif mismatch == "branches":
                row["branch_count"] = 2
            elif mismatch == "disabled":
                row["symbolic_disabled_at_path_len"] = None
        return row

    monkeypatch.setattr(profile, "run_mode", fake_run)
    model_path = tmp_path / "model.h5"
    model_path.write_bytes(b"cached model fixture")
    report = profile.run_profile_case({"model_path": str(model_path)}, tmp_path / "report")
    assert report["symbolic_comparable"] is (mismatch is None)
    if mismatch is None:
        assert report["symbolic_path_speedup"] == 5.0
    else:
        assert report["symbolic_path_speedup"] is None


def test_cli_defaults_keep_one_case_and_original_threshold():
    from pyct.profile_forward import parse_args

    args = parse_args([])
    assert args.model_name == "cifar10_concolic_transformer"
    assert args.case_index == 0
    assert args.timeout == 1800
    assert args.symbolic_path_threshold == 2000
    assert args.mode == "all"
    assert args.global_shift_kind == "aces-contrast"


@pytest.mark.parametrize("mode", [
    "all", "keras_concrete", "nnmodel_concrete", "symbolic_path", "symbolic_no_path",
])
@pytest.mark.parametrize("control", ["aces-brightness", "aces-contrast"])
def test_cli_accepts_selected_mode_and_supported_control(mode, control):
    from pyct.profile_forward import parse_args

    args = parse_args(["--mode", mode, "--global-shift-kind", control])
    assert args.mode == mode
    assert args.global_shift_kind == control


@pytest.mark.parametrize("argv", [
    ["--mode", "invalid"], ["--mode", "symbolic"],
    ["--global-shift-kind", "brightness"], ["--global-shift-kind", "aces-exposure"],
])
def test_cli_rejects_unsupported_mode_and_control(argv):
    from pyct.profile_forward import parse_args

    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def test_selected_mode_runs_once_and_records_brightness_provenance(tmp_path, monkeypatch):
    import engine.forward_profile as profile

    launched = []

    def fake_run(mode, options, directory):
        launched.append((mode, options["global_shift_kind"]))
        return {
            "mode": mode, "status": "completed", "label": 2, "wall_time": 1.0,
            "branch_count": 3, "symbolic_disabled_at_path_len": 3,
            "global_shift_kind": options["global_shift_kind"],
        }

    monkeypatch.setattr(profile, "run_mode", fake_run)
    model_path = tmp_path / "model.h5"
    model_path.write_bytes(b"cached model fixture")
    output = tmp_path / "selected"
    report = profile.run_profile_case({
        "model_path": str(model_path), "mode": "symbolic_path",
        "global_shift_kind": "aces-brightness",
    }, output)

    assert launched == [("symbolic_path", "aces-brightness")]
    assert len(report["results"]) == 1
    assert report["manifest"]["fixed_settings"]["shift_kind"] == "aces-brightness"
    assert report["symbolic_comparable"] is False
    assert report["symbolic_path_speedup"] is None
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["fixed_settings"]["shift_kind"] == "aces-brightness"


@pytest.mark.parametrize("options", [
    {"mode": "invalid", "global_shift_kind": "aces-brightness"},
    {"mode": "symbolic_path", "global_shift_kind": "aces-exposure"},
])
def test_profile_options_are_validated_before_artifact_creation(tmp_path, options):
    from engine.forward_profile import run_profile_case

    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_profile_case(options, output)
    assert not output.exists()


@pytest.mark.parametrize("control", ["aces-brightness", "aces-contrast"])
def test_worker_builds_requested_control_and_reports_same_metadata(
    tmp_path, monkeypatch, isolated_constraints, control
):
    import engine
    import engine.forward_profile as profile
    import libct.global_real as global_real
    from libct.solver import Solver

    built_controls = []
    real_builder = global_real.build_aces_like_global_real_config

    def recording_builder(image, config):
        built_controls.append(config["global_shift_kind"])
        return real_builder(image, config)

    predictor = SimpleNamespace(
        init_reference_model=lambda *args, **kwargs: None,
        init_model=lambda *args, **kwargs: None,
        predict_search=lambda **kwargs: 2,
        searchModel=SimpleNamespace(layers=[]),
    )
    monkeypatch.setitem(sys.modules, "engine.predictor_runtime", predictor)
    monkeypatch.setattr(engine, "predictor_runtime", predictor, raising=False)
    monkeypatch.setattr(global_real, "build_aces_like_global_real_config", recording_builder)
    monkeypatch.setattr(profile, "_load_case", lambda *args: (np.full((1, 1, 3), 0.5), 2))
    monkeypatch.setattr(Solver, "safety", getattr(Solver, "safety", 0), raising=False)
    events = []
    connection = SimpleNamespace(send=events.append, close=lambda: None)
    profile._profile_worker(connection, "symbolic_path", {
        "dataset_cache": "unused_cache", "model_path": "unused_model",
        "case_index": 0, "path_threshold": 2000, "timeout": 1.0,
        "mode": "symbolic_path", "global_shift_kind": control,
    }, str(tmp_path))

    assert built_controls == [control]
    result = next(event["result"] for event in events if event["event"] == "result")
    assert result["status"] == "completed"
    assert result["global_shift_kind"] == control
    assert result["label"] == 2


@pytest.mark.parametrize("argv", [
    ["--case-index", "-1"], ["--case-index", "10000"],
    ["--timeout", "0"], ["--timeout", "nan"], ["--timeout", "inf"],
    ["--setup-timeout", "-1"], ["--symbolic-path-threshold", "0"],
    ["--model-name", "../outside"],
])
def test_cli_rejects_invalid_case_deadlines_and_model_paths(argv):
    from pyct.profile_forward import parse_args

    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def test_cli_help_works_before_loading_models(capsys):
    from pyct.profile_forward import parse_args

    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"])
    assert exc.value.code == 0
    assert "--case-index" in capsys.readouterr().out

