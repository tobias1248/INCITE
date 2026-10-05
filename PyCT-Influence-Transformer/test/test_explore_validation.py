from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import libct.explore as explore
from libct.executor import CandidateExecutionRunner
from libct.record import ConcolicTestRecorder
from libct.global_real import GLOBAL_X_INPUT_NAME, build_aces_like_global_real_config


class _RecorderStub:
    def __init__(self) -> None:
        self.original_label = None
        self.attack_label = None
        self.original_input = None
        self.gen_constraint = []
        self.extra_meta = {}
        self.total_iter = -1
        self.queue_max = 0
        self.queue_last = 0
        self.solve_all_ctr = False
        self.no_ctr_calls = 0
        self.child_events = []

    def start(self) -> None:
        return None

    def iter_start(self, _solver) -> None:
        return None

    def execution_start(self) -> None:
        return None

    def execution_end(self) -> None:
        return None

    def iter_end(self, _solver_stats, _solve_constr_num) -> None:
        self.total_iter += 1

    def solve_constr_start(self) -> None:
        return None

    def solve_constr_end(self) -> None:
        return None

    def first_execution_end(self) -> None:
        return None

    def save_original_input(self, inputs) -> None:
        self.original_input = dict(inputs)

    def save_stats_dict(self, constraint_complexity=None) -> None:
        return None

    def save_sat_input(self, _inputs) -> None:
        return None

    def record_reference_prediction(self, _wall_time, *, phase) -> None:
        counts = self.extra_meta.setdefault("reference_prediction_phase_counts", {})
        counts[phase] = counts.get(phase, 0) + 1

    def find_adversarial_input(self, inputs, attack_label) -> None:
        self.attack_label = attack_label
        self.adversarial_input = dict(inputs)

    def total_timeout(self) -> None:
        return None

    def no_ctr_to_solve(self) -> None:
        self.solve_all_ctr = True
        self.no_ctr_calls += 1

    def mark_error(self, error_type, reason, *, phase=None, child_pid=None, event_type=None) -> None:
        self.extra_meta["status"] = "error"
        self.extra_meta["error_type"] = error_type
        self.extra_meta["error_reason"] = reason
        if phase is not None:
            self.extra_meta["error_phase"] = phase
        if child_pid is not None:
            self.extra_meta["child_pid"] = child_pid
        if event_type is not None:
            self.extra_meta["child_event_type"] = event_type

    def mark_child_event(self, event_type, message, *, phase=None, child_pid=None) -> None:
        self.child_events.append((event_type, message, phase, child_pid))
        self.extra_meta["child_event_type"] = event_type
        self.extra_meta["child_event_message"] = message
        if phase is not None:
            self.extra_meta["child_event_phase"] = phase
        if child_pid is not None:
            self.extra_meta["child_pid"] = child_pid


def test_hybrid_reference_margin_updates_before_branch_generation() -> None:
    recorder = ConcolicTestRecorder(None, "case_0")
    recorder.input_shape = (1, 1, 3)
    recorder.extra_meta["hybrid_de_original_label"] = 0
    config = {
        "transform_mode": "affine-brightness-contrast",
        "effective_min": -0.1,
        "effective_max": 0.1,
        "bounds_mode": "clip",
        "coefficient_by_input": {
            "v_0_0_0": 0.0,
            "v_0_0_1": 0.0,
            "v_0_0_2": 0.0,
        },
    }
    recorder.global_real_config = config
    observed = []

    def predict_batch(images):
        brightness = float(images[0].mean()) - 0.5
        if brightness > 0.09:
            return np.asarray([[0.4, 0.6]])
        if brightness > 0.07:
            return np.asarray([[0.51, 0.49]])
        return np.asarray([[0.6, 0.4]])

    engine = type("Engine", (), {})()
    engine.global_real_config = config
    engine.reference_score_predictor = predict_batch
    engine.current_reference_margin = None
    engine._get_recorder = lambda: recorder
    engine._one_execution = lambda *_args: observed.append(engine.current_reference_margin)
    runner = CandidateExecutionRunner(engine)
    inputs = {
        "v_0_0_0": 0.5,
        "v_0_0_1": 0.5,
        "v_0_0_2": 0.5,
        "__pyct_brightness": 0.05,
        "__pyct_contrast": 0.0,
    }

    runner.run_initial_execution(inputs, {})
    assert observed == pytest.approx([0.2])
    assert recorder.original_label == 0
    assert recorder.extra_meta["hybrid_pyct_seed_margin"] == pytest.approx(0.2)

    assert runner.validate_sat_candidate({**inputs, "__pyct_brightness": 0.08}) is False
    assert engine.current_reference_margin == pytest.approx(0.02)
    assert runner.validate_sat_candidate({**inputs, "__pyct_brightness": 0.1}) is True
    assert recorder.attack_label == 1
    assert recorder.extra_meta["hybrid_pyct_best_margin"] == pytest.approx(-0.2)
    assert recorder.extra_meta["hybrid_pyct_candidate_count"] == 2


def test_hybrid_reference_rejects_invalid_score_matrix() -> None:
    recorder = ConcolicTestRecorder(None, "case_0")
    recorder.input_shape = (1, 1, 3)
    recorder.extra_meta["hybrid_de_original_label"] = 0
    engine = type("Engine", (), {})()
    engine.global_real_config = {
        "transform_mode": "affine-brightness-contrast",
        "effective_min": -0.1,
        "effective_max": 0.1,
        "bounds_mode": "clip",
        "coefficient_by_input": {
            "v_0_0_0": 0.0,
            "v_0_0_1": 0.0,
            "v_0_0_2": 0.0,
        },
    }
    recorder.global_real_config = engine.global_real_config
    engine.reference_score_predictor = lambda _images: np.asarray([[float("nan"), 0.0]])
    engine._get_recorder = lambda: recorder
    engine._one_execution = lambda *_args: None

    with pytest.raises(ValueError, match="invalid class scores"):
        CandidateExecutionRunner(engine).run_initial_execution(
            {
                "v_0_0_0": 0.5,
                "v_0_0_1": 0.5,
                "v_0_0_2": 0.5,
                "__pyct_brightness": 0.0,
                "__pyct_contrast": 0.0,
            },
            {},
        )
    assert recorder.extra_meta["error_type"] == "reference_prediction_failure"


def test_hybrid_reference_rejects_seed_label_mismatch() -> None:
    recorder = ConcolicTestRecorder(None, "case_0")
    recorder.input_shape = (1, 1, 3)
    recorder.extra_meta["hybrid_de_original_label"] = 0
    config = {
        "transform_mode": "affine-brightness-contrast",
        "effective_min": -0.1,
        "effective_max": 0.1,
        "bounds_mode": "clip",
        "coefficient_by_input": {f"v_0_0_{i}": 0.0 for i in range(3)},
    }
    recorder.global_real_config = config
    engine = type("Engine", (), {})()
    engine.global_real_config = config
    engine.reference_score_predictor = lambda _images: np.asarray([[0.2, 0.8]])
    engine._get_recorder = lambda: recorder
    engine._one_execution = lambda *_args: pytest.fail("invalid seed must not execute")

    with pytest.raises(ValueError, match="seed label differs"):
        CandidateExecutionRunner(engine).run_initial_execution(
            {
                **{f"v_0_0_{i}": 0.5 for i in range(3)},
                "__pyct_brightness": 0.0,
                "__pyct_contrast": 0.0,
            },
            {},
        )
    assert recorder.extra_meta["error_type"] == "hybrid_seed_prediction_mismatch"


@pytest.mark.parametrize("kind", ["aces-brightness", "aces-contrast"])
@pytest.mark.parametrize("seed_label", [0, 1])
def test_aces_hybrid_initial_reference_keeps_source_label(kind, seed_label) -> None:
    seed = np.asarray([[[0.2, 0.4, 0.7]]], dtype=np.float64)
    config = build_aces_like_global_real_config(seed, {
        "global_shift_kind": kind, "effective_min": -0.1, "effective_max": 0.1,
        "bounds_mode": "clip", "pwl_max_segments": 8,
        "pwl_error_tolerance": 1.0 / 255.0, "hybrid_de_enabled": True,
    })
    recorder = ConcolicTestRecorder(None, "case_0")
    recorder.input_shape = seed.shape
    recorder.global_real_config = config
    recorder.extra_meta["hybrid_de_original_label"] = 0
    engine = type("Engine", (), {})()
    engine.global_real_config = config
    engine.reference_score_predictor = lambda _images: np.asarray(
        [[0.8, 0.2] if seed_label == 0 else [0.2, 0.8]]
    )
    engine._get_recorder = lambda: recorder
    observed = []
    engine._one_execution = lambda *_args: observed.append(engine.current_reference_margin)
    inputs = {f"v_0_0_{i}": float(seed[0, 0, i]) for i in range(3)}
    inputs[GLOBAL_X_INPUT_NAME] = 0.0
    runner = CandidateExecutionRunner(engine)

    if seed_label == 1:
        with pytest.raises(ValueError, match="seed label differs"):
            runner.run_initial_execution(inputs, {})
        assert observed == []
        assert recorder.extra_meta["error_type"] == "hybrid_seed_prediction_mismatch"
    else:
        runner.run_initial_execution(inputs, {})
        assert observed == pytest.approx([0.6])
        assert recorder.original_label == 0
        assert recorder.extra_meta["hybrid_pyct_seed_margin"] == pytest.approx(0.6)


def _make_engine(reference_execute):
    engine = explore.ExplorationEngine.__new__(explore.ExplorationEngine)
    engine.reference_execute = reference_execute
    engine.normalize = None
    engine.limit_change_range = None
    engine.constraints_collection_type = "queue"
    engine.constraints_to_solve = deque([object()])
    engine.idx = 0
    engine.only_first_forward = False
    engine.symbolic_path_threshold = None
    engine.symbolic_enabled = True
    engine.symbolic_disabled_at_path_len = None
    engine.original_args = {}
    engine.var_to_types = {}
    engine.concolic_name_list = []
    engine.concolic_flag_dict = {}
    engine.input_name = "case_0"
    engine.save_dir = None
    return engine


def test_statsdir_keeps_solver_stats_without_legacy_inputs_pickle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    statsdir = tmp_path / "stats"
    statsdir.mkdir()
    engine = explore.ExplorationEngine.__new__(explore.ExplorationEngine)
    engine.save_dir = None
    engine.input_name = "case_0"
    engine.only_first_forward = False
    engine.shap_score_alpha = None
    engine.symbolic_path_threshold = None
    engine.reference_execute = lambda **_data: 0
    engine.statsdir = str(statsdir)

    def fake_execution_loop(*_args, **_kwargs):
        explore.recorder.start()
        return False

    monkeypatch.setattr(engine, "_execution_loop", fake_execution_loop)
    monkeypatch.setattr(engine, "_can_use_concolic_wrapper", lambda *_args: False)
    monkeypatch.setattr(explore.Solver, "ctr_size", {}, raising=False)
    explore.Solver.stats = {
        "sat_number": 0,
        "sat_time": 0,
        "unsat_number": 0,
        "unsat_time": 0,
        "otherwise_number": 0,
        "otherwise_time": 0,
    }

    engine.explore(
        "test/test_candidate_execution_runner",
        {},
        root=str(ROOT),
        funcname="unused",
        collect_constraints_with="queue",
        idx=0,
    )

    assert (statsdir / "smt.csv").is_file()
    assert not (statsdir / "inputs.pkl").exists()


class _FakeConn:
    def __init__(self, *, poll_values=None, recv_value=None, recv_exc=None) -> None:
        self._poll_values = list(poll_values or [True])
        self._recv_value = recv_value
        self._recv_exc = recv_exc

    def poll(self, _timeout=None):
        if self._poll_values:
            return self._poll_values.pop(0)
        return False

    def recv(self):
        if self._recv_exc is not None:
            raise self._recv_exc
        return self._recv_value


class _FakeProcess:
    def __init__(self, pid=1234, *, alive=False, exitcode=1) -> None:
        self.pid = pid
        self._alive = alive
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return self._alive


def test_execution_loop_uses_keras_reference_for_labels(monkeypatch) -> None:
    recorder = _RecorderStub()
    explore.recorder = recorder
    explore.Solver.stats = {
        "sat_number": 0,
        "sat_time": 0,
        "unsat_number": 0,
        "unsat_time": 0,
        "otherwise_number": 0,
        "otherwise_time": 0,
    }

    reference_calls = []
    search_calls = []

    def reference_execute(**data):
        reference_calls.append(dict(data))
        return 0 if data["v_0_0"] == 0.0 else 1

    engine = _make_engine(reference_execute)

    def fake_one_execution(all_args, concolic_dict):
        search_calls.append(dict(all_args))
        return True

    monkeypatch.setattr(engine, "_one_execution", fake_one_execution)
    monkeypatch.setattr(
        explore.Solver,
        "find_model_from_constraint",
        lambda *_args, **_kwargs: {"v_0_0": 1.0},
    )

    timed_out = engine._execution_loop(0, {"v_0_0": 0.0}, {})

    assert timed_out is False
    assert recorder.original_label == 0
    assert recorder.attack_label == 1
    assert reference_calls == [{"v_0_0": 0.0}, {"v_0_0": 1.0}]
    assert search_calls == [{"v_0_0": 0.0}]
    assert recorder.extra_meta["reference_prediction_phase_counts"] == {
        "original_reference": 1,
        "candidate_reference": 1,
    }


def test_execution_loop_runs_search_only_after_reference_rejects_candidate(monkeypatch) -> None:
    recorder = _RecorderStub()
    explore.recorder = recorder
    explore.Solver.stats = {
        "sat_number": 0,
        "sat_time": 0,
        "unsat_number": 0,
        "unsat_time": 0,
        "otherwise_number": 0,
        "otherwise_time": 0,
    }

    reference_calls = []

    def reference_execute(**data):
        reference_calls.append(dict(data))
        return 0

    engine = _make_engine(reference_execute)
    search_calls = []
    monkeypatch.setattr(
        engine,
        "_one_execution",
        lambda all_args, _concolic_dict: search_calls.append(dict(all_args)) or True,
    )
    monkeypatch.setattr(
        explore.Solver,
        "find_model_from_constraint",
        lambda *_args, **_kwargs: {"v_0_0": 1.0},
    )

    timed_out = engine._execution_loop(1, {"v_0_0": 0.0}, {})

    assert timed_out is False
    assert recorder.original_label == 0
    assert recorder.attack_label is None
    assert reference_calls == [{"v_0_0": 0.0}, {"v_0_0": 1.0}]
    assert search_calls == [{"v_0_0": 0.0}, {"v_0_0": 1.0}]


@pytest.mark.parametrize("mode", ["full", "last"])
def test_execution_loop_counts_duplicate_candidates_and_reference_success(monkeypatch, mode):
    monkeypatch.setenv("PYCT_SMT_PATH_MODE", mode)
    recorder = _RecorderStub()
    monkeypatch.setattr(explore, "recorder", recorder)
    monkeypatch.setattr(explore.Solver, "stats", {
        "sat_number": 0, "sat_time": 0, "unsat_number": 0, "unsat_time": 0,
        "otherwise_number": 0, "otherwise_time": 0,
    }, raising=False)
    reference_calls = []

    def reference_execute(**inputs):
        reference_calls.append(dict(inputs))
        return int(inputs["v_0_0"] == 2.0)

    engine = _make_engine(reference_execute)
    engine.constraints_to_solve = deque([object(), object(), object()])
    search_calls = []
    monkeypatch.setattr(engine, "_one_execution", lambda inputs, _flags:
                        search_calls.append(dict(inputs)) or True)
    candidates = iter([{"v_0_0": 1.0}, {"v_0_0": 1.0}, {"v_0_0": 2.0}])
    monkeypatch.setattr(explore.Solver, "find_model_from_constraint",
                        lambda *_args, **_kwargs: next(candidates))

    assert engine._execution_loop(0, {"v_0_0": 0.0}, {}) is False

    assert recorder.extra_meta["smt_path_mode"] == mode
    assert recorder.extra_meta["smt_sat_candidate_count"] == 3
    assert recorder.extra_meta["smt_duplicate_candidate_count"] == 1
    assert recorder.extra_meta["smt_validated_candidate_count"] == 2
    assert recorder.extra_meta["smt_successful_candidate_count"] == 1
    assert recorder.extra_meta["smt_candidate_validation_wall_time_seconds"] >= 0.0
    assert recorder.attack_label == 1
    assert reference_calls == [{"v_0_0": 0.0}, {"v_0_0": 1.0}, {"v_0_0": 2.0}]
    assert search_calls == [{"v_0_0": 0.0}, {"v_0_0": 1.0}]


def test_execution_loop_ignores_search_label_disagreement(monkeypatch) -> None:
    recorder = _RecorderStub()
    explore.recorder = recorder
    explore.Solver.stats = {
        "sat_number": 0,
        "sat_time": 0,
        "unsat_number": 0,
        "unsat_time": 0,
        "otherwise_number": 0,
        "otherwise_time": 0,
    }

    def reference_execute(**_data):
        return 0

    engine = _make_engine(reference_execute)
    search_results = iter([0, 1])
    search_calls = []
    monkeypatch.setattr(
        engine,
        "_one_execution",
        lambda all_args, _concolic_dict: search_calls.append(dict(all_args))
        or next(search_results),
    )
    monkeypatch.setattr(
        explore.Solver,
        "find_model_from_constraint",
        lambda *_args, **_kwargs: {"v_0_0": 1.0},
    )

    timed_out = engine._execution_loop(1, {"v_0_0": 0.0}, {})

    assert timed_out is False
    assert recorder.attack_label is None
    assert recorder.original_label == 0
    assert search_calls == [{"v_0_0": 0.0}, {"v_0_0": 1.0}]


def test_execution_loop_only_first_forward_has_only_original_reference(monkeypatch) -> None:
    recorder = _RecorderStub()
    explore.recorder = recorder
    explore.Solver.stats = {
        "sat_number": 0,
        "sat_time": 0,
        "unsat_number": 0,
        "unsat_time": 0,
        "otherwise_number": 0,
        "otherwise_time": 0,
    }

    reference_calls = []

    def reference_execute(**data):
        reference_calls.append(dict(data))
        return 0

    engine = _make_engine(reference_execute)
    engine.only_first_forward = True
    search_calls = []
    monkeypatch.setattr(
        engine,
        "_one_execution",
        lambda all_args, _concolic_dict: search_calls.append(dict(all_args)) or True,
    )
    monkeypatch.setattr(
        explore.Solver,
        "find_model_from_constraint",
        lambda *_args, **_kwargs: {"v_0_0": 1.0},
    )

    timed_out = engine._execution_loop(1, {"v_0_0": 0.0}, {})

    assert timed_out is False
    assert recorder.attack_label is None
    assert reference_calls == [{"v_0_0": 0.0}]
    assert search_calls == [{"v_0_0": 0.0}]


def test_unpicklable_constraint_transfer_marks_error_and_preserves_queue() -> None:
    recorder = _RecorderStub()
    explore.recorder = recorder
    engine = _make_engine(lambda **_data: 0)
    original_queue = engine.constraints_to_solve

    with pytest.raises(explore.ConstraintTransferError):
        engine._apply_constraint_transfer_payload(engine.Unpicklable)

    assert engine.constraints_to_solve is original_queue
    assert list(engine.constraints_to_solve)
    assert recorder.extra_meta["status"] == "error"
    assert recorder.extra_meta["error_type"] == "constraint_transfer_failure"
    assert "unpicklable constraint/path payload" in recorder.extra_meta["error_reason"]
    assert recorder.solve_all_ctr is False


def test_handle_child_event_records_traceable_metadata(caplog) -> None:
    recorder = _RecorderStub()
    explore.recorder = recorder
    engine = _make_engine(lambda **_data: 0)
    envelope = {
        "kind": "child_event",
        "pid": 4321,
        "phase": "execute",
        "updated_args": {"v_0_0": 2.0},
        "result": engine.Timeout,
        "event_type": "soft_timeout",
        "message": "child soft timeout",
        "var_to_types": {"v_0_0": float},
        "concolic_name_list": ["v_0_0"],
        "concolic_flag_dict": {"v_0_0": 1},
    }
    all_args = {"v_0_0": 0.0}

    with caplog.at_level("WARNING", logger="ct.explore"):
        result = engine._handle_child_envelope(all_args, envelope)

    assert result is engine.Timeout
    assert all_args == {"v_0_0": 2.0}
    assert recorder.extra_meta["child_event_type"] == "soft_timeout"
    assert recorder.extra_meta["child_event_phase"] == "execute"
    assert recorder.extra_meta["child_pid"] == 4321
    assert "[CHILD-EVENT]" in caplog.text
    assert "input_name=case_0" in caplog.text


def test_handle_child_error_writes_traceback_and_marks_terminal_error(tmp_path: Path, caplog) -> None:
    recorder = _RecorderStub()
    explore.recorder = recorder
    engine = _make_engine(lambda **_data: 0)
    engine.save_dir = str(tmp_path / "case_error")
    envelope = {
        "kind": "child_error",
        "pid": 999,
        "phase": "execute",
        "updated_args": {"v_0_0": 3.0},
        "result": engine.Exception,
        "error_type": "child_unexpected_error",
        "message": "boom",
        "traceback": "traceback text",
    }

    with caplog.at_level("ERROR", logger="ct.explore"):
        with pytest.raises(RuntimeError, match="boom"):
            engine._handle_child_envelope({"v_0_0": 0.0}, envelope)

    assert recorder.extra_meta["status"] == "error"
    assert recorder.extra_meta["error_type"] == "child_unexpected_error"
    assert recorder.extra_meta["error_phase"] == "execute"
    assert recorder.extra_meta["child_pid"] == 999
    assert (Path(engine.save_dir) / "child_error_traceback.txt").read_text(encoding="utf-8") == "traceback text"
    assert "[CHILD-ERROR]" in caplog.text
    assert "save_dir=" in caplog.text


def test_receive_child_envelope_maps_eof_to_transfer_failure(tmp_path: Path, caplog) -> None:
    recorder = _RecorderStub()
    explore.recorder = recorder
    engine = _make_engine(lambda **_data: 0)
    engine.save_dir = str(tmp_path / "case_transport")

    with caplog.at_level("ERROR", logger="ct.explore"):
        with pytest.raises(explore.ConstraintTransferError):
            engine._receive_child_envelope(
                _FakeConn(recv_exc=EOFError("closed")),
                _FakeProcess(pid=321, alive=True, exitcode=None),
                1,
            )

    assert recorder.extra_meta["status"] == "error"
    assert recorder.extra_meta["error_type"] == "constraint_transfer_failure"
    assert recorder.extra_meta["error_phase"] == "transport"
    assert recorder.extra_meta["child_pid"] == 321
    assert "[PARENT-RECV-ERROR]" in caplog.text
    assert (Path(engine.save_dir) / "transfer_error_traceback.txt").is_file()


def test_receive_child_envelope_rejects_unknown_kind_as_protocol_failure(tmp_path: Path) -> None:
    recorder = _RecorderStub()
    explore.recorder = recorder
    engine = _make_engine(lambda **_data: 0)
    engine.save_dir = str(tmp_path / "case_protocol")

    with pytest.raises(explore.ConstraintTransferError):
        engine._receive_child_envelope(
            _FakeConn(recv_value={"kind": "weird", "pid": 22, "phase": "protocol", "result": None}),
            _FakeProcess(pid=22, alive=True, exitcode=None),
            1,
        )

    assert recorder.extra_meta["error_type"] == "constraint_transfer_failure"
    assert recorder.extra_meta["error_phase"] == "protocol"
    assert (Path(engine.save_dir) / "transfer_error_traceback.txt").read_text(encoding="utf-8").startswith("{")


def test_receive_child_envelope_maps_early_child_exit_to_transfer_failure(tmp_path: Path) -> None:
    recorder = _RecorderStub()
    explore.recorder = recorder
    engine = _make_engine(lambda **_data: 0)
    engine.save_dir = str(tmp_path / "case_exit")

    with pytest.raises(explore.ConstraintTransferError):
        engine._receive_child_envelope(
            _FakeConn(poll_values=[False, False]),
            _FakeProcess(pid=77, alive=False, exitcode=9),
            1,
        )

    assert recorder.extra_meta["error_type"] == "constraint_transfer_failure"
    assert recorder.extra_meta["error_phase"] == "transport"
    assert recorder.extra_meta["child_pid"] == 77


def test_execution_loop_fails_closed_on_first_transfer_failure(monkeypatch) -> None:
    recorder = _RecorderStub()
    explore.recorder = recorder
    explore.Solver.stats = {
        "sat_number": 0,
        "sat_time": 0,
        "unsat_number": 0,
        "unsat_time": 0,
        "otherwise_number": 0,
        "otherwise_time": 0,
    }
    engine = _make_engine(lambda **_data: 0)
    engine.constraints_to_solve = deque()

    def fail_transfer(_all_args, _concolic_dict):
        engine._apply_constraint_transfer_payload(engine.Unpicklable)

    monkeypatch.setattr(engine, "_one_execution", fail_transfer)

    with pytest.raises(explore.ConstraintTransferError):
        engine._execution_loop(0, {"v_0_0": 0.0}, {})

    assert recorder.extra_meta["status"] == "error"
    assert recorder.extra_meta["error_type"] == "constraint_transfer_failure"
    assert recorder.total_iter == -1
    assert recorder.no_ctr_calls == 0
    assert recorder.solve_all_ctr is False


def test_execution_loop_fails_closed_on_mid_iteration_transfer_failure(monkeypatch) -> None:
    recorder = _RecorderStub()
    explore.recorder = recorder
    explore.Solver.stats = {
        "sat_number": 0,
        "sat_time": 0,
        "unsat_number": 0,
        "unsat_time": 0,
        "otherwise_number": 0,
        "otherwise_time": 0,
    }
    engine = _make_engine(lambda **_data: 0)
    calls = 0

    def fake_one_execution(_all_args, _concolic_dict):
        nonlocal calls
        calls += 1
        if calls == 2:
            engine._apply_constraint_transfer_payload(engine.Unpicklable)
        return True

    monkeypatch.setattr(engine, "_one_execution", fake_one_execution)
    monkeypatch.setattr(
        explore.Solver,
        "find_model_from_constraint",
        lambda *_args, **_kwargs: {"v_0_0": 1.0},
    )

    with pytest.raises(explore.ConstraintTransferError):
        engine._execution_loop(0, {"v_0_0": 0.0}, {})

    assert calls == 2
    assert recorder.extra_meta["status"] == "error"
    assert recorder.extra_meta["error_type"] == "constraint_transfer_failure"
    assert recorder.no_ctr_calls == 0
    assert recorder.solve_all_ctr is False
