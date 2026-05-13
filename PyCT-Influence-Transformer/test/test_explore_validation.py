from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import libct.explore as explore


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


def _make_engine(validation_execute):
    engine = explore.ExplorationEngine.__new__(explore.ExplorationEngine)
    engine.validation_execute = validation_execute
    engine.normalize = None
    engine.limit_change_range = None
    engine.constraints_collection_type = "queue"
    engine.constraints_to_solve = deque([object()])
    engine.idx = 0
    engine.only_first_forward = False
    engine.symbolic_path_threshold = None
    engine.symbolic_enabled = True
    engine.symbolic_disabled_at_path_len = None
    engine.previous_result = None
    engine.original_args = {}
    engine.var_to_types = {}
    engine.concolic_name_list = []
    engine.concolic_flag_dict = {}
    engine.input_name = "case_0"
    engine.save_dir = None
    engine.in_out = []
    return engine


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


def test_execution_loop_uses_validation_predictor_for_labels(monkeypatch) -> None:
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

    validation_calls = []
    search_calls = []

    def validation_execute(**data):
        validation_calls.append(dict(data))
        return 0 if data["v_0_0"] == 0.0 else 1

    engine = _make_engine(validation_execute)

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
    assert validation_calls == [{"v_0_0": 0.0}, {"v_0_0": 1.0}]
    assert search_calls == [{"v_0_0": 0.0}]


def test_execution_loop_reuses_search_result_for_non_ternary_candidate(monkeypatch) -> None:
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

    validation_calls = []

    def validation_execute(**data):
        validation_calls.append(dict(data))
        return 0

    engine = _make_engine(validation_execute)
    engine.reuse_search_result_for_validation = True
    engine.single_coverage = False
    initial_search_calls = []
    candidate_search_calls = []
    applied_payloads = []

    monkeypatch.setattr(engine, "_one_execution", lambda all_args, _concolic_dict: initial_search_calls.append(dict(all_args)) or True)
    monkeypatch.setattr(
        engine,
        "_one_execution_deferred_constraints",
        lambda all_args, _concolic_dict: candidate_search_calls.append(dict(all_args)) or (0, "payload"),
    )

    def apply_payload(payload):
        applied_payloads.append(payload)
        engine.constraints_to_solve.append("generated")

    monkeypatch.setattr(engine, "_apply_constraint_transfer_payload", apply_payload)
    monkeypatch.setattr(
        explore.Solver,
        "find_model_from_constraint",
        lambda *_args, **_kwargs: {"v_0_0": 1.0},
    )

    timed_out = engine._execution_loop(1, {"v_0_0": 0.0}, {})

    assert timed_out is False
    assert recorder.original_label == 0
    assert recorder.attack_label is None
    assert validation_calls == [{"v_0_0": 0.0}]
    assert initial_search_calls == [{"v_0_0": 0.0}]
    assert candidate_search_calls == [{"v_0_0": 1.0}]
    assert applied_payloads == ["payload"]
    assert engine.in_out == [({"v_0_0": 1.0}, 0)]


def test_execution_loop_discards_deferred_constraints_for_reused_adversarial_candidate(monkeypatch) -> None:
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

    validation_calls = []

    def validation_execute(**data):
        validation_calls.append(dict(data))
        return 0

    engine = _make_engine(validation_execute)
    engine.reuse_search_result_for_validation = True
    engine.single_coverage = False
    initial_search_calls = []
    candidate_search_calls = []
    applied_payloads = []

    monkeypatch.setattr(engine, "_one_execution", lambda all_args, _concolic_dict: initial_search_calls.append(dict(all_args)) or True)
    monkeypatch.setattr(
        engine,
        "_one_execution_deferred_constraints",
        lambda all_args, _concolic_dict: candidate_search_calls.append(dict(all_args)) or (1, "payload"),
    )
    monkeypatch.setattr(engine, "_apply_constraint_transfer_payload", lambda payload: applied_payloads.append(payload))
    monkeypatch.setattr(
        explore.Solver,
        "find_model_from_constraint",
        lambda *_args, **_kwargs: {"v_0_0": 1.0},
    )

    timed_out = engine._execution_loop(0, {"v_0_0": 0.0}, {})

    assert timed_out is False
    assert recorder.attack_label == 1
    assert recorder.adversarial_input == {"v_0_0": 1.0}
    assert validation_calls == [{"v_0_0": 0.0}]
    assert initial_search_calls == [{"v_0_0": 0.0}]
    assert candidate_search_calls == [{"v_0_0": 1.0}]
    assert applied_payloads == []
    assert engine.in_out == []


def test_execution_loop_does_not_treat_reused_sentinel_as_label_change(monkeypatch) -> None:
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
    engine.reuse_search_result_for_validation = True
    engine.single_coverage = False
    candidate_search_calls = []

    monkeypatch.setattr(engine, "_one_execution", lambda _all_args, _concolic_dict: True)

    def deferred_timeout(all_args, _concolic_dict):
        candidate_search_calls.append(dict(all_args))
        engine.previous_result = engine.Timeout
        return engine.Timeout, None

    monkeypatch.setattr(
        engine,
        "_one_execution_deferred_constraints",
        deferred_timeout,
    )
    monkeypatch.setattr(
        explore.Solver,
        "find_model_from_constraint",
        lambda *_args, **_kwargs: {"v_0_0": 1.0},
    )

    timed_out = engine._execution_loop(1, {"v_0_0": 0.0}, {})

    assert timed_out is False
    assert recorder.attack_label is None
    assert engine.previous_result is engine.Timeout
    assert candidate_search_calls == [{"v_0_0": 1.0}]
    assert engine.in_out == [({"v_0_0": 1.0}, engine.Timeout)]


def test_execution_loop_without_reuse_keeps_validation_then_search(monkeypatch) -> None:
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

    validation_calls = []

    def validation_execute(**data):
        validation_calls.append(dict(data))
        return 0

    engine = _make_engine(validation_execute)
    engine.reuse_search_result_for_validation = False
    search_calls = []

    monkeypatch.setattr(engine, "_one_execution", lambda all_args, _concolic_dict: search_calls.append(dict(all_args)) or True)
    monkeypatch.setattr(
        explore.Solver,
        "find_model_from_constraint",
        lambda *_args, **_kwargs: {"v_0_0": 1.0},
    )

    timed_out = engine._execution_loop(1, {"v_0_0": 0.0}, {})

    assert timed_out is False
    assert recorder.attack_label is None
    assert validation_calls == [{"v_0_0": 0.0}, {"v_0_0": 1.0}]
    assert search_calls == [{"v_0_0": 0.0}, {"v_0_0": 1.0}]


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
