from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest

from libct.constraint import Constraint
from libct.executor.child_protocol import ChildProtocol, ConstraintTransferError


class _RecorderStub:
    def __init__(self) -> None:
        self.extra_meta = {}
        self.child_events = []

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


class _EngineStub:
    Exception = object()
    Timeout = object()
    Unpicklable = object()

    def __init__(self, *, save_dir=None, recorder=None) -> None:
        self.idx = 7
        self.input_name = "case_7"
        self.save_dir = save_dir
        self.var_to_types = {"old": "Int"}
        self.concolic_name_list = ["old_VAR"]
        self.concolic_flag_dict = {"old_VAR": 1}
        self.constraints_to_solve = deque(["original"])
        self.path = "old_path"
        self.symbolic_disabled_at_path_len = None
        self._recorder = recorder

    def _get_recorder(self):
        return self._recorder


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


def setup_function() -> None:
    Constraint.global_constraints.clear()


def test_validate_child_envelope_preserves_existing_contract() -> None:
    protocol = ChildProtocol(_EngineStub())
    ok_envelope = {
        "kind": "ok",
        "pid": 1,
        "phase": "execute",
        "result": "result",
        "constraint_payload": ([], [], "path"),
    }

    assert protocol.validate_child_envelope(ok_envelope) is ok_envelope

    with pytest.raises(ValueError, match="unknown envelope kind"):
        protocol.validate_child_envelope({"kind": "bad", "pid": 1, "phase": "execute", "result": None})
    with pytest.raises(ValueError, match="ok envelope missing constraint_payload"):
        protocol.validate_child_envelope({"kind": "ok", "pid": 1, "phase": "execute", "result": None})


def test_child_envelope_builders_include_shared_state() -> None:
    engine = _EngineStub()
    protocol = ChildProtocol(engine)

    envelope = protocol.build_child_error_envelope(
        pid=22,
        updated_args={"x": 1},
        error_type="child_unexpected_error",
        phase="execute",
        message="boom",
        error_class="RuntimeError",
        traceback_text="trace",
    )

    assert envelope["kind"] == "child_error"
    assert envelope["pid"] == 22
    assert envelope["updated_args"] == {"x": 1}
    assert envelope["var_to_types"] == engine.var_to_types
    assert envelope["concolic_name_list"] == engine.concolic_name_list
    assert envelope["concolic_flag_dict"] == engine.concolic_flag_dict
    assert envelope["traceback"] == "trace"


def test_handle_ok_envelope_applies_shared_state_and_constraint_payload() -> None:
    engine = _EngineStub()
    protocol = ChildProtocol(engine)
    new_constraints = [object()]
    new_queue = deque(["new"])
    new_path = object()
    all_args = {"x": 0}
    envelope = {
        "kind": "ok",
        "pid": 5,
        "phase": "execute",
        "updated_args": {"x": 2},
        "result": "done",
        "constraint_payload": (new_constraints, new_queue, new_path, 4),
        "var_to_types": {"x_VAR": "Real"},
        "concolic_name_list": ["x_VAR"],
        "concolic_flag_dict": {"x_VAR": 1},
    }

    result = protocol.handle_child_envelope(all_args, envelope)

    assert result == "done"
    assert all_args == {"x": 2}
    assert Constraint.global_constraints is new_constraints
    assert engine.constraints_to_solve is new_queue
    assert engine.path is new_path
    assert engine.symbolic_disabled_at_path_len == 4
    assert engine.var_to_types == {"x_VAR": "Real"}


def test_child_event_records_metadata_and_returns_result(caplog) -> None:
    recorder = _RecorderStub()
    engine = _EngineStub(recorder=recorder)
    protocol = ChildProtocol(engine)

    with caplog.at_level("WARNING", logger="ct.explore"):
        result = protocol.handle_child_envelope(
            {},
            {
                "kind": "child_event",
                "pid": 77,
                "phase": "execute",
                "updated_args": None,
                "result": engine.Timeout,
                "event_type": "soft_timeout",
                "message": "timeout",
            },
        )

    assert result is engine.Timeout
    assert recorder.extra_meta["child_event_type"] == "soft_timeout"
    assert recorder.extra_meta["child_pid"] == 77
    assert "[CHILD-EVENT]" in caplog.text
    assert "input_name=case_7" in caplog.text


def test_child_error_writes_diagnostic_and_raises(tmp_path: Path) -> None:
    recorder = _RecorderStub()
    engine = _EngineStub(save_dir=str(tmp_path / "case_error"), recorder=recorder)
    protocol = ChildProtocol(engine)

    with pytest.raises(RuntimeError, match="boom"):
        protocol.handle_child_envelope(
            {},
            {
                "kind": "child_error",
                "pid": 88,
                "phase": "execute",
                "updated_args": None,
                "result": engine.Exception,
                "error_type": "child_unexpected_error",
                "message": "boom",
                "traceback": "traceback text",
            },
        )

    assert recorder.extra_meta["status"] == "error"
    assert recorder.extra_meta["error_type"] == "child_unexpected_error"
    assert recorder.extra_meta["child_pid"] == 88
    assert (Path(engine.save_dir) / "child_error_traceback.txt").read_text(encoding="utf-8") == "traceback text"


def test_receive_child_envelope_maps_transport_and_protocol_failures(tmp_path: Path) -> None:
    recorder = _RecorderStub()
    engine = _EngineStub(save_dir=str(tmp_path / "case_transport"), recorder=recorder)
    protocol = ChildProtocol(engine)

    with pytest.raises(ConstraintTransferError):
        protocol.receive_child_envelope(
            _FakeConn(recv_exc=EOFError("closed")),
            _FakeProcess(pid=321, alive=True, exitcode=None),
            1,
        )

    assert recorder.extra_meta["error_type"] == "constraint_transfer_failure"
    assert recorder.extra_meta["error_phase"] == "transport"
    assert (Path(engine.save_dir) / "transfer_error_traceback.txt").is_file()

    with pytest.raises(ConstraintTransferError):
        protocol.receive_child_envelope(
            _FakeConn(recv_value={"kind": "weird", "pid": 22, "phase": "protocol", "result": None}),
            _FakeProcess(pid=22, alive=True, exitcode=None),
            1,
        )

    assert recorder.extra_meta["error_phase"] == "protocol"


def test_unpicklable_payload_fails_closed_and_preserves_queue() -> None:
    recorder = _RecorderStub()
    engine = _EngineStub(recorder=recorder)
    protocol = ChildProtocol(engine)
    original_queue = engine.constraints_to_solve

    with pytest.raises(ConstraintTransferError):
        protocol.apply_constraint_transfer_payload(engine.Unpicklable)

    assert engine.constraints_to_solve is original_queue
    assert list(engine.constraints_to_solve) == ["original"]
    assert recorder.extra_meta["status"] == "error"
    assert recorder.extra_meta["error_type"] == "constraint_transfer_failure"
    assert "unpicklable constraint/path payload" in recorder.extra_meta["error_reason"]
