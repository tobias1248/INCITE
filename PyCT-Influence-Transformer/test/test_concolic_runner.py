from __future__ import annotations

from typing import Any, Dict, Optional

from libct.executor import concolic
from libct.executor.concolic import ConcolicExecutionRunner
from libct import explore


class _Conn:
    def __init__(self) -> None:
        self.closed = False
        self.sent = []

    def close(self) -> None:
        self.closed = True

    def send(self, value: Any) -> None:
        self.sent.append(value)


class _Process:
    def __init__(self, *, target: Any, args: tuple[Any, ...], alive: bool = True) -> None:
        self.target = target
        self.args = args
        self.started = False
        self.killed = False
        self.join_timeouts = []
        self._alive = alive
        self.pid = 1234
        self.exitcode = None

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self._alive

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def join(self, timeout: Optional[float] = None) -> None:
        self.join_timeouts.append(timeout)


class _ParentEngine:
    def __init__(self) -> None:
        self.single_timeout = 7
        self.received = []
        self.handled = []

    def _receive_child_envelope(self, conn: _Conn, process: _Process, timeout_seconds: int) -> Dict[str, Any]:
        self.received.append((conn, process, timeout_seconds))
        return {"kind": "ok", "pid": process.pid, "phase": "execute", "result": "raw"}

    def _handle_child_envelope(self, all_args: Dict[str, Any], envelope: Dict[str, Any]) -> str:
        self.handled.append((all_args, envelope))
        all_args["handled"] = True
        return "handled-result"


def test_concolic_runner_parent_receives_handles_and_cleans_up(monkeypatch) -> None:
    read_conn = _Conn()
    send_conn = _Conn()
    processes = []

    monkeypatch.setattr(concolic.multiprocessing, "Pipe", lambda: (read_conn, send_conn))

    def fake_process(*, target: Any, args: tuple[Any, ...]) -> _Process:
        process = _Process(target=target, args=args, alive=True)
        processes.append(process)
        return process

    monkeypatch.setattr(concolic.multiprocessing, "Process", fake_process)

    engine = _ParentEngine()
    all_args = {"x": 1}
    result = ConcolicExecutionRunner(engine).run(all_args, {"x": 1})

    assert result == "handled-result"
    assert all_args == {"x": 1, "handled": True}
    assert engine.received == [(read_conn, processes[0], 12)]
    assert engine.handled == [
        (all_args, {"kind": "ok", "pid": 1234, "phase": "execute", "result": "raw"})
    ]
    assert processes[0].started is True
    assert processes[0].killed is True
    assert processes[0].join_timeouts == [0.1]
    assert read_conn.closed is True
    assert send_conn.closed is True


def test_exploration_engine_concolic_wrapper_lazily_delegates() -> None:
    class _Runner:
        def __init__(self) -> None:
            self.calls = []

        def run(self, all_args, concolic_dict):
            self.calls.append((all_args, concolic_dict))
            return "runner-result"

    engine = explore.ExplorationEngine.__new__(explore.ExplorationEngine)
    runner = _Runner()
    engine._concolic_runner = runner
    all_args = {"x": 1}
    concolic_dict = {"x": 1}

    result = engine._one_execution_concolic(all_args, concolic_dict)

    assert result == "runner-result"
    assert runner.calls == [(all_args, concolic_dict)]


class _Path:
    def __init__(self) -> None:
        self.reset_count = getattr(self, "reset_count", -1) + 1


class _ChildEngine:
    class Exception:
        pass

    class Timeout:
        pass

    def __init__(self) -> None:
        self.path = _Path()
        self.can_use_concolic_wrapper = False
        self.single_timeout = 5
        self.root = "/repo"
        self.modpath = "target.py"
        self.funcname = "target"
        self.lib = None
        self.statsdir = None
        self.constraints_to_solve = ["constraint"]
        self.symbolic_disabled_at_path_len = 3
        self.reset_calls = 0
        self.argument_calls = []

    def _reset_symbolic_guard(self) -> None:
        self.reset_calls += 1

    def _get_execute(self):
        return lambda value: value + 1

    def _get_concolic_arguments(self, execute, all_args, concolic_dict):
        self.argument_calls.append((execute, dict(all_args), dict(concolic_dict)))
        return [all_args["value"]], {}

    def _build_child_ok_envelope(self, **kwargs):
        return {"kind": "ok", **kwargs}

    def _build_child_event_envelope(self, **kwargs):
        return {"kind": "child_event", **kwargs}

    def _build_child_error_envelope(self, **kwargs):
        return {"kind": "child_error", **kwargs}


def test_concolic_runner_child_success_builds_ok_envelope(monkeypatch) -> None:
    monkeypatch.setattr(concolic, "prepare_child_environment", lambda: None)
    monkeypatch.setattr(concolic.os, "getpid", lambda: 999)
    monkeypatch.setattr(
        concolic.func_timeout,
        "func_timeout",
        lambda _timeout, func, args, kwargs: func(*args, **kwargs),
    )

    engine = _ChildEngine()
    send_conn = _Conn()
    ConcolicExecutionRunner(engine)._child_process(send_conn, {"value": 4}, {"value": 1})

    assert engine.reset_calls == 1
    assert engine.path.reset_count == 1
    assert len(engine.argument_calls) == 1
    assert send_conn.sent == [
        {
            "kind": "ok",
            "pid": 999,
            "updated_args": {"value": 4},
            "result": 5,
            "constraint_payload": (concolic.Constraint.global_constraints, ["constraint"], engine.path, 3),
        }
    ]
