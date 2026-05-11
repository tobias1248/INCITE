from __future__ import annotations

from typing import Any

from libct import explore
from libct.executor import primitive
from libct.executor.primitive import PrimitiveExecutionRunner


class _Conn:
    def __init__(self, recv_values=None, poll_values=None) -> None:
        self._recv_values = list(recv_values or [])
        self._poll_values = list(poll_values or [])
        self.closed = False
        self.sent = []

    def recv(self) -> Any:
        return self._recv_values.pop(0)

    def poll(self, _timeout: float) -> bool:
        if self._poll_values:
            return self._poll_values.pop(0)
        return True

    def send(self, value: Any) -> None:
        self.sent.append(value)

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(self, *, target: Any, args: tuple[Any, ...], alive: bool = True) -> None:
        self.target = target
        self.args = args
        self.started = False
        self.killed = False
        self._alive = alive

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self._alive

    def kill(self) -> None:
        self.killed = True
        self._alive = False


class _Engine:
    class Timeout:
        pass

    class Exception:
        pass

    class Unpicklable:
        pass

    def __init__(self) -> None:
        self.single_timeout = 5
        self.target_file = "target.py"
        self.coverage_data = "old-coverage"
        self.coverage_accumulated_missing_lines = {}
        self.in_out = []


class _PipeFactory:
    def __init__(self, pairs) -> None:
        self._pairs = list(pairs)

    def __call__(self):
        return self._pairs.pop(0)


def test_primitive_runner_parent_updates_coverage_state_and_cleans_up(monkeypatch) -> None:
    range_read = _Conn(recv_values=[{1, 2, 3}, {2, 3}, "answer"])
    range_send = _Conn()
    payload = ("new-coverage", {"target.py": {3}})
    payload_read = _Conn(recv_values=[payload])
    payload_send = _Conn()
    ready_read = _Conn(poll_values=[True])
    ready_send = _Conn()
    process_holder = []

    monkeypatch.setattr(
        primitive.multiprocessing,
        "Pipe",
        _PipeFactory(
            [
                (range_read, range_send),
                (payload_read, payload_send),
                (ready_read, ready_send),
            ]
        ),
    )

    def fake_process(*, target: Any, args: tuple[Any, ...]) -> _Process:
        process = _Process(target=target, args=args, alive=True)
        process_holder.append(process)
        return process

    monkeypatch.setattr(primitive.multiprocessing, "Process", fake_process)

    engine = _Engine()
    primitive_inputs = {"x": 1}
    result = PrimitiveExecutionRunner(engine).run(primitive_inputs)

    assert result == "answer"
    assert engine.module_lines_range == {1, 2, 3}
    assert engine.function_lines_range == {2, 3}
    assert engine.coverage_data == "new-coverage"
    assert engine.coverage_accumulated_missing_lines == {"target.py": {3}}
    assert engine.in_out == [({"x": 1}, "answer")]
    assert process_holder[0].started is True
    assert process_holder[0].killed is True
    assert all(
        conn.closed
        for conn in [range_read, range_send, payload_read, payload_send, ready_read, ready_send]
    )


def test_primitive_runner_ready_timeout_returns_timeout_and_preserves_fallback(monkeypatch) -> None:
    range_read = _Conn(recv_values=[{1, 2}, {2}])
    range_send = _Conn()
    payload_read = _Conn()
    payload_send = _Conn()
    ready_read = _Conn(poll_values=[False])
    ready_send = _Conn()
    process_holder = []

    monkeypatch.setattr(
        primitive.multiprocessing,
        "Pipe",
        _PipeFactory(
            [
                (range_read, range_send),
                (payload_read, payload_send),
                (ready_read, ready_send),
            ]
        ),
    )

    def fake_process(*, target: Any, args: tuple[Any, ...]) -> _Process:
        process = _Process(target=target, args=args, alive=True)
        process_holder.append(process)
        return process

    monkeypatch.setattr(primitive.multiprocessing, "Process", fake_process)

    engine = _Engine()
    primitive_inputs = {"x": 1}
    result = PrimitiveExecutionRunner(engine).run(primitive_inputs)

    assert result is engine.Timeout
    assert engine.module_lines_range == {1, 2}
    assert engine.function_lines_range == {2}
    assert engine.coverage_data == "old-coverage"
    assert engine.coverage_accumulated_missing_lines == {"target.py": {1, 2}}
    assert engine.in_out == [({"x": 1}, engine.Timeout)]
    assert process_holder[0].killed is True
    assert all(
        conn.closed
        for conn in [range_read, range_send, payload_read, payload_send, ready_read, ready_send]
    )


def test_exploration_engine_primitive_wrapper_lazily_delegates(monkeypatch) -> None:
    class _Runner:
        def __init__(self, engine) -> None:
            self.engine = engine
            self.calls = []

        def run(self, primitive_inputs):
            self.calls.append(primitive_inputs)
            return "runner-result"

    created = []

    def fake_runner(engine):
        runner = _Runner(engine)
        created.append(runner)
        return runner

    monkeypatch.setattr(explore, "PrimitiveExecutionRunner", fake_runner)

    engine = explore.ExplorationEngine.__new__(explore.ExplorationEngine)
    primitive_inputs = {"x": 1}
    result = engine._one_execution_primitive(primitive_inputs)

    assert result == "runner-result"
    assert len(created) == 1
    assert created[0].engine is engine
    assert created[0].calls == [primitive_inputs]
    assert engine._primitive_runner is created[0]
