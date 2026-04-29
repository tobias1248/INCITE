from __future__ import annotations

from libct.executor import LegacyConcolicExecutor


class _Engine:
    def __init__(self) -> None:
        self.calls = []

    def _one_execution_concolic(self, all_args, concolic_dict):
        self.calls.append(("concolic", dict(all_args), dict(concolic_dict)))
        all_args["x"] = 2
        return "concolic-result"

    def _one_execution_primitive(self, primitive_inputs):
        self.calls.append(("primitive", dict(primitive_inputs)))
        return "primitive-result"


def test_legacy_executor_delegates_to_engine_methods() -> None:
    engine = _Engine()
    executor = LegacyConcolicExecutor(engine)
    all_args = {"x": 1}

    concolic_result = executor.run_concolic(all_args, {"x": 1})
    primitive_result = executor.run_primitive({"x": 1})

    assert concolic_result == "concolic-result"
    assert primitive_result == "primitive-result"
    assert all_args == {"x": 2}
    assert engine.calls == [
        ("concolic", {"x": 1}, {"x": 1}),
        ("primitive", {"x": 1}),
    ]
