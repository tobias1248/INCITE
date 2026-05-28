from __future__ import annotations

from typing import Any, Dict

from libct.executor import CandidateExecutionRunner


class _Recorder:
    def __init__(self) -> None:
        self.original_label = 0
        self.attack_label = None
        self.adversarial_input = None

    def find_adversarial_input(self, inputs: Dict[str, Any], attack_label: Any) -> None:
        self.attack_label = attack_label
        self.adversarial_input = dict(inputs)


class _Engine:
    class Timeout:
        pass

    class Exception:
        pass

    class Unpicklable:
        pass

    class LazyLoading:
        pass

    def __init__(self) -> None:
        self.recorder = _Recorder()
        self.previous_result = None
        self.reuse_search_result_for_validation = True
        self.single_coverage = False
        self.in_out = []
        self.concolic_calls = []
        self.primitive_calls = []

    def _get_recorder(self) -> _Recorder:
        return self.recorder

    def _clone_primitive_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return dict(inputs)

    def _record_result(self, inputs: Dict[str, Any], result: Any) -> bool:
        self.previous_result = result
        return True

    def _one_execution_concolic(self, all_args: Dict[str, Any], concolic_dict: Dict[str, Any]) -> int:
        self.concolic_calls.append((dict(all_args), dict(concolic_dict)))
        all_args["x"] = 2
        return 0

    def _one_execution_primitive(self, primitive_inputs: Dict[str, Any]) -> int:
        self.primitive_calls.append(dict(primitive_inputs))
        return 0


def test_candidate_runner_detects_validated_adversarial_input() -> None:
    engine = _Engine()
    runner = CandidateExecutionRunner(engine)
    engine._predict_validation = lambda inputs: inputs["label"]  # type: ignore[attr-defined]

    assert runner.validate_sat_candidate({"label": 1}) is True
    assert engine.recorder.attack_label == 1
    assert engine.recorder.adversarial_input == {"label": 1}


def test_candidate_runner_ignores_sentinel_search_result() -> None:
    engine = _Engine()
    runner = CandidateExecutionRunner(engine)

    assert runner.search_result_changes_label({"x": 1}, engine.Timeout) is False
    assert engine.recorder.attack_label is None


def test_candidate_runner_non_coverage_execution_skips_primitive_pass() -> None:
    engine = _Engine()
    runner = CandidateExecutionRunner(engine)
    all_args = {"x": 1}

    assert runner.one_execution(all_args, {"x": 1}) is True

    assert all_args == {"x": 2}
    assert engine.concolic_calls == [({"x": 1}, {"x": 1})]
    assert engine.primitive_calls == []
    assert engine.in_out == [({"x": 2}, 0)]
    assert engine.previous_result == 0
