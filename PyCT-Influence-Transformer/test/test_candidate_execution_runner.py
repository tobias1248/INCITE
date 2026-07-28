from __future__ import annotations

from typing import Any, Dict

from libct.executor import CandidateExecutionRunner


class _Recorder:
    def __init__(self) -> None:
        self.original_label = 0
        self.attack_label = None
        self.adversarial_input = None
        self.extra_meta = {}
        self.reference_predictions = []

    def find_adversarial_input(self, inputs: Dict[str, Any], attack_label: Any) -> None:
        self.attack_label = attack_label
        self.adversarial_input = dict(inputs)

    def mark_error(self, error_type, reason, *, phase=None, **_kwargs) -> None:
        self.extra_meta.update(
            status="error",
            error_type=error_type,
            error_reason=reason,
            error_phase=phase,
        )

    def record_reference_prediction(self, wall_time, *, phase) -> None:
        self.reference_predictions.append((wall_time, phase))


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
        self.single_coverage = False
        self.concolic_calls = []
        self.primitive_calls = []
        self.reference_execute = lambda **data: data["label"]

    def _get_recorder(self) -> _Recorder:
        return self.recorder

    def _clone_primitive_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return dict(inputs)

    def _complete_primitive_arguments(self, _func, inputs):
        return [], dict(inputs)

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
    engine._predict_reference = (  # type: ignore[attr-defined]
        lambda inputs, phase: inputs["label"]
    )

    assert runner.validate_sat_candidate({"label": 1}) is True
    assert engine.recorder.attack_label == 1
    assert engine.recorder.adversarial_input == {"label": 1}


def test_candidate_runner_initial_execution_uses_reference_then_runs_search() -> None:
    engine = _Engine()
    runner = CandidateExecutionRunner(engine)
    engine._predict_reference = (  # type: ignore[attr-defined]
        lambda inputs, phase: inputs["label"]
    )
    engine._one_execution = runner.one_execution  # type: ignore[attr-defined]

    runner.run_initial_execution({"label": 1}, {"label": 1})

    assert engine.recorder.original_label == 1
    assert engine.concolic_calls == [({"label": 1}, {"label": 1})]
    assert engine.recorder.attack_label is None
    assert not hasattr(engine, "previous_result")
    assert not hasattr(engine, "in_out")


def test_candidate_runner_marks_reference_prediction_failure() -> None:
    engine = _Engine()
    runner = CandidateExecutionRunner(engine)

    def fail_reference(**_data):
        raise ValueError("bad Keras output")

    engine.reference_execute = fail_reference

    try:
        runner.predict_reference({"label": 0}, phase="candidate_reference")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected reference prediction failure")

    assert engine.recorder.extra_meta["status"] == "error"
    assert engine.recorder.extra_meta["error_type"] == "reference_prediction_failure"
    assert engine.recorder.extra_meta["error_phase"] == "candidate_reference"
    assert engine.recorder.reference_predictions[0][1] == "candidate_reference"


def test_candidate_runner_non_coverage_execution_skips_primitive_pass() -> None:
    engine = _Engine()
    runner = CandidateExecutionRunner(engine)
    all_args = {"x": 1}

    assert runner.one_execution(all_args, {"x": 1}) is True

    assert all_args == {"x": 2}
    assert engine.concolic_calls == [({"x": 1}, {"x": 1})]
    assert engine.primitive_calls == []
    assert not hasattr(engine, "previous_result")
    assert not hasattr(engine, "in_out")
