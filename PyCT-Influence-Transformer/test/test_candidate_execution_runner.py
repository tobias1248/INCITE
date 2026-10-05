from __future__ import annotations

from typing import Any, Dict

import pytest

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

    def record_global_real_probe(self, result, *, wall_time) -> None:
        self.extra_meta.update(
            global_real_probe_evaluated_x=list(result.evaluated_x),
            global_real_probe_wall_time_total=wall_time,
            global_real_probe_bracket_count=result.bracket_count,
            global_real_probe_refinement_steps=result.refinement_steps,
            global_real_probe_success=result.success,
        )


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


def _probe_config() -> Dict[str, Any]:
    return {
        "variable_name": "x",
        "effective_min": -0.1,
        "effective_max": 0.1,
        "transform_mode": "aces-like-pwl",
        "global_shift_kind": "aces-brightness",
        "probe_enabled": True,
        "probe_initial_points": 17,
        "probe_max_refinements": 8,
        "probe_tolerance_fraction": 1.0 / 1024.0,
    }


def test_candidate_runner_probes_global_real_sat_candidate() -> None:
    engine = _Engine()
    engine.recorder.original_label = 0
    engine.global_real_config = _probe_config()
    engine._predict_reference = (  # type: ignore[attr-defined]
        lambda inputs, phase: int(inputs["x"] >= 0.04)
    )

    runner = CandidateExecutionRunner(engine)

    assert runner.validate_sat_candidate({"x": 0.0}) is True
    assert engine.recorder.attack_label == 1
    assert 0.04 <= engine.recorder.adversarial_input["x"] <= 0.041
    assert engine.recorder.extra_meta["global_real_probe_success"] is True
    assert engine.recorder.extra_meta["global_real_probe_bracket_count"] >= 1
    assert all(
        phase == "candidate_probe"
        for _wall_time, phase in engine.recorder.reference_predictions
    )


def test_candidate_runner_returns_false_after_global_real_probe_exhaustion() -> None:
    engine = _Engine()
    engine.recorder.original_label = 0
    engine.global_real_config = _probe_config()
    engine._predict_reference = (  # type: ignore[attr-defined]
        lambda _inputs, phase: 0
    )

    runner = CandidateExecutionRunner(engine)

    assert runner.validate_sat_candidate({"x": 0.0}) is False
    assert engine.recorder.attack_label is None
    assert engine.recorder.extra_meta["global_real_probe_success"] is False
    assert len(engine.recorder.extra_meta["global_real_probe_evaluated_x"]) <= 17


def test_hybrid_aces_probe_preserves_sat_margin_for_subsequent_branches() -> None:
    engine = _Engine()
    engine.global_real_config = {**_probe_config(), "hybrid_de_enabled": True}
    engine.current_reference_margin = None
    runner = CandidateExecutionRunner(engine)
    seen = []

    def predict_margin(inputs, *, phase, original_label):
        seen.append((inputs["x"], phase, original_label))
        return 0, 0.3 + inputs["x"]

    runner._predict_hybrid_margin = predict_margin
    candidate = {"x": 0.03}

    assert runner.validate_sat_candidate(candidate) is False

    assert len(seen) > 1
    assert seen[-1][0] != candidate["x"]
    assert all(source_label == 0 for _, _, source_label in seen)
    assert engine.current_reference_margin == pytest.approx(0.33)
    assert engine.recorder.extra_meta["hybrid_pyct_last_margin"] == pytest.approx(0.33)
    assert engine.recorder.extra_meta["hybrid_pyct_best_margin"] == pytest.approx(0.2)
    assert engine.recorder.extra_meta["hybrid_pyct_candidate_count"] == 1
    assert engine.recorder.extra_meta["hybrid_pyct_probe_count"] == len(seen)
    assert candidate == {"x": 0.03}


def test_hybrid_aces_probe_records_success_against_original_source_label() -> None:
    engine = _Engine()
    engine.global_real_config = {**_probe_config(), "hybrid_de_enabled": True}
    runner = CandidateExecutionRunner(engine)

    def predict_margin(inputs, *, phase, original_label):
        assert original_label == 0
        return (1, -0.1) if inputs["x"] >= 0.04 else (0, 0.1)

    runner._predict_hybrid_margin = predict_margin

    assert runner.validate_sat_candidate({"x": 0.0}) is True
    assert engine.recorder.attack_label == 1
    assert 0.04 <= engine.recorder.adversarial_input["x"] <= 0.041


def test_hybrid_aces_without_probe_updates_sat_margin_once() -> None:
    engine = _Engine()
    engine.global_real_config = {
        **_probe_config(), "hybrid_de_enabled": True, "probe_enabled": False,
    }
    runner = CandidateExecutionRunner(engine)
    seen = []

    def predict_margin(inputs, *, phase, original_label):
        seen.append((dict(inputs), phase, original_label))
        return 0, 0.07

    runner._predict_hybrid_margin = predict_margin

    assert runner.validate_sat_candidate({"x": 0.02}) is False
    assert seen == [({"x": 0.02}, "candidate_reference", 0)]
    assert engine.current_reference_margin == pytest.approx(0.07)
    assert engine.recorder.extra_meta["hybrid_pyct_candidate_count"] == 1
    assert "hybrid_pyct_probe_count" not in engine.recorder.extra_meta
