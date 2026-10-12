from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from libct.executor import CandidateExecutionRunner, LegacyConcolicExecutor
from libct.executor import execution_pair
from libct.global_real import (
    GLOBAL_X_INPUT_NAME,
    build_aces_like_global_real_config,
    materialize_global_real_arguments,
)
from libct.record import ConcolicTestRecorder


@pytest.mark.parametrize("exact_flips", [False, True])
def test_standalone_no_probe_validates_and_saves_exact_candidate(monkeypatch, exact_flips):
    source = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float64)
    config = build_aces_like_global_real_config(source, {
        "variable_name": GLOBAL_X_INPUT_NAME,
        "effective_min": -0.1, "effective_max": 0.1, "bounds_mode": "clip",
        "global_shift_kind": "aces-brightness", "pwl_max_segments": 32,
        "pwl_error_tolerance": 1.0 / 255.0, "probe_enabled": False,
    })
    shift = (config["pwl_knots"][0] + config["pwl_knots"][1]) / 2
    candidate = {f"v_0_0_{i}": float(source[0, 0, i]) for i in range(3)}
    candidate[GLOBAL_X_INPUT_NAME] = shift
    exact, _, _ = materialize_global_real_arguments(candidate, config)
    approximate, _, _ = materialize_global_real_arguments(candidate, config, exact_transform=False)
    channel = max(exact, key=lambda name: abs(exact[name] - approximate[name]))
    assert abs(exact[channel] - approximate[channel]) > 1e-7
    midpoint = (exact[channel] + approximate[channel]) / 2

    def reference(**kwargs):
        matches_approximate = ((kwargs[channel] > midpoint) == (approximate[channel] > midpoint))
        return 1 if (exact_flips or matches_approximate) else 0

    assert reference(**approximate) == 1
    recorder = ConcolicTestRecorder(None, "case_0")
    recorder.global_real_config = config
    recorder.input_shape = source.shape
    recorder.original_label = 0
    engine = SimpleNamespace(global_real_config=config, reference_execute=reference, _get_recorder=lambda: recorder)
    runner = CandidateExecutionRunner(engine)
    calls = []

    def predict(inputs, *, phase):
        args, kwargs = runner.complete_primitive_arguments(reference, inputs)
        calls.append(kwargs)
        return reference(*args, **kwargs)

    engine._predict_reference = predict
    monkeypatch.setattr(execution_pair, "probe_scalar_domain", lambda *a, **kw: pytest.fail("No Probe called scalar probe"))
    monkeypatch.setattr(execution_pair, "probe_vector_domain", lambda *a, **kw: pytest.fail("No Probe called vector probe"))
    assert runner.validate_sat_candidate(candidate) is exact_flips
    assert calls == [exact]
    assert candidate[GLOBAL_X_INPUT_NAME] == shift
    if exact_flips:
        expected = np.asarray([[[exact[f"v_0_0_{i}"] for i in range(3)]]], dtype=np.float32)
        np.testing.assert_array_equal(recorder.adversarial_input, expected)
    else:
        assert recorder.attack_label is None
        assert recorder.adversarial_input is None


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
