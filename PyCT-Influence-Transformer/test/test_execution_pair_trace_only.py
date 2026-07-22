from __future__ import annotations

from types import SimpleNamespace

from libct.executor.execution_pair import CandidateExecutionRunner


def _engine(*, trace_only: bool):
    timeout = object()
    recorder = SimpleNamespace(original_label="unset")
    validation_calls = []
    engine = SimpleNamespace(
        trace_only=trace_only,
        idx=4,
        Timeout=timeout,
        Exception=object(),
        Unpicklable=object(),
        in_out=[],
        reuse_search_result_for_validation=True,
        single_coverage=False,
        previous_result=None,
    )
    engine._get_recorder = lambda: recorder
    engine._candidate_execution_can_validate = lambda: True
    engine._one_execution_deferred_constraints = lambda _args, _concolic: (timeout, None)
    engine._is_valid_label_result = lambda _result: False
    engine._predict_validation = lambda _args: validation_calls.append(True) or 8
    engine._apply_constraint_transfer_payload = lambda _payload: None
    engine._record_result = lambda _args, result: setattr(engine, "previous_result", result)
    return engine, recorder, validation_calls


def test_trace_only_timeout_skips_label_fallback() -> None:
    engine, recorder, validation_calls = _engine(trace_only=True)

    CandidateExecutionRunner(engine).run_initial_execution({"x": 1}, {"x": 1})

    assert validation_calls == []
    assert recorder.original_label is None
    assert engine.previous_result is engine.Timeout


def test_normal_timeout_keeps_label_fallback() -> None:
    engine, recorder, validation_calls = _engine(trace_only=False)

    CandidateExecutionRunner(engine).run_initial_execution({"x": 1}, {"x": 1})

    assert validation_calls == [True]
    assert recorder.original_label == 8
