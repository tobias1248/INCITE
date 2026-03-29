from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import engine.executor as executor


def test_validate_collect_mode_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported collect_constraints_with"):
        executor._validate_collect_mode("fifo")


def test_run_reuses_cached_predictor_and_attaches_extra_meta(monkeypatch) -> None:
    init_calls = []
    initialized_models = set()
    captured = {}

    def fake_init_model(model_path):
        init_calls.append(model_path)

    def fake_load_predictor(module_path, root):
        return object(), fake_init_model, object(), initialized_models

    class _FakeEngine:
        extra_meta = None

        def explore(self, *args, **kwargs):
            captured["explore_args"] = args
            captured["explore_kwargs"] = kwargs
            captured["extra_meta"] = self.extra_meta
            return (3, SimpleNamespace())

    fake_engine = _FakeEngine()

    monkeypatch.setattr(
        executor,
        "_resolve_model_artifacts",
        lambda model_name: (f"/tmp/{model_name}.h5", "/tmp/dnn_predict_common.py", "/tmp/root"),
    )
    monkeypatch.setattr(executor, "_load_predictor", fake_load_predictor)
    monkeypatch.setattr(executor, "_prepare_experiment_paths", lambda *args, **kwargs: ("/tmp/save", "/tmp/smt", "case_0"))
    monkeypatch.setattr(executor, "_build_explorer", lambda cfg: fake_engine)
    monkeypatch.setattr(executor.libct.explore, "clear_global_context", lambda: captured.setdefault("cleared", True))

    payload = dict(
        model_name="demo",
        in_dict={"v_0_0": 1.0},
        con_dict={"v_0_0": 1},
        norm=False,
        solve_order_stack=False,
        idx=9,
        save_exp={"attack_mode": "queue_solver1s", "ton": 1, "ton_next": 2},
        collect_constraints_with="queue",
        popped_log_attack_mode="queue_solver1s",
        score_alpha=0.8,
        symbolic_path_threshold=2000,
        solver_run_timeout=1,
        constraint_build_timeout=True,
        constraint_build_timeout_seconds=15,
    )

    first = executor.run(**payload)
    second = executor.run(**payload)

    assert first[0] == 3
    assert second[0] == 3
    assert init_calls == ["/tmp/demo.h5"]
    assert captured["extra_meta"] == {
        "model_name": "demo",
        "attack_mode": "queue_solver1s",
        "idx": 9,
        "score_alpha": 0.8,
        "symbolic_path_threshold": 2000,
        "constraint_build_timeout": True,
        "constraint_build_timeout_seconds": 15,
        "ton": 1,
        "ton_next": 2,
    }
    assert captured["explore_args"][0] == "/tmp/dnn_predict_common.py"
    assert captured["explore_kwargs"]["collect_constraints_with"] == "queue"
    assert captured["cleared"] is True
