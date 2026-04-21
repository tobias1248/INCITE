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


def test_resolve_model_artifacts_raises_when_model_file_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(executor.os.path, "isfile", lambda path: False)

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        executor._resolve_model_artifacts("missing-model")


def test_load_predictor_reuses_cached_module_entry(monkeypatch) -> None:
    executor._PREDICTOR_CACHE.clear()
    module = object()
    init_fn = object()
    predict_search_fn = object()
    predict_validation_fn = object()
    module_calls = []
    function_calls = []

    monkeypatch.setattr(
        executor,
        "get_module_from_rootdir_and_modpath",
        lambda root, module_path: module_calls.append((root, module_path)) or module,
    )
    monkeypatch.setattr(
        executor,
        "get_function_from_module_and_funcname",
        lambda mod, name: function_calls.append((mod, name)) or (
            init_fn
            if name == "init_model"
            else predict_search_fn
            if name == "predict_search"
            else predict_validation_fn
        ),
    )

    first = executor._load_predictor("/tmp/predictor_runtime.py", "/tmp/root")
    second = executor._load_predictor("/tmp/predictor_runtime.py", "/tmp/root")

    assert first == second
    assert module_calls == [("/tmp/root", "/tmp/predictor_runtime.py")]
    assert function_calls == [
        (module, "init_model"),
        (module, "predict_search"),
        (module, "predict_validation"),
    ]


def test_prepare_experiment_paths_returns_none_without_save_exp() -> None:
    assert executor._prepare_experiment_paths(
        "demo",
        "queue",
        None,
        False,
        1,
        True,
        30,
        None,
        None,
        None,
        None,
    ) == (None, None, None)


def test_prepare_experiment_paths_builds_save_and_smt_dirs(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        executor,
        "get_save_dir_from_save_exp",
        lambda **kwargs: calls.append(kwargs) or "/tmp/output",
    )

    result = executor._prepare_experiment_paths(
        "demo",
        "queue_solver1s",
        {"input_name": "case_0", "save_smt": True},
        True,
        9,
        False,
        15,
        0.8,
        2000,
        True,
        1.5,
    )

    assert result == ("/tmp/output", "/tmp/output", "case_0")
    assert len(calls) == 2
    assert calls[0]["only_first_forward"] is True
    assert calls[0]["score_alpha"] == 0.8
    assert calls[0]["symbolic_path_threshold"] == 2000
    assert calls[0]["ternary_simplification"] is True
    assert calls[0]["ternary_threshold_scale"] == 1.5


def test_run_reuses_cached_predictor_and_attaches_extra_meta(monkeypatch) -> None:
    init_calls = []
    initialized_models = set()
    captured = {}

    def fake_init_model(model_path, **kwargs):
        init_calls.append((model_path, kwargs))

    def fake_load_predictor(module_path, root):
        return object(), fake_init_model, "search-predict", "validation-predict", initialized_models

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
        lambda model_name: (f"/tmp/{model_name}.h5", "/tmp/engine/predictor_runtime.py", "/tmp/root"),
    )
    monkeypatch.setattr(executor, "_load_predictor", fake_load_predictor)
    monkeypatch.setattr(executor, "_prepare_experiment_paths", lambda *args, **kwargs: ("/tmp/save", "/tmp/smt", "case_0"))
    def fake_build_explorer(cfg):
        captured["cfg"] = cfg
        return fake_engine

    monkeypatch.setattr(executor, "_build_explorer", fake_build_explorer)
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
        ternary_simplification=True,
        ternary_threshold_scale=1.5,
        solver_run_timeout=1,
        constraint_build_timeout=True,
        constraint_build_timeout_seconds=15,
    )

    first = executor.run(**payload)
    second = executor.run(**payload)

    assert first[0] == 3
    assert second[0] == 3
    assert init_calls == [
        (
            "/tmp/demo.h5",
            {
                "ternary_simplification": False,
                "ternary_threshold_scale": 0.75,
                "role": "validation",
            },
        ),
        (
            "/tmp/demo.h5",
            {
                "ternary_simplification": True,
                "ternary_threshold_scale": 1.5,
                "role": "search",
            },
        )
    ]
    assert captured["cfg"].execute == "search-predict"
    assert captured["cfg"].validation_execute == "validation-predict"
    assert captured["extra_meta"] == {
        "model_name": "demo",
        "attack_mode": "queue_solver1s",
        "idx": 9,
        "score_alpha": 0.8,
        "symbolic_path_threshold": 2000,
        "ternary_simplification": True,
        "ternary_threshold_scale": 1.5,
        "constraint_build_timeout": True,
        "constraint_build_timeout_seconds": 15,
        "ton": 1,
        "ton_next": 2,
    }
    assert captured["explore_args"][0] == "/tmp/engine/predictor_runtime.py"
    assert captured["explore_kwargs"]["collect_constraints_with"] == "queue"
    assert captured["cleared"] is True


def test_run_uses_defaults_when_save_exp_and_optional_args_are_missing(monkeypatch) -> None:
    captured = {}
    init_calls = []

    class _FakeEngine:
        extra_meta = None

        def explore(self, *args, **kwargs):
            captured["explore_kwargs"] = kwargs
            captured["extra_meta"] = self.extra_meta
            return (1, SimpleNamespace())

    monkeypatch.setattr(
        executor,
        "_resolve_model_artifacts",
        lambda model_name: (f"/tmp/{model_name}.h5", "/tmp/engine/predictor_runtime.py", "/tmp/root"),
    )
    monkeypatch.setattr(
        executor,
        "_load_predictor",
        lambda module_path, root: (
            object(),
            lambda model_path, **kwargs: init_calls.append((model_path, kwargs)),
            "search-predict",
            "validation-predict",
            set(),
        ),
    )
    monkeypatch.setattr(executor, "_prepare_experiment_paths", lambda *args, **kwargs: (None, None, None))

    def fake_build_explorer(cfg):
        captured["cfg"] = cfg
        return _FakeEngine()

    monkeypatch.setattr(executor, "_build_explorer", fake_build_explorer)
    monkeypatch.setattr(executor.libct.explore, "clear_global_context", lambda: None)

    result = executor.run(
        model_name="demo",
        in_dict={"v_0_0": 1.0},
        con_dict={"v_0_0": 1},
        norm=True,
        solve_order_stack=True,
        idx=1,
        collect_constraints_with="stack",
    )

    assert result[0] == 1
    assert init_calls == [
        (
            "/tmp/demo.h5",
            {
                "ternary_simplification": False,
                "ternary_threshold_scale": 0.75,
                "role": "validation",
            },
        )
    ]
    assert captured["cfg"].execute == "validation-predict"
    assert captured["cfg"].validation_execute == "validation-predict"
    assert captured["extra_meta"] == {
        "model_name": "demo",
        "attack_mode": "unknown",
        "idx": 1,
        "score_alpha": None,
        "symbolic_path_threshold": None,
        "ternary_simplification": False,
        "ternary_threshold_scale": 0.75,
        "constraint_build_timeout": True,
        "constraint_build_timeout_seconds": 30,
    }
    assert captured["explore_kwargs"]["collect_constraints_with"] == "stack"
    assert captured["explore_kwargs"]["shap_value_pre_calculated"] is False


def test_run_distinguishes_initialized_models_by_ternary_runtime(monkeypatch) -> None:
    init_calls = []
    initialized_models = set()

    def fake_init_model(model_path, **kwargs):
        init_calls.append((model_path, kwargs))

    class _FakeEngine:
        extra_meta = None

        def explore(self, *args, **kwargs):
            return (1, SimpleNamespace())

    monkeypatch.setattr(
        executor,
        "_resolve_model_artifacts",
        lambda model_name: (f"/tmp/{model_name}.h5", "/tmp/engine/predictor_runtime.py", "/tmp/root"),
    )
    monkeypatch.setattr(
        executor,
        "_load_predictor",
        lambda module_path, root: (
            object(),
            fake_init_model,
            "search-predict",
            "validation-predict",
            initialized_models,
        ),
    )
    monkeypatch.setattr(executor, "_prepare_experiment_paths", lambda *args, **kwargs: (None, None, None))
    monkeypatch.setattr(executor, "_build_explorer", lambda cfg: _FakeEngine())
    monkeypatch.setattr(executor.libct.explore, "clear_global_context", lambda: None)

    base_payload = dict(
        model_name="demo",
        in_dict={"v_0_0": 1.0},
        con_dict={"v_0_0": 1},
        norm=True,
        solve_order_stack=False,
        idx=1,
        collect_constraints_with="queue",
    )

    executor.run(**base_payload)
    executor.run(**base_payload, ternary_simplification=True)
    executor.run(**base_payload, ternary_simplification=True, ternary_threshold_scale=1.5)

    assert init_calls == [
        ("/tmp/demo.h5", {"ternary_simplification": False, "ternary_threshold_scale": 0.75, "role": "validation"}),
        ("/tmp/demo.h5", {"ternary_simplification": True, "ternary_threshold_scale": 0.75, "role": "search"}),
        ("/tmp/demo.h5", {"ternary_simplification": True, "ternary_threshold_scale": 1.5, "role": "search"}),
    ]
