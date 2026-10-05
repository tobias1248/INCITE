from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import engine.executor as executor
from libct.global_real import GLOBAL_X_INPUT_NAME, materialize_global_real_arguments
from libct.global_real_de import GlobalRealDEResult


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
    init_reference_fn = object()
    init_fn = object()
    predict_search_fn = object()
    predict_reference_fn = object()
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
            init_reference_fn
            if name == "init_reference_model"
            else init_fn
            if name == "init_model"
            else predict_search_fn
            if name == "predict_search"
            else predict_reference_fn
        ),
    )

    first = executor._load_predictor("/tmp/predictor_runtime.py", "/tmp/root")
    second = executor._load_predictor("/tmp/predictor_runtime.py", "/tmp/root")

    assert first == second
    assert module_calls == [("/tmp/root", "/tmp/predictor_runtime.py")]
    assert function_calls == [
        (module, "init_reference_model"),
        (module, "init_model"),
        (module, "predict_search"),
        (module, "predict_reference"),
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
    reference_init_calls = []
    init_calls = []
    initialized_models = set()
    captured = {}

    def fake_init_model(model_path, **kwargs):
        init_calls.append((model_path, kwargs))

    def fake_load_predictor(module_path, root):
        return (
            object(),
            lambda path: reference_init_calls.append(path),
            fake_init_model,
            "search-predict",
            "reference-predict",
            initialized_models,
        )

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
        save_exp={
            "attack_mode": "queue_solver1s",
            "ton": 1,
            "ton_next": 2,
            "fallback": True,
            "fallback_type": "ternary",
            "fallback_trigger": "timeout",
            "fallback_source_attack_mode": "queue_solver1s",
            "fallback_source_ton": 1,
            "fallback_source_ton_next": 2,
        },
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
    assert reference_init_calls == ["/tmp/demo.h5", "/tmp/demo.h5"]
    assert init_calls == [
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
    assert captured["cfg"].reference_execute == "reference-predict"
    assert captured["extra_meta"] == {
        "model_name": "demo",
        "attack_mode": "queue_solver1s",
        "smt_path_mode": "full",
        "idx": 9,
        "score_alpha": 0.8,
        "symbolic_path_threshold": 2000,
        "ternary_simplification": True,
        "ternary_threshold_scale": 1.5,
        "constraint_build_timeout": True,
        "constraint_build_timeout_seconds": 15,
        "label_source": "keras_model_predict",
        "search_model": "NNModel",
        "ton": 1,
        "ton_next": 2,
        "fallback": True,
        "fallback_type": "ternary",
        "fallback_trigger": "timeout",
        "fallback_source_attack_mode": "queue_solver1s",
        "fallback_source_ton": 1,
        "fallback_source_ton_next": 2,
    }
    assert captured["explore_args"][0] == "/tmp/engine/predictor_runtime.py"
    assert captured["explore_kwargs"]["collect_constraints_with"] == "queue"
    assert captured["cleared"] is True


def test_run_uses_defaults_when_save_exp_and_optional_args_are_missing(monkeypatch) -> None:
    captured = {}
    reference_init_calls = []
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
            lambda path: reference_init_calls.append(path),
            lambda model_path, **kwargs: init_calls.append((model_path, kwargs)),
            "search-predict",
            "reference-predict",
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
    assert reference_init_calls == ["/tmp/demo.h5"]
    assert init_calls == [
        (
            "/tmp/demo.h5",
            {
                "ternary_simplification": False,
                "ternary_threshold_scale": 0.75,
                "role": "search",
            },
        )
    ]
    assert captured["cfg"].execute == "search-predict"
    assert captured["cfg"].reference_execute == "reference-predict"
    assert captured["extra_meta"] == {
        "model_name": "demo",
        "attack_mode": "unknown",
        "smt_path_mode": "full",
        "idx": 1,
        "score_alpha": None,
        "symbolic_path_threshold": None,
        "ternary_simplification": False,
        "ternary_threshold_scale": 0.75,
        "constraint_build_timeout": True,
        "constraint_build_timeout_seconds": 30,
        "label_source": "keras_model_predict",
        "search_model": "NNModel",
    }
    assert captured["explore_kwargs"]["collect_constraints_with"] == "stack"
    assert captured["explore_kwargs"]["shap_value_pre_calculated"] is False


def test_run_distinguishes_initialized_models_by_ternary_runtime(monkeypatch) -> None:
    reference_init_calls = []
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
            lambda path: reference_init_calls.append(path),
            fake_init_model,
            "search-predict",
            "reference-predict",
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

    assert reference_init_calls == ["/tmp/demo.h5", "/tmp/demo.h5", "/tmp/demo.h5"]
    assert init_calls == [
        ("/tmp/demo.h5", {"ternary_simplification": False, "ternary_threshold_scale": 0.75, "role": "search"}),
        ("/tmp/demo.h5", {"ternary_simplification": True, "ternary_threshold_scale": 0.75, "role": "search"}),
        ("/tmp/demo.h5", {"ternary_simplification": True, "ternary_threshold_scale": 1.5, "role": "search"}),
    ]


def test_run_fails_closed_when_reference_model_cannot_load(monkeypatch) -> None:
    def fail_reference(_path):
        raise ValueError("cannot load Keras model")

    monkeypatch.setattr(
        executor,
        "_resolve_model_artifacts",
        lambda model_name: (
            f"/tmp/{model_name}.h5",
            "/tmp/engine/predictor_runtime.py",
            "/tmp/root",
        ),
    )
    monkeypatch.setattr(
        executor,
        "_load_predictor",
        lambda module_path, root: (
            object(),
            fail_reference,
            lambda *_args, **_kwargs: None,
            "search-predict",
            "reference-predict",
            set(),
        ),
    )
    monkeypatch.setattr(
        executor,
        "_prepare_experiment_paths",
        lambda *args, **kwargs: (None, None, "case_0"),
    )
    monkeypatch.setattr(
        executor,
        "_build_explorer",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("unexpected explorer")),
    )

    iterations, recorder = executor.run(
        model_name="demo",
        in_dict={"v_0_0": 1.0},
        con_dict={"v_0_0": 1},
        norm=True,
        solve_order_stack=False,
        idx=0,
        collect_constraints_with="queue",
    )

    assert iterations == 0
    assert recorder.original_input is not None
    assert recorder.extra_meta["status"] == "error"
    assert recorder.extra_meta["error_type"] == "reference_prediction_failure"
    assert recorder.extra_meta["error_phase"] == "reference_model_load"


def test_hybrid_de_failure_uses_last_de_image_as_pyct_seed(monkeypatch, tmp_path) -> None:
    source = np.asarray(
        [[[0.2, 0.4, 0.6]], [[0.25, 0.45, 0.65]]],
        dtype=np.float32,
    )
    seed = (
        source + 0.05 * (source - source.mean(axis=(0, 1), keepdims=True))
    ).astype(np.float32)
    captured = {}

    class _FakeEngine:
        extra_meta = None

        def explore(self, *args, **kwargs):
            captured["in_dict"] = args[1]
            captured["concolic_dict"] = kwargs["concolic_dict"]
            captured["global_real_config"] = kwargs["global_real_config"]
            captured["extra_meta"] = self.extra_meta
            return (1, SimpleNamespace(
                record_hybrid_de_inputs=lambda *values: captured.setdefault(
                    "recorded_hybrid_inputs", values
                ),
                save_stats_dict=lambda: captured.setdefault("stats_saved", True),
            ))

    module = SimpleNamespace(
        predict_reference_batch=lambda images: np.tile(
            np.asarray([[0.9] + [0.1 / 9.0] * 9], dtype=np.float64),
            (len(images), 1),
        )
    )
    monkeypatch.setattr(
        executor,
        "_resolve_model_artifacts",
        lambda _name: ("/tmp/demo.h5", "/tmp/predictor_runtime.py", "/tmp/root"),
    )
    monkeypatch.setattr(
        executor,
        "_load_predictor",
        lambda *_args: (
            module,
            lambda _path: None,
            lambda *_args, **_kwargs: None,
            "search-predict",
            "reference-predict",
            set(),
        ),
    )
    monkeypatch.setattr(
        executor,
        "_prepare_experiment_paths",
        lambda *_args, **_kwargs: (str(tmp_path), None, "case_0"),
    )
    monkeypatch.setattr(executor, "_build_explorer", lambda _cfg: _FakeEngine())
    monkeypatch.setattr(executor.libct.explore, "clear_global_context", lambda: None)
    monkeypatch.setattr(
        executor,
        "run_global_real_differential_evolution",
        lambda *_args, **_kwargs: GlobalRealDEResult(
            success=False,
            original_label=0,
            best_label=0,
            best_x=0.05,
            best_score=0.7,
            best_margin=0.4,
            best_image=seed,
            iterations=75,
            function_evaluations=30400,
        ),
    )

    config = {
        "variable_name": GLOBAL_X_INPUT_NAME,
        "requested_min": -0.1,
        "requested_max": 0.1,
        "effective_min": -0.1,
        "effective_max": 0.1,
        "bounds_mode": "clip",
        "global_shift_kind": "contrast",
        "coefficient_by_input": {
            "v_0_0_0": -0.1,
            "v_0_0_1": -0.1,
            "v_0_0_2": -0.1,
            "v_1_0_0": 0.1,
            "v_1_0_1": 0.1,
            "v_1_0_2": 0.1,
        },
        "hybrid_de_enabled": True,
        "hybrid_de_maxiter": 75,
        "hybrid_de_population_size": 400,
        "hybrid_de_random_seed": 2024,
    }
    recorder = executor.run(
        model_name="demo",
        in_dict={
            "v_0_0_0": float(source[0, 0, 0]),
            "v_0_0_1": float(source[0, 0, 1]),
            "v_0_0_2": float(source[0, 0, 2]),
            "v_1_0_0": float(source[1, 0, 0]),
            "v_1_0_1": float(source[1, 0, 1]),
            "v_1_0_2": float(source[1, 0, 2]),
            GLOBAL_X_INPUT_NAME: 0.0,
        },
        con_dict={GLOBAL_X_INPUT_NAME: 1},
        norm=True,
        solve_order_stack=False,
        idx=0,
        popped_log_attack_mode="hybrid-de_contrast",
        global_real_config=config,
        input_for_shap=source,
    )

    assert recorder[0] == 1
    assert captured["in_dict"]["v_0_0_0"] == pytest.approx(source[0, 0, 0])
    assert captured["in_dict"]["v_1_0_2"] == pytest.approx(source[1, 0, 2])
    assert GLOBAL_X_INPUT_NAME not in captured["in_dict"]
    assert captured["in_dict"]["__pyct_brightness"] == 0.0
    assert captured["in_dict"]["__pyct_contrast"] == pytest.approx(0.05)
    assert captured["concolic_dict"] == {
        "__pyct_brightness": 1,
        "__pyct_contrast": 1,
    }
    assert captured["global_real_config"]["coefficient_by_input"] == pytest.approx(
        {
            "v_0_0_0": -0.025,
            "v_0_0_1": -0.025,
            "v_0_0_2": -0.025,
            "v_1_0_0": 0.025,
            "v_1_0_1": 0.025,
            "v_1_0_2": 0.025,
        }
    )
    assert captured["global_real_config"]["contrast_channel_means"] == pytest.approx(
        [0.225, 0.425, 0.625]
    )
    assert captured["extra_meta"]["hybrid_de_iterations"] == 75
    assert captured["extra_meta"]["hybrid_de_best_margin"] == pytest.approx(0.4)
    assert captured["global_real_config"]["transform_mode"] == "affine-brightness-contrast"
    materialized, _, _ = materialize_global_real_arguments(
        captured["in_dict"], captured["global_real_config"]
    )
    np.testing.assert_allclose(executor._image_from_input_dict(materialized), seed, atol=1e-7)
    assert captured["recorded_hybrid_inputs"][1] is not None
    assert captured["stats_saved"] is True
    np.testing.assert_array_equal(recorder[1].original_input, source)


def test_record_hybrid_de_success_without_starting_pyct() -> None:
    source = np.zeros((1, 1, 3), dtype=np.float32)
    adv = np.ones((1, 1, 3), dtype=np.float32)
    de_result = GlobalRealDEResult(
        success=True,
        original_label=0,
        best_label=1,
        best_x=0.1,
        best_score=0.1,
        best_margin=-0.8,
        best_image=adv,
        iterations=3,
        function_evaluations=1600,
    )

    _, recorder = executor._record_hybrid_de_success(
        save_dir=None,
        input_name="case_0",
        source_image=source,
        de_result=de_result,
        de_wall_time=1.5,
        de_cpu_time=1.0,
        global_real_config={"variable_name": GLOBAL_X_INPUT_NAME},
        extra_meta={"hybrid_de_status": "success"},
    )

    assert recorder.original_label == 0
    assert recorder.attack_label == 1
    np.testing.assert_array_equal(recorder.original_input, source)
    np.testing.assert_array_equal(recorder.adversarial_input, adv)
    assert recorder.extra_meta["hybrid_de_status"] == "success"


def test_run_attaches_complete_aces_like_pwl_metadata(monkeypatch) -> None:
    captured = {}

    class _FakeEngine:
        extra_meta = None

        def explore(self, *args, **kwargs):
            captured["extra_meta"] = self.extra_meta
            return (1, SimpleNamespace())

    monkeypatch.setattr(
        executor,
        "_resolve_model_artifacts",
        lambda model_name: (
            f"/tmp/{model_name}.h5",
            "/tmp/engine/predictor_runtime.py",
            "/tmp/root",
        ),
    )
    monkeypatch.setattr(
        executor,
        "_load_predictor",
        lambda module_path, root: (
            object(),
            lambda path: None,
            lambda model_path, **kwargs: None,
            "search-predict",
            "reference-predict",
            set(),
        ),
    )
    monkeypatch.setattr(
        executor,
        "_prepare_experiment_paths",
        lambda *args, **kwargs: (None, None, None),
    )
    monkeypatch.setattr(executor, "_build_explorer", lambda cfg: _FakeEngine())
    monkeypatch.setattr(executor.libct.explore, "clear_global_context", lambda: None)

    global_real_config = {
        "requested_min": -0.1,
        "requested_max": 0.1,
        "effective_min": -0.1,
        "effective_max": 0.1,
        "bounds_mode": "clip",
        "transform_mode": "aces-like-pwl",
        "global_shift_kind": "aces-brightness",
        "pwl_knots": [-0.1, 0.0, 0.1],
        "pwl_max_segments": 32,
        "pwl_segment_count": 2,
        "pwl_error_tolerance": 1.0 / 255.0,
        "pwl_max_abs_error": 0.001,
        "pwl_error_metric": "sampled-max-abs-rgb",
        "pwl_validator_version": "adaptive-31-point-v1",
        "aces_like_color_space": "OKLCh-sRGB",
        "aces_like_curve_version": "oklch-logit-v1",
        "aces_like_gamut_mapper": "css-color-4-local-minde-v1",
    }

    executor.run(
        model_name="demo",
        in_dict={"v_0": 0.5},
        con_dict={"v_0": 1},
        norm=True,
        solve_order_stack=False,
        idx=0,
        collect_constraints_with="queue",
        global_real_config=global_real_config,
    )

    metadata = captured["extra_meta"]
    assert metadata["global_real_transform_mode"] == "aces-like-pwl"
    assert metadata["global_real_pwl_knots"] == [-0.1, 0.0, 0.1]
    assert metadata["global_real_pwl_max_segments"] == 32
    assert metadata["global_real_pwl_segment_count"] == 2
    assert metadata["global_real_pwl_error_tolerance"] == pytest.approx(1.0 / 255.0)
    assert metadata["global_real_pwl_max_abs_error"] == pytest.approx(0.001)
    assert metadata["global_real_pwl_error_metric"] == "sampled-max-abs-rgb"
    assert metadata["global_real_pwl_validator_version"] == "adaptive-31-point-v1"
    assert metadata["global_real_aces_like_color_space"] == "OKLCh-sRGB"
    assert metadata["global_real_aces_like_curve_version"] == "oklch-logit-v1"
    assert metadata["global_real_aces_like_gamut_mapper"] == (
        "css-color-4-local-minde-v1"
    )
