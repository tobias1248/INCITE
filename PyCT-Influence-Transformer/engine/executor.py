#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Literal, Optional, Set, Tuple

import libct.explore
import numpy as np

from libct.global_real import (
    GLOBAL_X_INPUT_NAME,
    GLOBAL_BRIGHTNESS_INPUT_NAME,
    GLOBAL_CONTRAST_INPUT_NAME,
    TRANSFORM_MODE_AFFINE_BC,
    build_aces_like_global_real_config,
)
from libct.aces_like import ACES_LIKE_SHIFT_KINDS
from libct.global_real_de import (
    coefficients_for_shift,
    run_global_real_differential_evolution,
)
from libct.record import ConcolicTestRecorder
from libct.solver import resolve_smt_path_mode
from libct.utils import (
    get_function_from_module_and_funcname,
    get_in_dict_shape,
    get_module_from_rootdir_and_modpath,
)
from tasks.paths import get_save_dir_from_save_exp

PYCT_ROOT = "./"
MODEL_ROOT = os.path.join(PYCT_ROOT, "model")
VALID_COLLECT_MODES = {"priority_queue", "queue", "stack"}
DEFAULT_SOLVER = "cvc5"
ModelRuntimeKey = Tuple[str, bool, float]
PredictorCacheEntry = Tuple[
    ModuleType,
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
    Set[ModelRuntimeKey],
]
_PREDICTOR_CACHE: Dict[Tuple[str, str], PredictorCacheEntry] = {}


def _image_from_input_dict(input_dict: Dict[str, Any]) -> np.ndarray:
    shape = get_in_dict_shape(input_dict)
    if not shape:
        raise ValueError("hybrid-de requires a non-empty image input")
    image = np.zeros(shape, dtype=np.float32)
    for name, value in input_dict.items():
        if isinstance(name, str) and name.startswith("v_"):
            indices = tuple(int(part) for part in name.split("_")[1:])
            if len(indices) == len(shape):
                image[indices] = float(value)
    if not np.isfinite(image).all() or np.any(image < 0.0) or np.any(image > 1.0):
        raise ValueError("hybrid-de source image must be finite and inside [0, 1]")
    return image


def _coefficient_mapping(coefficients: np.ndarray) -> Dict[str, float]:
    return {
        "v_" + "_".join(str(part) for part in index): float(coefficients[index])
        for index in np.ndindex(coefficients.shape)
    }


def _record_hybrid_de_success(
    *,
    save_dir: Optional[str],
    input_name: Optional[str],
    source_image: np.ndarray,
    de_result: Any,
    de_wall_time: float,
    de_cpu_time: float,
    global_real_config: Dict[str, Any],
    extra_meta: Dict[str, Any],
) -> tuple[int, ConcolicTestRecorder]:
    recorder = ConcolicTestRecorder(save_dir, input_name)
    recorder.extra_meta.update(extra_meta)
    recorder.input_shape = tuple(int(dim) for dim in source_image.shape)
    recorder.original_label = int(de_result.original_label)
    recorder.attack_label = int(de_result.best_label)
    recorder.original_input = np.asarray(source_image, dtype=np.float32).copy()
    recorder.adversarial_input = np.asarray(de_result.best_image, dtype=np.float32).copy()
    recorder.global_real_config = global_real_config
    recorder.record_hybrid_de_inputs(source_image)
    recorder.start(
        elapsed_wall_time=de_wall_time,
        elapsed_cpu_time=de_cpu_time,
    )
    recorder.end(completed=True)
    return 0, recorder


def _finalize_hybrid_de_artifacts(
    result: tuple[int, Any],
    *,
    source_image: np.ndarray,
    seed_image: Optional[np.ndarray],
    de_wall_time: float,
    de_cpu_time: float,
) -> tuple[int, Any]:
    recorder = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    if recorder is None or not hasattr(recorder, "record_hybrid_de_inputs"):
        return result
    recorder.record_hybrid_de_inputs(source_image, seed_image)
    recorder.original_input = np.asarray(source_image, dtype=np.float32).copy()
    if isinstance(getattr(recorder, "extra_meta", None), dict):
        recorder.extra_meta["hybrid_total_wall_time_seconds"] = float(
            getattr(recorder, "total_wall_time", 0.0) or 0.0
        ) + float(de_wall_time)
        recorder.extra_meta["hybrid_total_cpu_time_seconds"] = float(
            getattr(recorder, "total_cpu_time", 0.0) or 0.0
        ) + float(de_cpu_time)
    recorder.total_wall_time = float(getattr(recorder, "total_wall_time", 0.0) or 0.0) + float(
        de_wall_time
    )
    recorder.total_cpu_time = float(getattr(recorder, "total_cpu_time", 0.0) or 0.0) + float(
        de_cpu_time
    )
    recorder.save_stats_dict()
    return result


@dataclass
class ExplorerConfig:
    model_path: str
    module: ModuleType
    execute: Callable[..., Any]
    reference_execute: Callable[..., Any]
    reference_score_predictor: Optional[Callable[[np.ndarray], np.ndarray]] = None
    solver: str = DEFAULT_SOLVER
    timeout: int = 900
    constraint_build_timeout: bool = True
    constraint_build_timeout_seconds: int = 30
    solver_run_timeout: Optional[int] = 60
    safety: int = 0
    verbose: int = 1
    logfile: Optional[str] = None
    statsdir: Optional[str] = None
    smtdir: Optional[str] = None
    save_dir: Optional[str] = None
    input_name: Optional[str] = None
    only_first_forward: bool = False
    shap_score_alpha: Optional[float] = None
    symbolic_path_threshold: Optional[int] = None
    ternary_simplification: bool = False
    ternary_threshold_scale: float = 0.75


def _resolve_model_artifacts(model_name: str) -> tuple[str, str, str]:
    model_path = os.path.join(MODEL_ROOT, f"{model_name}.h5")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    module_path = os.path.join(PYCT_ROOT, "engine", "predictor_runtime.py")
    root = os.path.dirname(__file__)
    root = os.path.dirname(root)
    return model_path, module_path, root


def _load_predictor(
    module_path: str,
    root: str,
) -> PredictorCacheEntry:
    cache_key = (os.path.abspath(root), os.path.abspath(module_path))
    cached = _PREDICTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    module = get_module_from_rootdir_and_modpath(root, module_path)
    func_init_reference_model = get_function_from_module_and_funcname(
        module,
        "init_reference_model",
    )
    func_init_model = get_function_from_module_and_funcname(module, "init_model")
    execute_search = get_function_from_module_and_funcname(module, "predict_search")
    execute_reference = get_function_from_module_and_funcname(module, "predict_reference")
    entry: PredictorCacheEntry = (
        module,
        func_init_reference_model,
        func_init_model,
        execute_search,
        execute_reference,
        set(),
    )
    _PREDICTOR_CACHE[cache_key] = entry
    return entry


def _prepare_experiment_paths(
    model_name: str,
    attack_mode: str,
    save_exp: Optional[dict[str, str]],
    only_first_forward: bool,
    timeout: Optional[int],
    constraint_build_timeout: Optional[bool],
    constraint_build_timeout_seconds: Optional[int],
    score_alpha: Optional[float],
    symbolic_path_threshold: Optional[int],
    ternary_simplification: Optional[bool],
    ternary_threshold_scale: Optional[float],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    save_dir = None
    smt_dir = None
    input_name = None

    if save_exp is None:
        return save_dir, smt_dir, input_name

    path_kwargs = {
        "save_exp": save_exp,
        "model_name": model_name,
        "attack_mode": attack_mode,
        "only_first_forward": only_first_forward,
        "timeout": timeout,
        "constraint_build_timeout": constraint_build_timeout,
        "constraint_build_timeout_seconds": constraint_build_timeout_seconds,
        "score_alpha": score_alpha,
        "symbolic_path_threshold": symbolic_path_threshold,
        "ternary_simplification": ternary_simplification,
        "ternary_threshold_scale": ternary_threshold_scale,
    }
    save_dir = get_save_dir_from_save_exp(**path_kwargs)
    input_name = save_exp.get("input_name")
    if save_exp.get("save_smt", False):
        smt_dir = get_save_dir_from_save_exp(**path_kwargs)

    return save_dir, smt_dir, input_name


def _validate_collect_mode(
    collect_mode: str,
) -> Literal["priority_queue", "queue", "stack"]:
    if collect_mode not in VALID_COLLECT_MODES:
        valid = ", ".join(sorted(VALID_COLLECT_MODES))
        raise ValueError(
            f"Unsupported collect_constraints_with='{collect_mode}'. "
            f"Expected one of: {valid}"
        )
    return collect_mode


def _build_explorer(explorer_cfg: ExplorerConfig) -> libct.explore.ExplorationEngine:
    return libct.explore.ExplorationEngine(
        solver=explorer_cfg.solver,
        timeout=explorer_cfg.timeout,
        constraint_build_timeout=explorer_cfg.constraint_build_timeout,
        constraint_build_timeout_seconds=explorer_cfg.constraint_build_timeout_seconds,
        solver_run_timeout=explorer_cfg.solver_run_timeout,
        safety=explorer_cfg.safety,
        store=None,
        verbose=explorer_cfg.verbose,
        logfile=explorer_cfg.logfile,
        statsdir=explorer_cfg.statsdir,
        smtdir=explorer_cfg.smtdir,
        save_dir=explorer_cfg.save_dir,
        input_name=explorer_cfg.input_name,
        module_=explorer_cfg.module,
        execute_=explorer_cfg.execute,
        reference_execute_=explorer_cfg.reference_execute,
        reference_score_predictor_=explorer_cfg.reference_score_predictor,
        only_first_forward=explorer_cfg.only_first_forward,
        shap_score_alpha=explorer_cfg.shap_score_alpha,
        symbolic_path_threshold=explorer_cfg.symbolic_path_threshold,
    )


def _build_initialization_error_result(
    *,
    save_dir: Optional[str],
    input_name: Optional[str],
    in_dict: Dict[str, Any],
    extra_meta: Dict[str, Any],
    error_type: str,
    error_reason: str,
    error_phase: str,
) -> tuple[int, ConcolicTestRecorder]:
    recorder = ConcolicTestRecorder(save_dir, input_name)
    recorder.extra_meta.update(extra_meta)
    recorder.input_shape = get_in_dict_shape(in_dict)
    recorder.start()
    recorder.mark_error(error_type, error_reason, phase=error_phase)
    recorder.save_original_input(in_dict)
    recorder.end(completed=False)
    return 0, recorder


def run(model_name, in_dict, con_dict, norm, solve_order_stack, idx,
        save_exp: dict[str, str] | None = None,
        max_iter=0, single_timeout=900, timeout=900, total_timeout=900, verbose=1,
        constraint_build_timeout: bool = True,
        constraint_build_timeout_seconds: int = 30,
        solver_run_timeout: Optional[int] = 60,
        random_seed: Optional[int] = None,
        limit_change_range=None,
        only_first_forward=False,
        collect_constraints_with='priority_queue',
        input_for_shap=None,
        background_dataset_for_shap=None,
        shap_value_pre_calculated: Optional[bool] = None,
        popped_log_attack_mode=None,
        score_alpha: Optional[float] = None,
        symbolic_path_threshold: Optional[int] = None,
        ternary_simplification: bool = False,
        ternary_threshold_scale: float = 0.75,
        global_real_config=None,
        shap_output_root=None,
) -> tuple[int, Any]:

    collect_mode: Literal["priority_queue", "queue", "stack"] = (
        _validate_collect_mode(collect_constraints_with)
    )
    model_path, module_path, root = _resolve_model_artifacts(model_name)
    search_runtime_key: ModelRuntimeKey = (
        model_path,
        bool(ternary_simplification),
        float(ternary_threshold_scale),
    )
    attack_mode = popped_log_attack_mode or (save_exp.get("attack_mode") if save_exp else "unknown")
    save_dir, smtdir, input_name = _prepare_experiment_paths(
        model_name,
        attack_mode,
        save_exp,
        only_first_forward,
        timeout,
        constraint_build_timeout,
        constraint_build_timeout_seconds,
        score_alpha,
        symbolic_path_threshold,
        ternary_simplification,
        ternary_threshold_scale,
    )

    extra_meta = {
        "model_name": model_name,
        "attack_mode": attack_mode,
        "idx": idx,
        "score_alpha": score_alpha,
        "symbolic_path_threshold": symbolic_path_threshold,
        "ternary_simplification": bool(ternary_simplification),
        "ternary_threshold_scale": float(ternary_threshold_scale),
        "constraint_build_timeout": bool(constraint_build_timeout),
        "constraint_build_timeout_seconds": (
            int(constraint_build_timeout_seconds)
            if constraint_build_timeout_seconds is not None
            else None
        ),
        "label_source": "keras_model_predict",
        "search_model": "NNModel",
        "smt_path_mode": resolve_smt_path_mode(),
    }
    if random_seed is not None:
        extra_meta["random_seed"] = int(random_seed)
    if global_real_config is not None:
        extra_meta.update(
            {
                "global_real_requested_min": global_real_config.get("requested_min"),
                "global_real_requested_max": global_real_config.get("requested_max"),
                "global_real_effective_min": global_real_config.get("effective_min"),
                "global_real_effective_max": global_real_config.get("effective_max"),
                "global_real_bounds_mode": global_real_config.get("bounds_mode"),
                "global_real_transform_mode": global_real_config.get(
                    "transform_mode", "affine"
                ),
                "global_real_pwl_knots": global_real_config.get("pwl_knots"),
                "global_real_pwl_max_segments": global_real_config.get(
                    "pwl_max_segments"
                ),
                "global_real_pwl_segment_count": global_real_config.get(
                    "pwl_segment_count"
                ),
                "global_real_pwl_error_tolerance": global_real_config.get(
                    "pwl_error_tolerance"
                ),
                "global_real_pwl_max_abs_error": global_real_config.get(
                    "pwl_max_abs_error"
                ),
                "global_real_pwl_error_metric": global_real_config.get(
                    "pwl_error_metric"
                ),
                "global_real_pwl_validator_version": global_real_config.get(
                    "pwl_validator_version"
                ),
                "global_real_aces_like_color_space": global_real_config.get(
                    "aces_like_color_space"
                ),
                "global_real_aces_like_curve_version": global_real_config.get(
                    "aces_like_curve_version"
                ),
                "global_real_aces_like_gamut_mapper": global_real_config.get(
                    "aces_like_gamut_mapper"
                ),
                "global_real_shap_sign_epsilon": global_real_config.get(
                    "shap_sign_epsilon"
                ),
                "global_real_shift_kind": global_real_config.get("global_shift_kind"),
                "global_real_contrast_channel_means": global_real_config.get(
                    "contrast_channel_means"
                ),
                "global_real_nonzero_sign_count": global_real_config.get(
                    "nonzero_sign_count"
                ),
                "global_real_shap_target_class": global_real_config.get(
                    "shap_target_class"
                ),
            }
        )
        if global_real_config.get("hybrid_de_enabled"):
            if global_real_config.get("global_shift_kind") in ACES_LIKE_SHIFT_KINDS:
                de_axes = global_real_config.get("hybrid_de_search_axes", global_real_config["global_shift_kind"].split("-", 1)[1])
                auto_axis = de_axes if de_axes != "both" else global_real_config["global_shift_kind"].split("-", 1)[1]
                pyct_axes = global_real_config.get("hybrid_pyct_search_axes", "contrast" if auto_axis == "brightness" else "brightness")
                if de_axes not in ("brightness", "contrast", "both") or pyct_axes not in ("brightness", "contrast", "both"):
                    raise ValueError("ACES hybrid search axes must be brightness, contrast or both")
                extra_meta.update(
                    hybrid_de_search_axes=de_axes, hybrid_pyct_search_axes=pyct_axes,
                    hybrid_de_dimensions=2 if de_axes == "both" else 1,
                    hybrid_pyct_dimensions=2 if pyct_axes == "both" else 1,
                    hybrid_handoff_mode="seed-relative",
                    hybrid_aces_transform_order=global_real_config.get("transform_order", "brightness-contrast"),
                )
            extra_meta.update(
                {
                    "hybrid_de_strategy": (
                        "best1bin-margin-batched-joint2-v1"
                        if global_real_config.get("global_shift_kind") in ACES_LIKE_SHIFT_KINDS and de_axes == "both"
                        else "best1bin-margin-batched-v2"
                    ),
                    "hybrid_reference_transform": (
                        "exact"
                        if global_real_config.get("global_shift_kind") in ACES_LIKE_SHIFT_KINDS
                        else "affine"
                    ),
                    "hybrid_de_maxiter": global_real_config.get(
                        "hybrid_de_maxiter", 75
                    ),
                    "hybrid_de_population_size": global_real_config.get(
                        "hybrid_de_population_size", 400
                    ),
                    "hybrid_de_random_seed": global_real_config.get(
                        "hybrid_de_random_seed"
                    ),
                }
            )
    if save_exp:
        if "ton" in save_exp:
            extra_meta["ton"] = save_exp.get("ton")
        if "ton_next" in save_exp:
            extra_meta["ton_next"] = save_exp.get("ton_next")
        for key in (
            "fallback",
            "fallback_type",
            "fallback_trigger",
            "fallback_source_attack_mode",
            "fallback_source_ton",
            "fallback_source_ton_next",
        ):
            if key in save_exp:
                extra_meta[key] = save_exp.get(key)

    (
        module,
        func_init_reference_model,
        func_init_model,
        execute_search,
        execute_reference,
        initialized_models,
    ) = _load_predictor(module_path, root)
    try:
        func_init_reference_model(model_path)
    except Exception as exc:
        return _build_initialization_error_result(
            save_dir=save_dir,
            input_name=input_name,
            in_dict=in_dict,
            extra_meta=extra_meta,
            error_type="reference_prediction_failure",
            error_reason=str(exc),
            error_phase="reference_model_load",
        )

    hybrid_de_result = None
    hybrid_de_source_image = None
    hybrid_de_wall_time = 0.0
    hybrid_de_cpu_time = 0.0
    if isinstance(global_real_config, dict) and global_real_config.get(
        "hybrid_de_enabled"
    ):
        hybrid_de_started = time.perf_counter()
        hybrid_de_cpu_started = time.process_time()
        try:
            hybrid_de_source_image = _image_from_input_dict(in_dict)
            predict_batch = getattr(module, "predict_reference_batch", None)
            if not callable(predict_batch):
                raise RuntimeError("reference predictor does not support batched inputs")
            source_predictions = np.asarray(
                predict_batch(hybrid_de_source_image[np.newaxis, ...]),
                dtype=np.float64,
            )
            if source_predictions.shape != (1, 10) or not np.isfinite(
                source_predictions
            ).all():
                raise ValueError(
                    "hybrid-de currently requires a finite CIFAR10 class-probability vector"
                )
            original_label = int(np.argmax(source_predictions[0]))
            de_kwargs = {}
            de_shift_kind = global_real_config["global_shift_kind"]
            if de_shift_kind in ACES_LIKE_SHIFT_KINDS:
                if de_axes != "both":
                    de_shift_kind = "aces-" + de_axes
                if "hybrid_de_search_axes" in global_real_config:
                    de_kwargs.update(search_axes=de_axes, transform_order=global_real_config.get("transform_order", "brightness-contrast"))
            hybrid_de_result = run_global_real_differential_evolution(
                hybrid_de_source_image,
                shift_kind=de_shift_kind,
                lower=float(global_real_config["effective_min"]),
                upper=float(global_real_config["effective_max"]),
                original_label=original_label,
                predict_batch=predict_batch,
                random_seed=int(global_real_config.get("hybrid_de_random_seed", 0)),
                maxiter=int(global_real_config.get("hybrid_de_maxiter", 75)),
                population_size=int(
                    global_real_config.get("hybrid_de_population_size", 400)
                ),
                **de_kwargs,
            )
        except Exception as exc:
            error_meta = dict(extra_meta)
            error_meta["hybrid_de_status"] = "error"
            hybrid_de_wall_time = time.perf_counter() - hybrid_de_started
            hybrid_de_cpu_time = time.process_time() - hybrid_de_cpu_started
            error_meta["hybrid_de_wall_time_seconds"] = hybrid_de_wall_time
            error_meta["hybrid_de_cpu_time_seconds"] = hybrid_de_cpu_time
            result = _build_initialization_error_result(
                save_dir=save_dir,
                input_name=input_name,
                in_dict=in_dict,
                extra_meta=error_meta,
                error_type="hybrid_de_failure",
                error_reason=str(exc),
                error_phase="hybrid_de",
            )
            if hybrid_de_source_image is not None:
                return _finalize_hybrid_de_artifacts(
                    result,
                    source_image=hybrid_de_source_image,
                    seed_image=None,
                    de_wall_time=hybrid_de_wall_time,
                    de_cpu_time=hybrid_de_cpu_time,
                )
            return result

        de_wall_time = time.perf_counter() - hybrid_de_started
        de_cpu_time = time.process_time() - hybrid_de_cpu_started
        hybrid_de_wall_time = de_wall_time
        hybrid_de_cpu_time = de_cpu_time
        extra_meta.update(
            {
                "hybrid_de_status": "success" if hybrid_de_result.success else "failed",
                "hybrid_de_shift_kind": "aces-both" if global_real_config["global_shift_kind"] in ACES_LIKE_SHIFT_KINDS and de_axes == "both" else de_shift_kind,
                "hybrid_de_original_label": hybrid_de_result.original_label,
                "hybrid_de_best_label": hybrid_de_result.best_label,
                "hybrid_de_best_x": hybrid_de_result.best_x,
                "hybrid_de_best_score": hybrid_de_result.best_score,
                "hybrid_de_best_margin": hybrid_de_result.best_margin,
                "hybrid_de_iterations": hybrid_de_result.iterations,
                "hybrid_de_function_evaluations": (
                    hybrid_de_result.function_evaluations
                ),
                "hybrid_de_wall_time_seconds": de_wall_time,
                "hybrid_de_cpu_time_seconds": de_cpu_time,
                "hybrid_de_source_artifact": "source_input.npy",
            }
        )
        if global_real_config["global_shift_kind"] in ACES_LIKE_SHIFT_KINDS:
            params = getattr(hybrid_de_result, "best_params", None)
            if params is None:
                params = (hybrid_de_result.best_x, 0.0) if de_axes == "brightness" else (0.0, hybrid_de_result.best_x)
            extra_meta["hybrid_de_best_params"] = list(params)
        if hybrid_de_result.success:
            return _record_hybrid_de_success(
                save_dir=save_dir,
                input_name=input_name,
                source_image=hybrid_de_source_image,
                de_result=hybrid_de_result,
                de_wall_time=de_wall_time,
                de_cpu_time=de_cpu_time,
                global_real_config=global_real_config,
                extra_meta=extra_meta,
            )

        seed_image = hybrid_de_result.best_image
        de_kind = global_real_config["global_shift_kind"]
        extra_meta["hybrid_de_seed_x"] = hybrid_de_result.best_x
        extra_meta["hybrid_de_seed_artifact"] = "de_seed_input.npy"
        if de_kind in ACES_LIKE_SHIFT_KINDS:
            handoff_started = time.perf_counter()
            handoff_cpu_started = time.process_time()
            pyct_kind = de_kind if pyct_axes == "both" else "aces-" + pyct_axes
            extra_meta.update(
                hybrid_pyct_shift_kind="aces-both" if pyct_axes == "both" else pyct_kind,
                hybrid_pyct_seed_x=0.0,
                hybrid_de_seed_params=extra_meta["hybrid_de_best_params"],
                hybrid_pyct_seed_params=[0.0, 0.0],
                hybrid_transform_order=[extra_meta["hybrid_de_shift_kind"], "aces-both" if pyct_axes == "both" else pyct_kind],
            )
            try:
                config = dict(global_real_config)
                config["global_shift_kind"] = pyct_kind
                config["search_axes"] = pyct_axes
                global_real_config = build_aces_like_global_real_config(seed_image, config)
            except Exception as exc:
                handoff_wall_time = time.perf_counter() - handoff_started
                handoff_cpu_time = time.process_time() - handoff_cpu_started
                extra_meta["hybrid_pyct_handoff_wall_time_seconds"] = handoff_wall_time
                extra_meta["hybrid_pyct_handoff_cpu_time_seconds"] = handoff_cpu_time
                result = _build_initialization_error_result(
                    save_dir=save_dir,
                    input_name=input_name,
                    in_dict=in_dict,
                    extra_meta=extra_meta,
                    error_type="hybrid_pyct_pwl_failure",
                    error_reason=str(exc),
                    error_phase="hybrid_pyct_handoff",
                )
                return _finalize_hybrid_de_artifacts(
                    result,
                    source_image=hybrid_de_source_image,
                    seed_image=seed_image,
                    de_wall_time=hybrid_de_wall_time + handoff_wall_time,
                    de_cpu_time=hybrid_de_cpu_time + handoff_cpu_time,
                )
            handoff_wall_time = time.perf_counter() - handoff_started
            handoff_cpu_time = time.process_time() - handoff_cpu_started
            hybrid_de_wall_time += handoff_wall_time
            hybrid_de_cpu_time += handoff_cpu_time
            # The DE axis is fixed. All pixel constants now belong to the
            # selected seed; X=0 on the other axis is its exact identity.
            in_dict = _coefficient_mapping(seed_image)
            if pyct_axes == "both":
                in_dict.update({GLOBAL_BRIGHTNESS_INPUT_NAME: 0.0, GLOBAL_CONTRAST_INPUT_NAME: 0.0})
                con_dict = {GLOBAL_BRIGHTNESS_INPUT_NAME: 1, GLOBAL_CONTRAST_INPUT_NAME: 1}
            else:
                in_dict[GLOBAL_X_INPUT_NAME] = 0.0
                con_dict = {GLOBAL_X_INPUT_NAME: 1}
            input_for_shap = seed_image
            extra_meta.update(
                hybrid_pyct_handoff_wall_time_seconds=handoff_wall_time,
                hybrid_pyct_handoff_cpu_time_seconds=handoff_cpu_time,
                global_real_shift_kind=pyct_kind,
            )
            for key in (
                "transform_mode", "pwl_knots", "pwl_max_segments", "pwl_segment_count",
                "pwl_error_tolerance", "pwl_max_abs_error", "pwl_error_metric", "pwl_validator_version",
            ):
                extra_meta["global_real_" + key] = global_real_config.get(key)
            for key in ("pwl_b_knots", "pwl_c_knots", "pwl_triangle_count", "pwl_max_triangles", "transform_order"):
                extra_meta["global_real_" + key] = global_real_config.get(key)
        else:
            in_dict = dict(in_dict)
            in_dict.pop(global_real_config["variable_name"], None)
            if de_kind == "brightness":
                seed_brightness, seed_contrast = hybrid_de_result.best_x, 0.0
            else:
                seed_brightness, seed_contrast = 0.0, hybrid_de_result.best_x
            in_dict[GLOBAL_BRIGHTNESS_INPUT_NAME] = seed_brightness
            in_dict[GLOBAL_CONTRAST_INPUT_NAME] = seed_contrast
            con_dict = {
                GLOBAL_BRIGHTNESS_INPUT_NAME: 1,
                GLOBAL_CONTRAST_INPUT_NAME: 1,
            }
            coefficients = coefficients_for_shift(hybrid_de_source_image, "contrast")
            global_real_config = dict(global_real_config)
            global_real_config["coefficient_by_input"] = _coefficient_mapping(coefficients)
            global_real_config["transform_mode"] = TRANSFORM_MODE_AFFINE_BC
            spatial_axes = tuple(range(hybrid_de_source_image.ndim - 1))
            channel_means = np.mean(
                hybrid_de_source_image,
                axis=spatial_axes,
                keepdims=True,
                dtype=np.float64,
            )
            global_real_config["contrast_channel_means"] = channel_means.reshape(-1).tolist()
            extra_meta["global_real_contrast_channel_means"] = global_real_config["contrast_channel_means"]
            extra_meta["global_real_transform_mode"] = TRANSFORM_MODE_AFFINE_BC
            extra_meta["hybrid_pyct_seed_brightness"] = seed_brightness
            extra_meta["hybrid_pyct_seed_contrast"] = seed_contrast

    if search_runtime_key not in initialized_models:
        try:
            func_init_model(
                model_path,
                ternary_simplification=ternary_simplification,
                ternary_threshold_scale=ternary_threshold_scale,
                role="search",
            )
        except Exception as exc:
            result = _build_initialization_error_result(
                save_dir=save_dir,
                input_name=input_name,
                in_dict=in_dict,
                extra_meta=extra_meta,
                error_type="search_model_initialization_failure",
                error_reason=str(exc),
                error_phase="search_model_initialization",
            )
            if hybrid_de_result is not None:
                return _finalize_hybrid_de_artifacts(
                    result,
                    source_image=hybrid_de_source_image,
                    seed_image=hybrid_de_result.best_image,
                    de_wall_time=hybrid_de_wall_time,
                    de_cpu_time=hybrid_de_cpu_time,
                )
            return result
        initialized_models.add(search_runtime_key)

    explorer_cfg = ExplorerConfig(
        model_path=model_path,
        module=module,
        execute=execute_search,
        reference_execute=execute_reference,
        reference_score_predictor=(
            getattr(module, "predict_reference_batch")
            if hybrid_de_result is not None
            else None
        ),
        timeout=timeout,
        constraint_build_timeout=constraint_build_timeout,
        constraint_build_timeout_seconds=constraint_build_timeout_seconds,
        solver_run_timeout=solver_run_timeout,
        verbose=verbose,
        smtdir=smtdir,
        save_dir=save_dir,
        input_name=input_name,
        only_first_forward=only_first_forward,
        shap_score_alpha=score_alpha,
        symbolic_path_threshold=symbolic_path_threshold,
        ternary_simplification=ternary_simplification,
        ternary_threshold_scale=ternary_threshold_scale,
    )

    engine = _build_explorer(explorer_cfg)
    engine.extra_meta = extra_meta

    result: tuple[int, Any] = engine.explore(
        module_path,
        in_dict,
        idx=idx,
        concolic_dict=con_dict,
        root=root,
        funcname="predict",
        max_iterations=max_iter,
        single_timeout=single_timeout,
        total_timeout=total_timeout,
        deadcode=set(),
        include_exception=False,
        lib=None,
        file_as_total=False,
        norm=norm,
        solve_order_stack=solve_order_stack,
        limit_change_range=limit_change_range,
        model_path=model_path,
        input_for_shap=input_for_shap,
        background_dataset_for_shap=background_dataset_for_shap,
        shap_value_pre_calculated=(
            bool(shap_value_pre_calculated)
            if shap_value_pre_calculated is not None
            else False
        ),
        collect_constraints_with=collect_mode,
        popped_log_attack_mode=popped_log_attack_mode or "unknown",
        global_real_config=global_real_config,
        shap_output_root=shap_output_root,
    )

    libct.explore.clear_global_context()
    del engine
    gc.collect()

    if hybrid_de_result is not None:
        result = _finalize_hybrid_de_artifacts(
            result,
            source_image=hybrid_de_source_image,
            seed_image=hybrid_de_result.best_image,
            de_wall_time=hybrid_de_wall_time,
            de_cpu_time=hybrid_de_cpu_time,
        )

    return result
