#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Literal, Optional, Set, Tuple

import libct.explore

from libct.utils import (
    get_function_from_module_and_funcname,
    get_module_from_rootdir_and_modpath,
)
from tasks.paths import get_save_dir_from_save_exp

PYCT_ROOT = "./"
MODEL_ROOT = os.path.join(PYCT_ROOT, "model")
VALID_COLLECT_MODES = {"priority_queue", "queue", "stack"}
DEFAULT_SOLVER = "cvc5"
ModelRuntimeKey = Tuple[str, bool, float]
InitializedPredictorKey = Tuple[str, ModelRuntimeKey]
PredictorCacheEntry = Tuple[
    ModuleType,
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
    Set[InitializedPredictorKey],
]
_PREDICTOR_CACHE: Dict[Tuple[str, str], PredictorCacheEntry] = {}


@dataclass
class ExplorerConfig:
    model_path: str
    module: ModuleType
    execute: Callable[..., Any]
    validation_execute: Callable[..., Any]
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
    func_init_model = get_function_from_module_and_funcname(module, "init_model")
    execute_search = get_function_from_module_and_funcname(module, "predict_search")
    execute_validation = get_function_from_module_and_funcname(module, "predict_validation")
    entry: PredictorCacheEntry = (
        module,
        func_init_model,
        execute_search,
        execute_validation,
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


def _validate_collect_mode(collect_mode: str) -> Literal["priority_queue", "queue", "stack"]:
    if collect_mode not in VALID_COLLECT_MODES:
        valid = ", ".join(sorted(VALID_COLLECT_MODES))
        raise ValueError(f"Unsupported collect_constraints_with='{collect_mode}'. Expected one of: {valid}")
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
        validation_execute_=explorer_cfg.validation_execute,
        only_first_forward=explorer_cfg.only_first_forward,
        shap_score_alpha=explorer_cfg.shap_score_alpha,
        symbolic_path_threshold=explorer_cfg.symbolic_path_threshold,
    )


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
        input_for_shap=None, background_dataset_for_shap=None, shap_value_pre_calculated: Optional[bool] = None,
        popped_log_attack_mode=None,
        score_alpha: Optional[float] = None,
        symbolic_path_threshold: Optional[int] = None,
        ternary_simplification: bool = False,
        ternary_threshold_scale: float = 0.75) -> tuple[int, Any]:

    collect_mode: Literal["priority_queue", "queue", "stack"] = _validate_collect_mode(collect_constraints_with)
    model_path, module_path, root = _resolve_model_artifacts(model_name)
    search_runtime_key: ModelRuntimeKey = (
        model_path,
        bool(ternary_simplification),
        float(ternary_threshold_scale),
    )
    validation_runtime_key: ModelRuntimeKey = (
        model_path,
        False,
        0.75,
    )

    module, func_init_model, execute_search, execute_validation, initialized_models = _load_predictor(module_path, root)
    if ("validation", validation_runtime_key) not in initialized_models:
        func_init_model(
            model_path,
            ternary_simplification=False,
            ternary_threshold_scale=0.75,
            role="validation",
        )
        initialized_models.add(("validation", validation_runtime_key))

    if ternary_simplification:
        if ("search", search_runtime_key) not in initialized_models:
            func_init_model(
                model_path,
                ternary_simplification=ternary_simplification,
                ternary_threshold_scale=ternary_threshold_scale,
                role="search",
            )
            initialized_models.add(("search", search_runtime_key))
        execute = execute_search
    else:
        execute = execute_validation

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

    explorer_cfg = ExplorerConfig(
        model_path=model_path,
        module=module,
        execute=execute,
        validation_execute=execute_validation,
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
    }
    if random_seed is not None:
        extra_meta["random_seed"] = int(random_seed)
    if save_exp:
        if "ton" in save_exp:
            extra_meta["ton"] = save_exp.get("ton")
        if "ton_next" in save_exp:
            extra_meta["ton_next"] = save_exp.get("ton_next")
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
        shap_value_pre_calculated=bool(shap_value_pre_calculated) if shap_value_pre_calculated is not None else False,
        collect_constraints_with=collect_mode,
        popped_log_attack_mode=popped_log_attack_mode or "unknown",
    )

    libct.explore.clear_global_context()
    del engine
    gc.collect()

    return result
