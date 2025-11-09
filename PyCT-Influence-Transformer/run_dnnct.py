#!/usr/bin/env python3
from __future__ import annotations

import os
import gc
import libct.explore
from dataclasses import dataclass
from typing import Any, Callable, Optional
from types import ModuleType

from utils.experiment_task_specs import get_save_dir_from_save_exp
from libct.utils import (
    get_module_from_rootdir_and_modpath,
    get_function_from_module_and_funcname,
)

PYCT_ROOT = './'
MODEL_ROOT = os.path.join(PYCT_ROOT, 'model')
VALID_COLLECT_MODES = {"priority_queue", "queue", "stack"}
DEFAULT_SOLVER = "cvc4"


@dataclass
class ExplorerConfig:
    model_path: str
    module: ModuleType
    execute: Callable[..., Any]
    solver: str = DEFAULT_SOLVER
    timeout: int = 900
    safety: int = 0
    verbose: int = 1
    logfile: Optional[str] = None
    statsdir: Optional[str] = None
    smtdir: Optional[str] = None
    save_dir: Optional[str] = None
    input_name: Optional[str] = None
    only_first_forward: bool = False


def _resolve_model_artifacts(model_name: str) -> tuple[str, str, str]:
    model_path = os.path.join(MODEL_ROOT, f"{model_name}.h5")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    module_path = os.path.join(PYCT_ROOT, "dnn_predict_common.py")
    root = os.path.dirname(__file__)
    return model_path, module_path, root


def _load_predictor(module_path: str, root: str) -> tuple[ModuleType, Callable[..., Any]]:
    module = get_module_from_rootdir_and_modpath(root, module_path)
    func_init_model = get_function_from_module_and_funcname(module, "init_model")
    execute = get_function_from_module_and_funcname(module, "predict")
    return module, func_init_model, execute


def _prepare_experiment_paths(
    model_name: str,
    collect_mode: str,
    save_exp: Optional[dict[str, str]],
    only_first_forward: bool,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    save_dir = None
    smt_dir = None
    input_name = None

    if save_exp is None:
        return save_dir, smt_dir, input_name

    path_kwargs = {
        "save_exp": save_exp,
        "model_name": model_name,
        "s_or_q": collect_mode,
        "only_first_forward": only_first_forward,
    }
    save_dir = get_save_dir_from_save_exp(**path_kwargs)
    input_name = save_exp.get("input_name")
    if save_exp.get("save_smt", False):
        smt_dir = get_save_dir_from_save_exp(**path_kwargs)

    return save_dir, smt_dir, input_name


def _validate_collect_mode(collect_mode: str) -> str:
    if collect_mode not in VALID_COLLECT_MODES:
        valid = ", ".join(sorted(VALID_COLLECT_MODES))
        raise ValueError(f"Unsupported collect_constraints_with='{collect_mode}'. Expected one of: {valid}")
    return collect_mode


def _build_explorer(explorer_cfg: ExplorerConfig) -> libct.explore.ExplorationEngine:
    return libct.explore.ExplorationEngine(
        solver=explorer_cfg.solver,
        timeout=explorer_cfg.timeout,
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
        only_first_forward=explorer_cfg.only_first_forward,
    )


def run(model_name, in_dict, con_dict, norm, solve_order_stack, idx,
        save_exp: dict[str, str] | None = None,
        max_iter=0, single_timeout=900, timeout=900, total_timeout=900, verbose=1,
        limit_change_range=None,
        only_first_forward=False,
        collect_constraints_with='priority_queue',
        input_for_shap=None, background_dataset_for_shap=None, shap_value_pre_calculated=None,
        popped_log_attack_mode=None):

    collect_mode = _validate_collect_mode(collect_constraints_with)
    model_path, module_path, root = _resolve_model_artifacts(model_name)

    module, func_init_model, execute = _load_predictor(module_path, root)
    func_init_model(model_path)

    save_dir, smtdir, input_name = _prepare_experiment_paths(
        model_name,
        collect_mode,
        save_exp,
        only_first_forward,
    )

    explorer_cfg = ExplorerConfig(
        model_path=model_path,
        module=module,
        execute=execute,
        timeout=timeout,
        verbose=verbose,
        smtdir=smtdir,
        save_dir=save_dir,
        input_name=input_name,
        only_first_forward=only_first_forward,
    )

    engine = _build_explorer(explorer_cfg)

    result = engine.explore(
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
        shap_value_pre_calculated=shap_value_pre_calculated,
        collect_constraints_with=collect_mode,
        popped_log_attack_mode=popped_log_attack_mode or "unknown",
    )

    libct.explore.clear_global_context()
    del engine
    gc.collect()

    return result

