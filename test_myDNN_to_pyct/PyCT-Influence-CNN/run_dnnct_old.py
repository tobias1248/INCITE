#!/usr/bin/env python3

import os
import libct.explore
import libct.tnn_explore
import json
from libct.utils import get_module_from_rootdir_and_modpath, get_function_from_module_and_funcname
from utils_out.pyct_attack_exp import get_save_dir_from_save_exp  # <-- 修正成 utils_out

PYCT_ROOT = os.path.dirname(__file__)  # <-- 改成當前 repo 根目錄更穩
MODEL_ROOT = os.path.join(PYCT_ROOT, 'model')


def get_matrix_shape(model_name):
    if "mnist" in model_name.lower():
        return (28, 28, 1)
    elif "imdb" in model_name.lower():
        return (500, 32, 1)
    else:
        return None


def calculate_matrix_shape(in_dict):
    keys = list(in_dict.keys())
    if not keys:
        return None
    dimensions = [int(dim) for dim in keys[0].split('_')[1:]]
    max_dims = [max(int(key.split('_')[i+1])
                    for key in keys) + 1 for i in range(len(dimensions))]
    return tuple(max_dims)


def run(model_name, in_dict, con_dict, norm, solve_order_stack,
        model_type="cnn", save_exp=None,
        max_iter=0, single_timeout=900, timeout=900, total_timeout=900, verbose=1,
        limit_change_percentage=None, only_first_forward=False, delta_factor=0.75):

    model_path = os.path.join(MODEL_ROOT, f"{model_name}.h5")
    modpath = os.path.join(PYCT_ROOT, "dnn_predict_common.py")
    func = "predict"
    funcname = func
    save_dir = None
    smtdir = None
    matrix_shape = get_matrix_shape(model_name)

    dump_projstats = False
    file_as_total = False
    formula = None
    include_exception = False
    lib = None
    logfile = None
    root = os.path.dirname(__file__)
    safety = 0

    if matrix_shape is None:
        matrix_shape = calculate_matrix_shape(in_dict)

    statsdir = None
    if dump_projstats:
        statsdir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "project_statistics",
            os.path.abspath(root).split('/')[-1], modpath, funcname
        )

    module = get_module_from_rootdir_and_modpath(root, modpath)
    func_init_model = get_function_from_module_and_funcname(
        module, "init_model")
    execute = get_function_from_module_and_funcname(module, funcname)
    # func_init_model(model_path, model_type=model_type,
    #                 delta_factor=delta_factor)

    func_init_model(model_path)

    if save_exp is not None:
        s_or_q = "stack" if solve_order_stack else "queue"
        save_dir = get_save_dir_from_save_exp(
            save_exp, model_name, s_or_q, only_first_forward=only_first_forward)

        if save_exp.get('save_smt', False):
            smtdir = save_dir

    if model_type == "origin":
        engine = libct.explore.ExplorationEngine(
            solver='cvc4', timeout=timeout, safety=safety,
            store=formula, verbose=verbose, logfile=logfile,
            statsdir=statsdir, smtdir=smtdir,
            save_dir=save_dir, input_name=save_exp['input_name'],
            module_=module, execute_=execute,
            only_first_forward=only_first_forward
        )
    elif model_type == "qnn":
        engine = libct.tnn_explore.ExplorationEngine(
            solver='cvc4', timeout=timeout, safety=safety,
            store=formula, verbose=verbose, logfile=logfile,
            statsdir=statsdir, smtdir=smtdir,
            save_dir=save_dir, input_name=save_exp['input_name'],
            module_=module, execute_=execute,
            only_first_forward=only_first_forward,
            model_name=model_name, matrix_shape=matrix_shape
        )

    result = engine.explore(
        modpath, in_dict, concolic_dict=con_dict, root=root, funcname=func, max_iterations=max_iter,
        single_timeout=single_timeout, total_timeout=total_timeout, deadcode=set(),
        include_exception=include_exception, lib=lib,
        file_as_total=file_as_total, norm=norm, solve_order_stack=solve_order_stack,
        limit_change_range=limit_change_percentage,
    )

    return result
