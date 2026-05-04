from __future__ import annotations

import json
import logging
import os
import queue as py_queue
import signal
import time
import traceback
from multiprocessing import Event, JoinableQueue, Process
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestration.progress import (
    collect_stage_cases,
    derive_stage_outcome_payload,
    extract_last_ton,
    load_stats_payload,
    should_run_payload,
    should_run_ton,
    update_ton_progress_stats,
)
from orchestration.runners import QueueRunner, RandomAssignRunner, ShapRunner
from tasks.builders.cifar10 import cifar10_transformer_random, cifar10_transformer_shap
from tasks.builders.fashion_mnist import (
    fashion_mnist_transformer_random,
    fashion_mnist_transformer_shap,
)
from tasks.builders.mnist import mnist_transformer_random, mnist_transformer_shap
from tasks.paths import get_save_dir_from_save_exp

logger = logging.getLogger("ct.cli")


def _resolve_task_save_dir(task: Dict[str, Any], attack_mode: str) -> Optional[Path]:
    save_exp = task.get("save_exp") or {}
    model_name = task.get("model_name")
    if not save_exp or not model_name:
        return None
    resolved_attack_mode = save_exp.get(
        "attack_mode",
        task.get("popped_log_attack_mode", attack_mode),
    )
    return Path(
        get_save_dir_from_save_exp(
            save_exp,
            model_name,
            resolved_attack_mode,
            only_first_forward=bool(save_exp.get("only_first_forward", False)),
        )
    )


def _write_worker_failure_stats(task: Dict[str, Any], attack_mode: str, reason: str) -> None:
    if not isinstance(task, dict):
        return
    save_dir = _resolve_task_save_dir(task, attack_mode)
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    stats_path = save_dir / "stats.json"
    if stats_path.is_file():
        return
    save_exp = task.get("save_exp") or {}
    payload = {
        "meta": {
            "input_name": save_exp.get("input_name"),
            "attack_label": None,
            "is_finish": False,
            "is_timeout": False,
            "solve_all_ctr": False,
            "status": "error",
            "error_type": "worker_execution_failure",
            "error_phase": "launcher",
            "error_reason": reason,
            "ton": save_exp.get("ton"),
            "ton_next": save_exp.get("ton_next"),
        },
        "summary": {},
        "solver": {},
        "constraints": {},
        "constraint_complexity": None,
        "iters_summary": {},
    }
    stats_path.write_text(json.dumps(payload), encoding="utf-8")
    (save_dir / "worker_execution_failure_traceback.txt").write_text(
        reason,
        encoding="utf-8",
    )


def _resolve_experiment_layout(attack_mode: str, ton_values) -> str:
    if not ton_values:
        raise ValueError("ton_values must be non-empty.")
    if attack_mode not in ("shap", "random", "random-assign", "queue"):
        raise ValueError(f"Unsupported attack mode: {attack_mode}")
    return attack_mode


def _worker(
    task_queue: JoinableQueue,
    timeout: int,
    constraint_build_timeout: bool,
    constraint_build_timeout_seconds: int,
    solver_run_timeout,
    error_retry_limit: int,
    norm_01: bool,
    attack_mode: str,
    pixel_source: str,
    base_seed: int,
    shutdown_event: Event,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    worker_pid = os.getpid()
    runner = None
    try:
        if shutdown_event.is_set():
            logger.info("[WORKER-SHUTDOWN] pid=%s aborting before start", worker_pid)
            return
        if attack_mode == "random-assign":
            runner = RandomAssignRunner(
                timeout=timeout,
                constraint_build_timeout=constraint_build_timeout,
                constraint_build_timeout_seconds=constraint_build_timeout_seconds,
                solver_run_timeout=solver_run_timeout,
                norm=norm_01,
                pixel_source=pixel_source,
                base_seed=base_seed,
            )
        elif attack_mode == "queue":
            # Retry lives in the runner because the runner owns re-executing
            # one TON payload; progress only classifies the returned outcome.
            runner = QueueRunner(
                timeout=timeout,
                constraint_build_timeout=constraint_build_timeout,
                constraint_build_timeout_seconds=constraint_build_timeout_seconds,
                solver_run_timeout=solver_run_timeout,
                norm=norm_01,
                collect_constraints_with="queue",
                error_retry_limit=error_retry_limit,
            )
        else:
            # Retry lives in the runner because the runner owns re-executing
            # one TON payload; progress only classifies the returned outcome.
            runner = ShapRunner(
                timeout=timeout,
                constraint_build_timeout=constraint_build_timeout,
                constraint_build_timeout_seconds=constraint_build_timeout_seconds,
                solver_run_timeout=solver_run_timeout,
                norm=norm_01,
                error_retry_limit=error_retry_limit,
            )

        while not shutdown_event.is_set():
            try:
                task = task_queue.get(timeout=1)
            except py_queue.Empty:
                continue
            if task is None:
                task_queue.task_done()
                break
            try:
                task_snapshot = dict(task) if isinstance(task, dict) else task
                runner.run_tasks([task])
            except Exception as exc:
                save_dir = (
                    _resolve_task_save_dir(task_snapshot, attack_mode)
                    if isinstance(task_snapshot, dict)
                    else None
                )
                input_name = (
                    (task_snapshot.get("save_exp") or {}).get("input_name")
                    if isinstance(task_snapshot, dict)
                    else None
                )
                reason = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
                logger.exception(
                    "[WORKER-TASK-ERROR] pid=%s idx=%s attack=%s input_name=%s save_dir=%s",
                    worker_pid,
                    task_snapshot.get("idx") if isinstance(task_snapshot, dict) else "unknown",
                    attack_mode,
                    input_name,
                    save_dir,
                )
                if isinstance(task_snapshot, dict):
                    _write_worker_failure_stats(task_snapshot, attack_mode, reason)
            finally:
                task_queue.task_done()
    except KeyboardInterrupt:
        logger.info("[WORKER-INTERRUPT] pid=%s received interrupt", worker_pid)
    finally:
        logger.info("[WORKER-EXIT] pid=%s", worker_pid)


def run_launcher(args: Any) -> None:
    interrupted = False
    shutdown_event = Event()
    running_processes: List[Process] = []

    def _handle_signal(signum, _frame):
        try:
            signame = signal.Signals(signum).name
        except ValueError:
            signame = str(signum)
        logger.warning("Received signal %s; initiating shutdown", signame)
        shutdown_event.set()
        for proc in running_processes:
            if proc.is_alive():
                try:
                    os.kill(proc.pid, signum)
                except ProcessLookupError:
                    continue
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except ValueError:
            pass

    if args.num_process < 1:
        raise ValueError("--num-process must be >= 1")
    if args.first_n < 1:
        raise ValueError("--first-n must be >= 1")
    if args.timeout < 1:
        raise ValueError("--timeout must be >= 1 second")
    if args.constraint_build_timeout_seconds < 1:
        raise ValueError("--constraint-build-timeout-seconds must be >= 1")
    if args.error_retry_limit < 0:
        raise ValueError("--error-retry-limit must be >= 0")
    if args.spawn_delay < 0:
        raise ValueError("--spawn-delay must be non-negative")

    attack_mode = _resolve_experiment_layout(args.attack_mode, args.pixel_search)
    os.environ["PYCT_TIMEOUT"] = str(args.timeout)
    os.environ["PYCT_CONSTRAINT_BUILD_TIMEOUT_ENABLED"] = (
        "1" if args.constraint_build_timeout else "0"
    )
    os.environ["PYCT_CONSTRAINT_BUILD_TIMEOUT_SECONDS"] = str(
        args.constraint_build_timeout_seconds
    )
    if args.score_alpha is not None:
        os.environ["PYCT_SCORE_ALPHA"] = str(args.score_alpha)
    else:
        os.environ.pop("PYCT_SCORE_ALPHA", None)
    os.environ["PYCT_SYMBOLIC_PATH_THRESHOLD"] = str(args.symbolic_path_threshold)
    os.environ["PYCT_ENABLE_CONSTRAINT_LOG"] = "1" if args.enable_constraint_log else "0"
    os.environ["PYCT_TERNARY_SIMPLIFICATION"] = "1" if args.ternary_simplification else "0"
    os.environ["PYCT_TERNARY_THRESHOLD_SCALE"] = str(args.ternary_threshold_scale)

    attack_mode_parts = [attack_mode]
    if args.attack_mode == "shap" and args.pixel_selector == "patch-shap":
        attack_mode_parts.append("patchshap")
    if args.attack_mode == "shap" and args.pixel_selector == "token-shap":
        attack_mode_parts.append("tokenshap")
    if args.solver_run_timeout and args.solver_run_timeout > 0:
        attack_mode_parts.append(f"solver{args.solver_run_timeout}s")
    attack_mode_for_paths = "_".join(attack_mode_parts)
    force_refresh = args.force_refresh
    first_n_range = range(0, args.first_n)

    if args.dataset == "cifar10":
        shap_fn = cifar10_transformer_shap
        random_fn = cifar10_transformer_random
    elif args.dataset == "mnist":
        shap_fn = mnist_transformer_shap
        random_fn = mnist_transformer_random
    else:
        shap_fn = fashion_mnist_transformer_shap
        random_fn = fashion_mnist_transformer_random

    shap_kwargs = {}
    if args.dataset == "cifar10":
        shap_kwargs["pixel_selector"] = args.pixel_selector

    if args.attack_mode == "shap":
        inputs = shap_fn(
            args.model_name,
            first_n_img=first_n_range,
            force=True,
            ton_values=args.pixel_search,
            attack_mode=attack_mode_for_paths,
            **shap_kwargs,
        )
    elif args.attack_mode == "random":
        inputs = random_fn(
            args.model_name,
            first_n_img=first_n_range,
            ton_values=args.pixel_search,
            force=True,
            base_seed=args.random_seed,
            attack_mode=attack_mode_for_paths,
        )
    elif args.attack_mode == "random-assign":
        exp_prefix = f"random_assign_{args.pixel_source}"
        if args.pixel_source == "random":
            inputs = random_fn(
                args.model_name,
                first_n_img=first_n_range,
                ton_values=args.pixel_search,
                force=True,
                base_seed=args.random_seed,
                exp_prefix=exp_prefix,
                attack_mode=attack_mode_for_paths,
            )
        else:
            inputs = shap_fn(
                args.model_name,
                first_n_img=first_n_range,
                force=True,
                ton_values=args.pixel_search,
                exp_prefix=exp_prefix,
                attack_mode=attack_mode_for_paths,
                **shap_kwargs,
            )
    elif args.attack_mode == "queue":
        inputs = shap_fn(
            args.model_name,
            first_n_img=first_n_range,
            force=True,
            ton_values=args.pixel_search,
            exp_prefix="queue",
            attack_mode=attack_mode_for_paths,
            **shap_kwargs,
        )
    else:
        raise ValueError(f"Unsupported attack mode: {args.attack_mode}")

    for payload in inputs:
        payload["score_alpha"] = args.score_alpha
        payload["symbolic_path_threshold"] = args.symbolic_path_threshold
        payload["ternary_simplification"] = args.ternary_simplification
        if args.ternary_simplification:
            payload["ternary_threshold_scale"] = args.ternary_threshold_scale
        else:
            payload.pop("ternary_threshold_scale", None)

    logger.info(
        "Prepared %s input(s) for attack=%s ton_sequence=%s",
        len(inputs),
        attack_mode_for_paths,
        ",".join(str(v) for v in args.pixel_search),
    )
    time.sleep(1)

    worker_count = max(1, args.num_process)
    task_queue: JoinableQueue = JoinableQueue()

    def _start_workers() -> None:
        if running_processes:
            return
        for _ in range(worker_count):
            if shutdown_event.is_set():
                logger.info("Shutdown requested; stop worker bootstrap")
                break
            process = Process(
                target=_worker,
                args=(
                    task_queue,
                    args.timeout,
                    args.constraint_build_timeout,
                    args.constraint_build_timeout_seconds,
                    args.solver_run_timeout if args.solver_run_timeout > 0 else None,
                    args.error_retry_limit,
                    args.norm_01,
                    args.attack_mode,
                    args.pixel_source,
                    args.random_seed,
                    shutdown_event,
                ),
            )
            process.start()
            running_processes.append(process)
            time.sleep(args.spawn_delay)

    def _stop_workers() -> None:
        if not running_processes:
            return
        for _ in running_processes:
            task_queue.put(None)
        task_queue.join()
        for process in running_processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
            process.join()
        running_processes.clear()

    def _run_stage_tasks(stage_inputs: List[Dict[str, Any]]) -> None:
        if not stage_inputs or shutdown_event.is_set():
            return
        _start_workers()
        for task in stage_inputs:
            task_queue.put(task)
        task_queue.join()

    cases = collect_stage_cases(inputs)
    if not cases:
        if not force_refresh:
            filtered_inputs = [payload for payload in inputs if should_run_payload(payload, force_refresh=force_refresh)]
            inputs = filtered_inputs
        try:
            _run_stage_tasks(inputs)
        except KeyboardInterrupt:
            interrupted = True
            shutdown_event.set()
    else:
        ton_sequence = list(args.pixel_search)
        try:
            for ton_index, ton_value in enumerate(ton_sequence):
                if shutdown_event.is_set():
                    break
                next_ton = ton_sequence[ton_index + 1] if ton_index + 1 < len(ton_sequence) else None
                stage_tasks: List[Dict[str, Any]] = []
                for case in cases:
                    plan = case["plans"].get(ton_value)
                    if not plan:
                        continue
                    if not should_run_ton(case, ton_value, ton_sequence, force_refresh=force_refresh):
                        continue
                    payload = dict(case["base_payload"])
                    payload["con_dict"] = plan["con_dict"]
                    save_exp = dict(plan["save_exp"])
                    save_exp["ton"] = ton_value
                    save_exp["ton_next"] = next_ton
                    payload["save_exp"] = save_exp
                    stage_tasks.append(payload)

                _run_stage_tasks(stage_tasks)
                if shutdown_event.is_set():
                    break

                next_candidates = 0
                for case in cases:
                    stats_path = Path(case["save_dir"]) / "stats.json"
                    stats, reason = load_stats_payload(stats_path)
                    if not stats:
                        continue
                    last_ton = extract_last_ton(stats)
                    if last_ton != ton_value:
                        continue
                    should_continue, reason = derive_stage_outcome_payload(stats)
                    status = "continue" if should_continue else "stop"
                    update_ton_progress_stats(
                        stats_path,
                        current_ton=ton_value,
                        status=status,
                        reason=reason,
                        next_ton=next_ton,
                    )
                if next_ton is not None:
                    for case in cases:
                        if should_run_ton(case, next_ton, ton_sequence, force_refresh=force_refresh):
                            next_candidates += 1
                if next_ton is None or next_candidates == 0:
                    break
        except KeyboardInterrupt:
            interrupted = True
            shutdown_event.set()

    _stop_workers()

    if interrupted or shutdown_event.is_set():
        logger.info("Tasks interrupted; shutdown requested")
    else:
        logger.info("All tasks completed")


__all__ = ["run_launcher"]
