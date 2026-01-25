from __future__ import annotations

import json
import logging
import os
import signal
import time
from multiprocessing import Event, Process, Queue
import queue as py_queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from start_config import _INPUT_PREFIX
from utils.experiment_runner import update_ton_progress_stats
from utils.experiment_task_specs import (
    fashion_mnist_transformer_random,
    fashion_mnist_transformer_shap,
    get_save_dir_from_save_exp,
)

logger = logging.getLogger("ct.cli")


def _resolve_experiment_layout(
    attack_mode: str,
    ton_values: Sequence[int],
    *,
    pixel_source: str = "random",
) -> str:
    if not ton_values:
        raise ValueError("ton_values must be non-empty.")
    if attack_mode not in ("shap", "random", "random-assign", "queue"):
        raise ValueError(f"Unsupported attack mode: {attack_mode}")
    return attack_mode


def _stats_indicate_completion(payload: Dict[str, Any]) -> bool:
    """Return True when stats.json shows a completed attack run."""
    meta = payload.get("meta") or {}
    attack_label = payload.get("attack_label", meta.get("attack_label"))
    is_finished = bool(meta.get("is_finish"))
    is_timeout = bool(meta.get("is_timeout"))
    return bool(attack_label is not None or is_finished or is_timeout)


def _derive_resume_plan(
    model_name: str,
    attack_mode: str,
    first_n: int,
) -> Tuple[int, bool]:
    base_dir = Path("exp") / model_name / attack_mode
    if not base_dir.is_dir():
        return 0, False

    latest_idx: Optional[int] = None
    for candidate in base_dir.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith(_INPUT_PREFIX):
            continue
        try:
            idx = int(candidate.name.split("_")[-1])
        except ValueError:
            continue
        if idx >= first_n:
            continue
        if latest_idx is None or idx > latest_idx:
            latest_idx = idx

    if latest_idx is None:
        return 0, False

    candidate_dir = base_dir / f"{_INPUT_PREFIX}{latest_idx}"
    stats_path = candidate_dir / "stats.json"
    completed = False
    if stats_path.is_file():
        try:
            with stats_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            completed = _stats_indicate_completion(payload)
        except (json.JSONDecodeError, OSError):
            completed = False

    if not completed:
        return latest_idx, True

    resume_idx = latest_idx + 1
    return min(resume_idx, first_n), False


def _derive_stage_outcome(stats_path: Path) -> Tuple[bool, str]:
    stats, reason = _load_stats_payload(stats_path)
    if not stats:
        return False, reason
    return _derive_stage_outcome_payload(stats)


def _load_stats_payload(stats_path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    if not stats_path.is_file():
        return None, "missing_stats"
    try:
        with stats_path.open("r", encoding="utf-8") as handle:
            return json.load(handle), "ok"
    except (OSError, json.JSONDecodeError):
        return None, "invalid_stats"


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _extract_last_ton(stats: Dict[str, Any]) -> Optional[int]:
    meta = stats.get("meta") or {}
    ton = _coerce_int(meta.get("ton"))
    if ton is not None:
        return ton
    progress = meta.get("progress") or stats.get("progress") or {}
    ton = _coerce_int(progress.get("ton_current"))
    if ton is not None:
        return ton
    ton_progress = meta.get("ton_progress") or stats.get("ton_progress") or {}
    ton = _coerce_int(ton_progress.get("current"))
    if ton is not None:
        return ton
    return None


def _derive_stage_outcome_payload(stats: Dict[str, Any]) -> Tuple[bool, str]:
    meta = stats.get("meta") or {}
    attack_label = meta.get("attack_label")
    solved_all = bool(meta.get("solve_all_ctr"))
    is_timeout = bool(meta.get("is_timeout"))
    if attack_label is not None:
        return False, "adv_found"
    if solved_all:
        return True, "solve_all_ctr"
    if is_timeout:
        return True, "timeout"
    return False, "incomplete"


def _should_run_ton(
    case: Dict[str, Any],
    ton_value: int,
    ton_sequence: Sequence[int],
    *,
    force_refresh: bool,
) -> bool:
    if force_refresh:
        return True
    stats_path = Path(case["save_dir"]) / "stats.json"
    stats, _ = _load_stats_payload(stats_path)
    if not stats:
        return ton_value == ton_sequence[0]
    meta = stats.get("meta") or {}
    if meta.get("attack_label") is not None:
        return False
    last_ton = _extract_last_ton(stats)
    if last_ton is None:
        return ton_value == ton_sequence[0]
    if last_ton > ton_value:
        return False
    should_continue, reason = _derive_stage_outcome_payload(stats)
    if last_ton == ton_value:
        return reason == "incomplete"
    try:
        idx = list(ton_sequence).index(last_ton)
    except ValueError:
        return ton_value == ton_sequence[0]
    if idx + 1 >= len(ton_sequence) or ton_sequence[idx + 1] != ton_value:
        return False
    return should_continue


def _should_run_payload(payload: Dict[str, Any], *, force_refresh: bool) -> bool:
    if force_refresh:
        return True
    save_exp = payload.get("save_exp") or {}
    attack_mode = save_exp.get("attack_mode", payload.get("popped_log_attack_mode", "unknown"))
    save_dir = get_save_dir_from_save_exp(
        save_exp,
        payload.get("model_name", "unknown"),
        attack_mode,
        only_first_forward=bool(save_exp.get("only_first_forward", False)),
    )
    stats_path = Path(save_dir) / "stats.json"
    stats, _ = _load_stats_payload(stats_path)
    if not stats:
        return True
    return not _stats_indicate_completion(stats)


def _collect_stage_cases(inputs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for payload in inputs:
        ton_plans = payload.get("ton_plans") or []
        if not ton_plans:
            continue
        base_payload = dict(payload)
        base_payload.pop("ton_plans", None)
        plan_by_ton = {
            plan.get("ton"): plan for plan in ton_plans if plan.get("ton") is not None
        }
        first_plan = ton_plans[0]
        save_exp = first_plan.get("save_exp", {})
        attack_mode = save_exp.get("attack_mode", base_payload.get("popped_log_attack_mode", "unknown"))
        save_dir = get_save_dir_from_save_exp(
            save_exp,
            base_payload["model_name"],
            attack_mode,
            only_first_forward=bool(save_exp.get("only_first_forward", False)),
        )
        cases.append(
            {
                "idx": base_payload.get("idx"),
                "input_name": save_exp.get("input_name"),
                "base_payload": base_payload,
                "plans": plan_by_ton,
                "save_dir": save_dir,
            }
        )
    return cases


def _worker(
    task_queue: Queue,
    timeout: int,
    constraint_build_timeout: bool,
    solver_run_timeout: Optional[int],
    norm_01: bool,
    attack_mode: str,
    pixel_source: str,
    base_seed: int,
    shutdown_event: Event,
) -> None:
    """Entry point for each subprocess that forwards to the util helper."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    worker_pid = os.getpid()
    try:
        if shutdown_event.is_set():
            logger.info("[WORKER-SHUTDOWN] pid=%s aborting before start", worker_pid)
            return
        while not shutdown_event.is_set():
            try:
                task = task_queue.get(timeout=1)
            except py_queue.Empty:
                break
            if task is None:
                break

            if attack_mode == "random-assign":
                from utils.experiment_runner import run_attack_with_random_assign

                run_attack_with_random_assign(
                    [task],
                    timeout=timeout,
                    constraint_build_timeout=constraint_build_timeout,
                    solver_run_timeout=solver_run_timeout,
                    norm=norm_01,
                    pixel_source=pixel_source,
                    base_seed=base_seed,
                )
            else:
                from utils.experiment_runner import run_attack_with_shap

                run_attack_with_shap(
                    [task],
                    timeout=timeout,
                    constraint_build_timeout=constraint_build_timeout,
                    solver_run_timeout=solver_run_timeout,
                    norm=norm_01,
                )
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
    if args.spawn_delay < 0:
        raise ValueError("--spawn-delay must be non-negative")

    attack_mode = _resolve_experiment_layout(
        args.attack_mode,
        args.pixel_search,
        pixel_source=args.pixel_source,
    )
    attack_mode_for_paths = (
        attack_mode
        if not args.solver_run_timeout or args.solver_run_timeout <= 0
        else f"{attack_mode}_solver{args.solver_run_timeout}s"
    )
    force_refresh = args.force_refresh
    force_generate = True
    first_n_range = range(0, args.first_n)
    if force_refresh:
        logger.info("Force refresh enabled; scheduling inputs from idx=0 to %s", args.first_n - 1)
    else:
        logger.info("Full scan enabled; scheduling inputs from idx=0 to %s", args.first_n - 1)

    def _select_shap_fn():
        if args.dataset == "cifar10":
            from utils.experiment_task_specs import cifar10_transformer_shap
            return cifar10_transformer_shap
        if args.dataset == "mnist":
            from utils.experiment_task_specs import mnist_transformer_shap
            return mnist_transformer_shap
        return fashion_mnist_transformer_shap

    def _select_random_fn():
        if args.dataset == "cifar10":
            from utils.experiment_task_specs import cifar10_transformer_random
            return cifar10_transformer_random
        if args.dataset == "mnist":
            from utils.experiment_task_specs import mnist_transformer_random
            return mnist_transformer_random
        return fashion_mnist_transformer_random

    shap_fn = _select_shap_fn()
    random_fn = _select_random_fn()

    if args.attack_mode == "shap":
        inputs = shap_fn(
            args.model_name,
            first_n_img=first_n_range,
            force=force_generate,
            ton_values=args.pixel_search,
            attack_mode=attack_mode_for_paths,
        )
    elif args.attack_mode == "random":
        inputs = random_fn(
            args.model_name,
            first_n_img=first_n_range,
            ton_values=args.pixel_search,
            force=force_generate,
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
                force=force_generate,
                base_seed=args.random_seed,
                exp_prefix=exp_prefix,
                attack_mode=attack_mode_for_paths,
            )
        else:
            inputs = shap_fn(
                args.model_name,
                first_n_img=first_n_range,
                force=force_generate,
                ton_values=args.pixel_search,
                exp_prefix=exp_prefix,
                attack_mode=attack_mode_for_paths,
            )
    elif args.attack_mode == "queue":
        # For queue mode, reuse SHAP task generation and execute via QueueRunner inline.
        inputs = shap_fn(
            args.model_name,
            first_n_img=first_n_range,
            force=force_generate,
            ton_values=args.pixel_search,
            exp_prefix="queue",
            attack_mode=attack_mode_for_paths,
        )
    else:
        raise ValueError(f"Unsupported attack mode: {args.attack_mode}")

    for payload in inputs:
        payload["score_alpha"] = args.score_alpha

    logger.info(
        "Prepared %s input(s) for attack=%s ton_sequence=%s",
        len(inputs),
        attack_mode_for_paths,
        ",".join(str(v) for v in args.pixel_search),
    )
    time.sleep(3)

    def _run_stage_tasks(stage_inputs: List[Dict[str, Any]]) -> None:
        if not stage_inputs or shutdown_event.is_set():
            return
        running_processes.clear()
        if args.attack_mode == "queue":
            from utils.experiment_runner import run_attack_with_queue

            run_attack_with_queue(
                stage_inputs,
                timeout=args.timeout,
                constraint_build_timeout=args.constraint_build_timeout,
                solver_run_timeout=args.solver_run_timeout if args.solver_run_timeout > 0 else None,
                norm=args.norm_01,
                collect_constraints_with="queue",
            )
            return

        task_queue: Queue = Queue()
        for task in stage_inputs:
            task_queue.put(task)
        for _ in range(args.num_process):
            task_queue.put(None)

        try:
            worker_count = min(args.num_process, max(len(stage_inputs), 1))
            for _ in range(worker_count):
                if shutdown_event.is_set():
                    logger.info("Shutdown requested; skipping remaining tasks")
                    break
                process = Process(
                    target=_worker,
                    args=(
                        task_queue,
                        args.timeout,
                        args.constraint_build_timeout,
                        args.solver_run_timeout if args.solver_run_timeout > 0 else None,
                        args.norm_01,
                        args.attack_mode,
                        args.pixel_source,
                        args.random_seed,
                        shutdown_event,
                    ),
                )
                logger.info(
                    "[WORKER-START] timeout=%s norm=%s attack=%s pixel_src=%s pid_pending",
                    args.timeout,
                    args.norm_01,
                    args.attack_mode,
                    args.pixel_source,
                )
                process.start()
                running_processes.append(process)
                time.sleep(args.spawn_delay)

            for process in running_processes:
                while process.is_alive():
                    process.join(timeout=0.5)
                    if shutdown_event.is_set():
                        break
                logger.info("[WORKER-DONE] pid=%s exitcode=%s", process.pid, process.exitcode)
        finally:
            for process in running_processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            running_processes.clear()

    cases = _collect_stage_cases(inputs)
    if not cases:
        if not force_refresh:
            filtered_inputs = [payload for payload in inputs if _should_run_payload(payload, force_refresh=force_refresh)]
            skipped = len(inputs) - len(filtered_inputs)
            if skipped:
                logger.info("Skipping %s already-completed task(s) after scan", skipped)
            inputs = filtered_inputs
        logger.info("No ton_plans found; running %s task(s) directly", len(inputs))
        try:
            _run_stage_tasks(inputs)
        except KeyboardInterrupt:
            logger.warning("Main loop interrupted; shutting down workers")
            interrupted = True
            shutdown_event.set()
    else:
        ton_sequence = list(args.pixel_search)
        try:
            for ton_index, ton_value in enumerate(ton_sequence):
                if shutdown_event.is_set():
                    logger.info("Shutdown requested; skipping remaining stages")
                    break
                next_ton = ton_sequence[ton_index + 1] if ton_index + 1 < len(ton_sequence) else None
                stage_tasks: List[Dict[str, Any]] = []
                for case in cases:
                    plan = case["plans"].get(ton_value)
                    if not plan:
                        continue
                    if not _should_run_ton(
                        case,
                        ton_value,
                        ton_sequence,
                        force_refresh=force_refresh,
                    ):
                        continue
                    payload = dict(case["base_payload"])
                    payload["con_dict"] = plan["con_dict"]
                    save_exp = dict(plan["save_exp"])
                    save_exp["ton"] = ton_value
                    save_exp["ton_next"] = next_ton
                    payload["save_exp"] = save_exp
                    stage_tasks.append(payload)

                logger.info(
                    "[TON-STAGE] ton=%s tasks=%s cases=%s",
                    ton_value,
                    len(stage_tasks),
                    len(cases),
                )
                _run_stage_tasks(stage_tasks)
                if shutdown_event.is_set():
                    logger.info("Shutdown requested; stopping stage loop")
                    break

                next_candidates = 0
                for case in cases:
                    stats_path = Path(case["save_dir"]) / "stats.json"
                    stats, reason = _load_stats_payload(stats_path)
                    if not stats:
                        if reason != "missing_stats":
                            logger.warning(
                                "[TON-STAGE] idx=%s ton=%s stats_update=failed reason=%s",
                                case["idx"],
                                ton_value,
                                reason,
                            )
                        continue
                    last_ton = _extract_last_ton(stats)
                    if last_ton != ton_value:
                        continue
                    should_continue, reason = _derive_stage_outcome_payload(stats)
                    status = "continue" if should_continue else "stop"
                    updated = update_ton_progress_stats(
                        stats_path,
                        current_ton=ton_value,
                        status=status,
                        reason=reason,
                        next_ton=next_ton,
                    )
                    if not updated:
                        logger.warning(
                            "[TON-STAGE] idx=%s ton=%s stats_update=failed reason=%s",
                            case["idx"],
                            ton_value,
                            reason,
                        )
                if next_ton is not None:
                    for case in cases:
                        if _should_run_ton(
                            case,
                            next_ton,
                            ton_sequence,
                            force_refresh=force_refresh,
                        ):
                            next_candidates += 1

                logger.info(
                    "[TON-STAGE-END] ton=%s next_candidates=%s",
                    ton_value,
                    next_candidates,
                )
                if next_ton is None or next_candidates == 0:
                    break
        except KeyboardInterrupt:
            logger.warning("Main loop interrupted; shutting down workers")
            interrupted = True
            shutdown_event.set()

    if interrupted or shutdown_event.is_set():
        logger.info("Tasks interrupted; shutdown requested")
    else:
        logger.info("All tasks completed")


__all__ = [
    "run_launcher",
]
