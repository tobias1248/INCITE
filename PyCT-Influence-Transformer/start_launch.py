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
from utils.experiment_task_specs import (
    fashion_mnist_transformer_random,
    fashion_mnist_transformer_shap,
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


def _worker(
    task_queue: Queue,
    timeout: int,
    constraint_build_timeout: bool,
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
    resume_index = 0
    force_refresh = args.force_refresh
    if not args.force_refresh:
        resume_index, require_force = _derive_resume_plan(args.model_name, attack_mode, args.first_n)
        force_refresh = require_force
        if resume_index >= args.first_n:
            logger.info("All requested inputs already completed; nothing to do.")
            return

    first_n_range = range(resume_index, args.first_n)
    logger.info("Scheduling inputs from idx=%s to %s", resume_index, args.first_n - 1)
    if resume_index > 0:
        logger.info("Resuming from idx=%s (force=%s)", resume_index, "yes" if force_refresh else "no")

    if args.attack_mode == "shap":
        inputs = fashion_mnist_transformer_shap(
            args.model_name,
            first_n_img=first_n_range,
            force=force_refresh,
            ton_values=args.pixel_search,
            attack_mode="shap",
        )
    elif args.attack_mode == "random":
        inputs = fashion_mnist_transformer_random(
            args.model_name,
            first_n_img=first_n_range,
            ton_values=args.pixel_search,
            force=force_refresh,
            base_seed=args.random_seed,
            attack_mode="random",
        )
    elif args.attack_mode == "random-assign":
        exp_prefix = f"random_assign_{args.pixel_source}"
        if args.pixel_source == "random":
            inputs = fashion_mnist_transformer_random(
                args.model_name,
                first_n_img=first_n_range,
                ton_values=args.pixel_search,
                force=force_refresh,
                base_seed=args.random_seed,
                exp_prefix=exp_prefix,
                attack_mode="random-assign",
            )
        else:
            inputs = fashion_mnist_transformer_shap(
                args.model_name,
                first_n_img=first_n_range,
                force=force_refresh,
                ton_values=args.pixel_search,
                exp_prefix=exp_prefix,
                attack_mode="random-assign",
            )
    elif args.attack_mode == "queue":
        # For queue mode, reuse SHAP task generation and execute via QueueRunner inline.
        inputs = fashion_mnist_transformer_shap(
            args.model_name,
            first_n_img=first_n_range,
            force=force_refresh,
            ton_values=args.pixel_search,
            exp_prefix="queue",
            attack_mode="queue",
        )
        from utils.experiment_runner import run_attack_with_queue

        logger.info("Starting queue-mode run with %s task(s)", len(inputs))
        try:
            run_attack_with_queue(
                inputs,
                timeout=args.timeout,
                constraint_build_timeout=args.constraint_build_timeout,
                norm=args.norm_01,
                collect_constraints_with="queue",
            )
            logger.info("Queue-mode run completed")
        except KeyboardInterrupt:
            logger.warning("Queue-mode interrupted by user; stopping tasks")
        return
    else:
        raise ValueError(f"Unsupported attack mode: {args.attack_mode}")

    logger.info(
        "Prepared %s input(s) for attack=%s ton_sequence=%s",
        len(inputs),
        args.attack_mode,
        ",".join(str(v) for v in args.pixel_search),
    )
    time.sleep(3)

    task_queue: Queue = Queue()
    for task in inputs:
        task_queue.put(task)
    # Sentinels to allow clean worker exit
    for _ in range(args.num_process):
        task_queue.put(None)

    try:
        worker_count = min(args.num_process, max(len(inputs), 1))
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
    except KeyboardInterrupt:
        logger.warning("Main loop interrupted; shutting down workers")
        interrupted = True
        shutdown_event.set()
    finally:
        for process in running_processes:
            if process.is_alive():
                process.terminate()
            process.join()

    if interrupted or shutdown_event.is_set():
        logger.info("Tasks interrupted; shutdown requested")
    else:
        logger.info("All tasks completed")


__all__ = [
    "run_launcher",
]
