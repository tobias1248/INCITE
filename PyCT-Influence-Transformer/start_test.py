#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import json
import time
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.experiment_task_specs import (
    fashion_mnist_transformer_random,
    fashion_mnist_transformer_shap,
)

_INPUT_PREFIX = "fashion_mnist_test_"
_QUEUE_TYPE = "priority_queue"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for transformer experiments."""
    parser = argparse.ArgumentParser(
        description="Run PyCT transformer experiments across multiple processes."
    )
    parser.add_argument(
        "--model-name",
        default="transformer_fashion_mnist",
        help="Identifier for the model artifact to load.",
    )
    parser.add_argument(
        "--model-type",
        default="transformer",
        help="Logical model type propagated to subprocess tasks.",
    )
    parser.add_argument(
        "--num-process",
        type=int,
        default=1,
        help="Number of worker processes used to dispatch attacks.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-input timeout in seconds passed to each subprocess.",
    )
    parser.add_argument(
        "--ton",
        type=int,
        default=1,
        help="Number of pixels/features to perturb per attack when supported.",
    )
    parser.add_argument(
        "--attack-mode",
        default="shap",
        choices=("shap", "random", "random-assign"),
        help="Select attack strategy.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=2024,
        help="Seed controlling deterministic random selection (random/random-assign modes).",
    )
    parser.add_argument(
        "--pixel-source",
        default="random",
        choices=("random", "shap"),
        help="Pixel selection strategy used for random-assign baseline.",
    )
    parser.add_argument(
        "--norm-01",
        dest="norm_01",
        action="store_true",
        help="Request normalization of inputs into the [0, 1] range.",
    )
    parser.add_argument(
        "--no-norm-01",
        dest="norm_01",
        action="store_false",
        help="Disable input normalization. This is the default.",
    )
    parser.set_defaults(norm_01=False)
    parser.add_argument(
        "--first-n",
        type=int,
        default=100,
        help="Number of Fashion-MNIST test images to enqueue from index 0.",
    )
    parser.add_argument(
        "--spawn-delay",
        type=float,
        default=1.0,
        help="Delay in seconds between spawning subprocesses.",
    )
    parser.add_argument(
        "--solver",
        default="cvc4",
        choices=("cvc4",),
        help="SMT solver to configure before invoking PyCT (limited to supported options).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Rebuild cached outputs even when existing experiment folders are present.",
    )
    return parser.parse_args(argv)


def _configure_solver(selected_solver: str) -> None:
    """Monkey-patch ExplorationEngine to honor the requested solver."""
    if not selected_solver:
        return

    try:
        import libct.explore as explore_module
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("Unable to import libct.explore; solver configuration aborted.") from exc

    engine_cls = explore_module.ExplorationEngine
    original_init = getattr(engine_cls, "_pyct_original_init", engine_cls.__init__)

    if getattr(engine_cls, "_pyct_configured_solver", None) == selected_solver:
        return

    @functools.wraps(original_init)
    def patched_init(self, *, solver: str = "cvc4", **kwargs: Any) -> None:
        solver_to_use = selected_solver if solver == "cvc4" else solver
        original_init(self, solver=solver_to_use, **kwargs)

    engine_cls.__init__ = patched_init  # type: ignore[method-assign]
    engine_cls._pyct_original_init = original_init  # type: ignore[attr-defined]
    engine_cls._pyct_configured_solver = selected_solver  # type: ignore[attr-defined]


def _resolve_experiment_layout(
    attack_mode: str,
    ton: int,
    *,
    pixel_source: str = "random",
) -> Tuple[str, str]:
    if attack_mode == "shap":
        return _QUEUE_TYPE, "shap_1"
    if attack_mode == "random":
        return _QUEUE_TYPE, f"random_select/random_{ton}"
    if attack_mode == "random-assign":
        if pixel_source == "random":
            return _QUEUE_TYPE, f"random_assign_random/random_{ton}"
        if pixel_source == "shap":
            return _QUEUE_TYPE, f"random_assign_shap/shap_{ton}"
        raise ValueError(f"Unsupported pixel source: {pixel_source}")
    raise ValueError(f"Unsupported attack mode: {attack_mode}")


def _derive_resume_plan(
    model_name: str,
    queue_type: str,
    experiment_name: str,
    first_n: int,
) -> Tuple[int, bool]:
    base_dir = Path("exp") / model_name / queue_type / experiment_name
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
    is_finish = False
    if stats_path.is_file():
        try:
            with stats_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            is_finish = bool(payload.get("meta", {}).get("is_finish"))
        except (json.JSONDecodeError, OSError):
            is_finish = False

    if not is_finish:
        return latest_idx, True

    resume_idx = latest_idx + 1
    return min(resume_idx, first_n), False


def _worker(
    sub_tasks: List[Dict[str, Any]],
    timeout: int,
    norm_01: bool,
    model_type: str,
    solver: str,
    attack_mode: str,
    pixel_source: str,
    base_seed: int,
) -> None:
    """Entry point for each subprocess that forwards to the util helper."""
    if attack_mode == "random-assign":
        from utils.experiment_runner import run_attack_with_random_assign

        run_attack_with_random_assign(
            sub_tasks,
            timeout,
            norm_01,
            pixel_source=pixel_source,
            base_seed=base_seed,
            model_type=model_type,
        )
        return

    _configure_solver(solver)
    from utils.experiment_runner import run_attack_with_shap

    run_attack_with_shap(
        sub_tasks,
        timeout,
        norm_01,
        model_type=model_type,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.num_process < 1:
        raise ValueError("--num-process must be >= 1")
    if args.first_n < 1:
        raise ValueError("--first-n must be >= 1")
    if args.timeout < 1:
        raise ValueError("--timeout must be >= 1 second")
    if args.spawn_delay < 0:
        raise ValueError("--spawn-delay must be non-negative")
    if args.attack_mode == "shap" and args.ton != 1:
        raise ValueError("SHAP mode currently supports only --ton=1.")
    if (
        args.attack_mode == "random-assign"
        and args.pixel_source == "shap"
        and args.ton != 1
    ):
        raise ValueError("Random-assign with SHAP pixels currently supports only --ton=1.")
    if args.ton < 1:
        raise ValueError("--ton must be >= 1")

    queue_type, exp_name = _resolve_experiment_layout(
        args.attack_mode,
        args.ton,
        pixel_source=args.pixel_source,
    )
    resume_index = 0
    force_refresh = args.force_refresh
    if not args.force_refresh:
        resume_index, require_force = _derive_resume_plan(
            args.model_name,
            queue_type,
            exp_name,
            args.first_n,
        )
        force_refresh = require_force
        if resume_index >= args.first_n:
            print("All requested inputs already completed; nothing to do.")
            return

    first_n_range = range(resume_index, args.first_n)
    if resume_index > 0:
        print(f"[INFO] Resuming from idx={resume_index} (force={'yes' if force_refresh else 'no'}).")

    if args.attack_mode == "shap":
        inputs = fashion_mnist_transformer_shap(
            args.model_name,
            first_n_img=first_n_range,
            force=force_refresh,
        )
    elif args.attack_mode == "random":
        inputs = fashion_mnist_transformer_random(
            args.model_name,
            first_n_img=first_n_range,
            ton_values=[args.ton],
            force=force_refresh,
            base_seed=args.random_seed,
        )
    elif args.attack_mode == "random-assign":
        exp_prefix = f"random_assign_{args.pixel_source}"
        if args.pixel_source == "random":
            inputs = fashion_mnist_transformer_random(
                args.model_name,
                first_n_img=first_n_range,
                ton_values=[args.ton],
                force=force_refresh,
                base_seed=args.random_seed,
                exp_prefix=exp_prefix,
            )
        else:
            inputs = fashion_mnist_transformer_shap(
                args.model_name,
                first_n_img=first_n_range,
                force=force_refresh,
                exp_prefix=exp_prefix,
            )
    else:
        raise ValueError(f"Unsupported attack mode: {args.attack_mode}")

    print("#" * 40, f"number of inputs: {len(inputs)}", "#" * 45)
    time.sleep(3)

    all_subprocess_tasks: List[List[Dict[str, Any]]] = [[] for _ in range(args.num_process)]
    cursor = 0
    for task in inputs:
        all_subprocess_tasks[cursor].append(task)
        cursor = (cursor + 1) % args.num_process

    running_processes: List[Process] = []
    for sub_tasks in all_subprocess_tasks:
        if not sub_tasks:
            continue
        process = Process(
            target=_worker,
            args=(
                sub_tasks,
                args.timeout,
                args.norm_01,
                args.model_type,
                args.solver,
                args.attack_mode,
                args.pixel_source,
                args.random_seed,
            ),
        )
        process.start()
        running_processes.append(process)
        time.sleep(args.spawn_delay)

    for process in running_processes:
        process.join()

    print("done")


if __name__ == "__main__":
    main()
