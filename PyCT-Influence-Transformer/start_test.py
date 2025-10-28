import argparse
import functools
import json
import time
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_QUEUE_TYPE = "priority_queue"
_EXPERIMENT_NAME = "shap_1"
_INPUT_PREFIX = "fashion_mnist_test_"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Define and parse command-line arguments for the transformer multi-run CLI."""
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
        help="Rebuild cached SHAP inputs even when existing experiment folders are present.",
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


def _derive_resume_plan(model_name: str, first_n: int) -> Tuple[int, bool]:
    """Infer the starting index and whether forcing refresh is required for resumption."""
    base_dir = Path("exp") / model_name / _QUEUE_TYPE / _EXPERIMENT_NAME
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
) -> None:
    """Entry point for each subprocess that forwards to the existing util helper."""
    _configure_solver(solver)
    from utils.experiment_runner import run_attack_with_shap

    run_attack_with_shap(
        sub_tasks,
        timeout,
        norm_01,
        model_type=model_type,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI wrapper for running batch transformer concolic tests with configurable settings."""
    args = parse_args(argv)

    if args.num_process < 1:
        raise ValueError("--num-process must be >= 1")
    if args.first_n < 1:
        raise ValueError("--first-n must be >= 1")
    if args.timeout < 1:
        raise ValueError("--timeout must be >= 1 second")
    if args.spawn_delay < 0:
        raise ValueError("--spawn-delay must be non-negative")

    from utils.experiment_task_specs import fashion_mnist_transformer_shap

    resume_index = 0
    force_refresh = args.force_refresh
    if not args.force_refresh:
        resume_index, require_force = _derive_resume_plan(args.model_name, args.first_n)
        force_refresh = require_force
        if resume_index >= args.first_n:
            print("All requested inputs already completed; nothing to do.")
            return

    first_n_range = range(resume_index, args.first_n)
    if resume_index > 0:
        print(f"[INFO] Resuming from idx={resume_index} (force={'yes' if force_refresh else 'no'}).")

    inputs = fashion_mnist_transformer_shap(
        args.model_name,
        first_n_img=first_n_range,
        force=force_refresh,
    )
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
            args=(sub_tasks, args.timeout, args.norm_01, args.model_type, args.solver),
        )
        process.start()
        running_processes.append(process)
        time.sleep(args.spawn_delay)

    for process in running_processes:
        process.join()

    print("done")


if __name__ == "__main__":
    main()
