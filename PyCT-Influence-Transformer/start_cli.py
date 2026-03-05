from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, Optional, Sequence, Tuple

from start_config import (
    _DEFAULT_PIXEL_SEARCH,
    _LOG_LEVEL_CHOICES,
)

logger = logging.getLogger("ct.cli")


def _parse_pixel_search(value: str) -> Tuple[int, ...]:
    """Parse comma-separated ton values into a strict, ordered tuple."""
    parts = [part.strip() for part in value.split(",")]
    sequence: list[int] = []
    for part in parts:
        if not part:
            continue
        ton = int(part)
        if ton < 1:
            raise argparse.ArgumentTypeError("pixel search values must be >= 1.")
        if ton not in sequence:
            sequence.append(ton)
    if not sequence:
        raise argparse.ArgumentTypeError("pixel search sequence cannot be empty.")
    return tuple(sequence)


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
        "--num-process",
        type=int,
        default=1,
        help="Number of worker processes used to dispatch attacks.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-wave timeout in seconds; resets every time ton value increases.",
    )
    parser.add_argument(
        "--no-constraint-build-timeout",
        dest="constraint_build_timeout",
        action="store_false",
        help="Disable the 30s timeout when constructing SMT formulas (default: enabled).",
    )
    parser.set_defaults(constraint_build_timeout=True)
    parser.add_argument(
        "--solver-run-timeout",
        type=int,
        default=60,
        help="Wall-clock timeout (seconds) for each SMT solver invocation; 0 disables this wrapper timeout (default: 60).",
    )
    parser.add_argument(
        "--score-alpha",
        type=float,
        default=None,
        help="Weight for path_len term in priority score (0..1). Required unless --attack-mode queue.",
    )
    parser.add_argument(
        "--symbolic-path-threshold",
        type=int,
        default=8000,
        help="Disable symbolic tracking when path_len reaches this threshold (default: 8000).",
    )
    parser.add_argument(
        "--pixel-search",
        type=_parse_pixel_search,
        default=_parse_pixel_search(",".join(str(v) for v in _DEFAULT_PIXEL_SEARCH)),
        help="Comma-separated ton sequence to try per input (default: 1,2,4,8,16,32).",
    )
    parser.add_argument(
        "--attack-mode",
        default="shap",
        choices=("shap", "random", "random-assign", "queue"),
        help="Select attack strategy.",
    )
    parser.add_argument(
        "--dataset",
        default="fashion_mnist",
        choices=("fashion_mnist", "cifar10", "mnist"),
        help="Dataset to use for task generation.",
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
        "--force-refresh",
        action="store_true",
        help="Rebuild cached outputs even when existing experiment folders are present.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=_LOG_LEVEL_CHOICES,
        help="Root logging level for the launcher (default: INFO).",
    )
    parser.add_argument(
        "--explore-log-level",
        choices=_LOG_LEVEL_CHOICES,
        help="Override log level for ct.explore (falls back to --log-level).",
    )
    parser.add_argument(
        "--solver-log-level",
        choices=_LOG_LEVEL_CHOICES,
        help="Override log level for ct.solver (falls back to --log-level).",
    )
    parser.add_argument(
        "--log-file",
        help="Optional path to append structured logs in addition to stdout.",
    )
    args = parser.parse_args(argv)
    if args.attack_mode != "queue" and args.score_alpha is None:
        parser.error("--score-alpha is required unless --attack-mode queue")
    return args


def configure_logging(args: argparse.Namespace) -> None:
    """Initialize logging once per invocation."""
    log_kwargs: Dict[str, Any] = {
        "level": getattr(logging, args.log_level.upper(), logging.INFO),
        "format": "%(levelname)s | %(name)s | %(message)s",
    }
    if args.log_file:
        log_kwargs["filename"] = args.log_file
        log_kwargs["filemode"] = "a"
    logging.basicConfig(**log_kwargs)

    overrides = (
        ("ct.explore", args.explore_log_level),
        ("ct.solver", args.solver_log_level),
    )
    for name, level in overrides:
        if not level:
            continue
        logging.getLogger(name).setLevel(getattr(logging, level.upper(), log_kwargs["level"]))


__all__ = ["parse_args", "configure_logging"]
