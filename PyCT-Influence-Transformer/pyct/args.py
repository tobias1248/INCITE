from __future__ import annotations

import argparse
import logging
import math
from typing import Any, Dict, Optional, Sequence, Tuple

from libct.aces_like import (
    DEFAULT_PWL_ERROR_TOLERANCE,
    DEFAULT_PWL_MAX_SEGMENTS,
)

from pyct.config import (
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


def _parse_case_indices(value: str) -> Tuple[int, ...]:
    """Parse comma-separated case indices into a strict, ordered tuple."""
    parts = [part.strip() for part in value.split(",")]
    sequence: list[int] = []
    for part in parts:
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            raise argparse.ArgumentTypeError("case indices must be >= 0.")
        if idx not in sequence:
            sequence.append(idx)
    if not sequence:
        raise argparse.ArgumentTypeError("case indices sequence cannot be empty.")
    return tuple(sequence)


def _parse_non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0.")
    return parsed


def _parse_non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0.")
    return parsed


def _parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1.")
    return parsed


def _parse_positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and > 0.")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for experiment launcher."""
    parser = argparse.ArgumentParser(
        description="Run PyCT attack experiments across multiple processes."
    )
    parser.add_argument(
        "--model-name",
        default="transformer_fashion_mnist",
        help="Model artifact name under ./model (without .h5 extension).",
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
        help="Per-stage timeout in seconds (applies for each ton stage).",
    )
    parser.add_argument(
        "--no-constraint-build-timeout",
        dest="constraint_build_timeout",
        action="store_false",
        help="Disable the 30s timeout when constructing SMT formulas (default: enabled).",
    )
    parser.set_defaults(constraint_build_timeout=True)
    parser.add_argument(
        "--constraint-build-timeout-seconds",
        type=int,
        default=30,
        help=(
            "Timeout in seconds for SMT formula construction when build timeout "
            "is enabled (default: 30)."
        ),
    )
    parser.add_argument(
        "--solver-run-timeout",
        type=int,
        default=60,
        help="Wall-clock timeout (seconds) per SMT solver invocation; 0 disables wrapper timeout.",
    )
    parser.add_argument(
        "--score-alpha",
        type=float,
        default=None,
        help=(
            "Weight of path_len penalty in priority score (0..1). Required "
            "except for queue and hybrid-de; hybrid-de ranks by Keras margin."
        ),
    )
    parser.add_argument(
        "--symbolic-path-threshold",
        type=int,
        default=8000,
        help="Disable symbolic tracking when path_len reaches this threshold (default: 8000).",
    )
    parser.add_argument(
        "--enable-constraint-log",
        action="store_true",
        help="Enable verbose push/pop constraint logs (default: disabled).",
    )
    parser.add_argument(
        "--ternary-simplification",
        action="store_true",
        help="Enable threshold-based ternary simplification for supported layers.",
    )
    parser.add_argument(
        "--ternary-threshold-scale",
        type=_parse_non_negative_float,
        default=0.75,
        help=(
            "Non-negative scale for ternary delta: threshold_scale * "
            "mean(abs(W)) (default: 0.75)."
        ),
    )
    parser.add_argument(
        "--ternary-fallback",
        action="store_true",
        help="Retry timeout cases with ternary simplification for shap/queue attacks.",
    )
    parser.add_argument(
        "--pixel-search",
        type=_parse_pixel_search,
        default=_parse_pixel_search(",".join(str(v) for v in _DEFAULT_PIXEL_SEARCH)),
        help="Comma-separated ton sequence per input, e.g. 1,2,4,8,16,32.",
    )
    parser.add_argument(
        "--attack-mode",
        default="shap",
        choices=("shap", "random", "random-assign", "queue", "global-real", "hybrid-de"),
        help="Attack strategy: shap/random/random-assign/queue/global-real/hybrid-de.",
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
        help="Random seed for random coordinate generation (random/random-assign modes).",
    )
    parser.add_argument(
        "--pixel-source",
        default="random",
        choices=("random", "shap"),
        help="Pixel source for random-assign mode only (ignored by shap/random/queue).",
    )
    parser.add_argument(
        "--pixel-selector",
        default="pixel-shap",
        choices=("pixel-shap", "patch-shap", "token-shap"),
        help=(
            "Coordinate selector for SHAP attacks (default: pixel-shap). "
            "patch-shap/token-shap are CIFAR10-only and require "
            "--pixel-search 1."
        ),
    )
    parser.add_argument(
        "--norm-01",
        dest="norm_01",
        action="store_true",
        help=(
            "Constrain solver-generated concolic input variables to the [0, 1] "
            "image range. This is enabled by default for supported image datasets."
        ),
    )
    parser.add_argument(
        "--no-norm-01",
        dest="norm_01",
        action="store_false",
        help=(
            "Unsafe/debug-only: do not add [0, 1] range constraints for "
            "solver-generated input variables. Rejected for supported image datasets."
        ),
    )
    parser.set_defaults(norm_01=True)
    parser.add_argument(
        "--first-n",
        type=int,
        default=100,
        help="Number of inputs to process from index 0.",
    )
    parser.add_argument(
        "--case-indices",
        type=_parse_case_indices,
        help="Optional comma-separated case indices to process instead of --first-n, e.g. 3,7,11.",
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
        help="Rerun stages even when stats indicate the corresponding stage has already completed.",
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
    parser.add_argument(
        "--error-retry-limit",
        type=_parse_non_negative_int,
        default=2,
        help=(
            "Retry limit for constraint_transfer_failure on the same ton stage "
            "(default: 2)."
        ),
    )
    parser.add_argument(
        "--global-x-min",
        type=float,
        default=-0.1,
        help="Lower bound for GlobalReal X, or each hybrid brightness/contrast variable (default: -0.1).",
    )
    parser.add_argument(
        "--global-x-max",
        type=float,
        default=0.1,
        help="Upper bound for GlobalReal X, or each hybrid brightness/contrast variable (default: 0.1).",
    )
    parser.add_argument(
        "--de-maxiter",
        type=_parse_non_negative_int,
        default=75,
        help="Maximum DE generations before a failed case is passed to PyCT (default: 75).",
    )
    parser.add_argument(
        "--de-population-size",
        type=_parse_positive_int,
        default=400,
        help="Number of candidate images in each DE population (default: 400).",
    )
    parser.add_argument(
        "--global-x-bounds-mode",
        choices=("clip", "strict"),
        default="clip",
        help=(
            "GlobalReal input bounds: clip uses per-element clipping; strict "
            "shrinks X so every shifted element remains in [0,1]."
        ),
    )
    parser.add_argument(
        "--global-shift-kind",
        choices=(
            "shap-sign",
            "brightness",
            "contrast",
            "aces-brightness",
            "aces-contrast",
        ),
        default="shap-sign",
        help=(
            "Shared GlobalReal direction: shap-sign follows per-pixel SHAP signs; "
            "brightness shifts all channels equally; contrast scales deviations "
            "from each image's RGB channel mean; aces-brightness and aces-contrast "
            "use the ACES-like piecewise-linear transform."
        ),
    )
    parser.add_argument(
        "--shap-sign-epsilon",
        type=float,
        default=0.0,
        help="Treat target-class SHAP magnitudes <= epsilon as zero (default: 0).",
    )
    parser.add_argument(
        "--shap-output-root",
        default="shap_target_class",
        help="Root directory for canonical target-class SHAP caches.",
    )
    parser.add_argument(
        "--aces-pwl-max-segments",
        type=_parse_non_negative_int,
        default=DEFAULT_PWL_MAX_SEGMENTS,
        help="Maximum number of segments for ACES-like shared-X PWL (default: 32).",
    )
    parser.add_argument(
        "--aces-pwl-error-tolerance",
        type=_parse_non_negative_float,
        default=DEFAULT_PWL_ERROR_TOLERANCE,
        help="Maximum sampled RGB error for ACES-like PWL (default: 1/255).",
    )
    parser.add_argument(
        "--no-global-real-probe",
        dest="global_real_probe",
        action="store_false",
        help="Disable concrete X probing after ACES-like SAT candidates.",
    )
    parser.set_defaults(global_real_probe=True)
    parser.add_argument(
        "--global-real-probe-points",
        type=_parse_positive_int,
        default=17,
        help="Initial concrete X probe budget for ACES-like candidates (default: 17).",
    )
    parser.add_argument(
        "--global-real-probe-refinements",
        type=_parse_non_negative_int,
        default=8,
        help="Maximum bracket refinement steps per X probe (default: 8).",
    )
    parser.add_argument(
        "--global-real-probe-tolerance-fraction",
        type=_parse_positive_float,
        default=1.0 / 1024.0,
        help=(
            "X bracket tolerance as a fraction of the effective range "
            "(default: 1/1024)."
        ),
    )
    args = parser.parse_args(argv)
    if args.attack_mode not in {"queue", "hybrid-de"} and args.score_alpha is None:
        parser.error("--score-alpha is required unless --attack-mode queue or hybrid-de")
    if args.ternary_fallback and args.ternary_simplification:
        parser.error("--ternary-fallback cannot be combined with --ternary-simplification")
    if args.ternary_fallback and args.attack_mode not in {"shap", "queue"}:
        parser.error("--ternary-fallback requires --attack-mode shap or --attack-mode queue")
    if not args.norm_01 and args.dataset in {"fashion_mnist", "cifar10", "mnist"}:
        parser.error(
            "--no-norm-01 is not supported for image datasets. PyCT image attacks "
            "assume normalized model inputs in [0, 1] and must bind solver-generated "
            "input variables to that range."
        )
    if args.attack_mode in {"global-real", "hybrid-de"}:
        if args.dataset != "cifar10":
            parser.error(
                f"--attack-mode {args.attack_mode} currently requires --dataset cifar10"
            )
        if not math.isfinite(args.global_x_min) or not math.isfinite(args.global_x_max):
            parser.error("--global-x-min and --global-x-max must be finite")
        if args.global_x_min > args.global_x_max:
            parser.error("--global-x-min must be <= --global-x-max")
        if not args.global_x_min <= 0.0 <= args.global_x_max:
            parser.error("GlobalReal X bounds must include 0")
        if not math.isfinite(args.shap_sign_epsilon) or args.shap_sign_epsilon < 0:
            parser.error("--shap-sign-epsilon must be finite and >= 0")
        if args.attack_mode == "hybrid-de":
            if args.global_shift_kind not in {
                "brightness", "contrast", "aces-brightness", "aces-contrast"
            }:
                parser.error(
                    "--attack-mode hybrid-de requires --global-shift-kind "
                    "brightness, contrast, aces-brightness or aces-contrast"
                )
            if args.global_x_bounds_mode != "clip":
                parser.error("--attack-mode hybrid-de requires --global-x-bounds-mode clip")
            if args.global_x_min >= args.global_x_max:
                parser.error("hybrid-de requires --global-x-min < --global-x-max")
            if args.de_population_size < 5:
                parser.error("--de-population-size must be >= 5")
        if args.global_shift_kind.startswith("aces-"):
            if args.global_x_bounds_mode != "clip":
                parser.error("ACES-like shifts require --global-x-bounds-mode clip")
            if args.global_x_min >= args.global_x_max:
                parser.error("ACES-like GlobalReal X bounds must satisfy min < max")
            if args.aces_pwl_max_segments < 1:
                parser.error("--aces-pwl-max-segments must be >= 1 for ACES-like shifts")
            if (
                not math.isfinite(args.aces_pwl_error_tolerance)
                or args.aces_pwl_error_tolerance <= 0
            ):
                parser.error(
                    "--aces-pwl-error-tolerance must be finite and > 0 for ACES-like shifts"
                )
            if args.global_real_probe_tolerance_fraction <= 0.0:
                parser.error(
                    "--global-real-probe-tolerance-fraction must be > 0"
                )
    if args.pixel_selector in {"patch-shap", "token-shap"}:
        if args.attack_mode != "shap":
            parser.error(f"--pixel-selector {args.pixel_selector} requires --attack-mode shap")
        if args.dataset != "cifar10":
            parser.error(
                f"--pixel-selector {args.pixel_selector} requires --dataset cifar10"
            )
        if tuple(args.pixel_search) != (1,):
            parser.error(f"--pixel-selector {args.pixel_selector} requires --pixel-search 1")
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
