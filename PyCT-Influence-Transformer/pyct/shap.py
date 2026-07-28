import argparse
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence

from explainability.shap_contract import (
    DEFAULT_TARGET_CLASS_SHAP_ROOT,
    TARGET_CLASS_ATTRIBUTION,
    TARGET_CLASS_SHAP_SCHEMA_VERSION,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for SHAP map calculation workflow."""
    parser = argparse.ArgumentParser(
        description="Compute SHAP value maps for transformer experiments."
    )
    parser.add_argument(
        "--model-name",
        default="transformer_fashion_mnist",
        help="Model identifier used to locate weights and SHAP artifacts.",
    )
    parser.add_argument(
        "--dataset",
        choices=("fashion_mnist", "cifar10", "mnist"),
        required=True,
        help="Dataset to use when preparing inputs/backgrounds for SHAP ['fashion_mnist', 'cifar10', 'mnist'].",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=100,
        help="Number of inputs (starting from index 0) to process.",
    )
    parser.add_argument(
        "--explainer-type",
        choices=("gradient", "kernel"),
        default="gradient",
        help="SHAP explainer implementation to use.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_TARGET_CLASS_SHAP_ROOT,
        help=(
            "Root directory where target-class SHAP JSON outputs are stored "
            f"(default: {DEFAULT_TARGET_CLASS_SHAP_ROOT})."
        ),
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Recompute SHAP even if cached results already exist.",
    )
    parser.add_argument(
        "--background-per-class",
        type=int,
        default=3,
        help="Number of background samples per class for SHAP (default: 3).",
    )
    parser.add_argument(
        "--background-seed",
        type=int,
        default=2233,
        help="Random seed for SHAP background sampling (default: 2233).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.0,
        help="Optional delay before processing inputs.",
    )
    return parser.parse_args(argv)


def build_timing_summary(
    artifacts: Sequence[Dict[str, Any]],
    *,
    model_name: str,
    dataset: str,
    explainer_type: str,
    output_root: str,
) -> Dict[str, Any]:
    """Build a stable JSON-serializable SHAP timing summary."""
    compute_times: List[float] = [
        float(item.get("compute_seconds", 0.0))
        for item in artifacts
        if bool(item.get("computed", False))
    ]
    cached_count = sum(1 for item in artifacts if bool(item.get("was_cached", False)))
    total_compute_seconds = float(sum(compute_times))
    summary: Dict[str, Any] = {
        "model": model_name,
        "dataset": dataset,
        "explainer_type": explainer_type,
        "output_root": output_root,
        "schema_version": TARGET_CLASS_SHAP_SCHEMA_VERSION,
        "attribution_target": TARGET_CLASS_ATTRIBUTION,
        "total_inputs": len(artifacts),
        "computed": len(compute_times),
        "cached": cached_count,
        "total_compute_seconds": total_compute_seconds,
        "mean_seconds": None,
        "median_seconds": None,
        "min_seconds": None,
        "max_seconds": None,
    }
    if compute_times:
        summary.update(
            {
                "mean_seconds": float(statistics.mean(compute_times)),
                "median_seconds": float(statistics.median(compute_times)),
                "min_seconds": float(min(compute_times)),
                "max_seconds": float(max(compute_times)),
            }
        )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for generating SHAP maps."""
    args = parse_args(argv)

    if args.first_n < 1:
        raise ValueError("--first-n must be >= 1")
    if args.sleep_seconds < 0:
        raise ValueError("--sleep-seconds must be non-negative")

    # Import heavy dependencies lazily to keep startup fast.
    from tasks.builders.cifar10 import cifar10_cal_shap_specs
    from tasks.builders.fashion_mnist import fashion_mnist_transformer_shap_calculate_all
    from tasks.builders.mnist import mnist_transformer_shap_calculate_all

    if args.sleep_seconds:
        time.sleep(args.sleep_seconds)

    os.environ["PYCT_BG_PER_CLASS"] = str(args.background_per_class)
    os.environ["PYCT_BG_SEED"] = str(args.background_seed)

    dataset_handlers = {
        "fashion_mnist": fashion_mnist_transformer_shap_calculate_all,
        "cifar10": cifar10_cal_shap_specs,
        "mnist": mnist_transformer_shap_calculate_all,
    }
    handler = dataset_handlers[args.dataset]

    artifacts = handler(
        args.model_name,
        first_n_img=args.first_n,
        force_refresh=args.force_refresh,
        explainer_type=args.explainer_type,
        output_root=args.output_root,
    )
    print("#" * 40, f"processed inputs: {len(artifacts)}", "#" * 45)
    for info in artifacts:
        status = "SKIP" if info["was_cached"] else "CALC"
        print(f"[{status}] idx={info['idx']} -> {info['output_path']}")

    summary = build_timing_summary(
        artifacts,
        model_name=args.model_name,
        dataset=args.dataset,
        explainer_type=args.explainer_type,
        output_root=args.output_root,
    )
    print("[SHAP-SUMMARY-JSON] " + json.dumps(summary, sort_keys=True))
    print("SHAP calculations complete.")


if __name__ == "__main__":
    main()
