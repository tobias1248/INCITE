import argparse
import os
import time
from typing import Optional, Sequence


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
        default="shap_value_all_layer",
        help="Root directory where SHAP JSON outputs are stored.",
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

    print("SHAP calculations complete.")


if __name__ == "__main__":
    main()
