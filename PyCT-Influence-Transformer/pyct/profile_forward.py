"""CLI for diagnostic profiling of one cached CIFAR10 case."""
from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="cifar10_concolic_transformer")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--mode", choices=("all", "keras_concrete", "nnmodel_concrete", "symbolic_path", "symbolic_no_path"),
                        default="all", help="Run only this mode, or all four (default)")
    parser.add_argument("--global-shift-kind", choices=("aces-brightness", "aces-contrast"),
                        default="aces-contrast", help="ACES-like symbolic input control")
    parser.add_argument("--timeout", type=float, default=1800, help="Forward deadline per mode, seconds")
    parser.add_argument("--setup-timeout", type=float, default=300, help="Hard setup deadline per mode, seconds")
    parser.add_argument("--symbolic-path-threshold", type=int, default=2000)
    parser.add_argument("--dataset-cache", type=Path, help="Offline CIFAR10 test_batch file")
    parser.add_argument("--output-dir", type=Path, help="New directory; refuses reuse")
    args = parser.parse_args(argv)
    if not args.model_name or Path(args.model_name).name != args.model_name or args.model_name in (".", ".."):
        parser.error("model-name must be a filename stem")
    if not 0 <= args.case_index < 10000:
        parser.error("case-index must be in [0,9999]")
    for name in ("timeout", "setup_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name + " must be finite and positive")
    if args.symbolic_path_threshold < 1:
        parser.error("symbolic-path-threshold must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Set before any NumPy/TensorFlow imports, including spawned child imports.
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
                      TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
                      PYCT_ALLOW_DATASET_DOWNLOAD="0")
    from datasets.keras_cache import get_keras_datasets_dir
    from tasks.paths import get_repo_root
    model_path = get_repo_root() / "model" / (args.model_name + ".h5")
    cache = args.dataset_cache or get_keras_datasets_dir() / "cifar-10-batches-py" / "test_batch"
    if not model_path.is_file() or not cache.is_file():
        raise SystemExit("Missing local model or CIFAR10 cache: {} / {}".format(model_path, cache))
    output = args.output_dir or get_repo_root() / "exp" / "forward_profile" / datetime.now().strftime("%Y%m%d_%H%M%S")
    options = {"model_path": str(model_path.resolve()), "dataset_cache": str(cache.resolve()),
               "case_index": args.case_index, "timeout": args.timeout,
               "mode": args.mode, "global_shift_kind": args.global_shift_kind,
               "setup_timeout": args.setup_timeout, "path_threshold": args.symbolic_path_threshold}
    from engine.forward_profile import run_profile_case
    summary = run_profile_case(options, output)
    print("Results: {}".format(output.resolve()))
    return int(any(result["status"] == "error" for result in summary["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
