#!/usr/bin/env python3
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from test_compare_resnet18 import parse_args, compare_logits  # noqa: E402


if __name__ == "__main__":
    args = parse_args()
    compare_logits(
        args.model_path,
        args.sample_idx,
        args.atol,
        args.input_shape,
        args.num_classes
    )
