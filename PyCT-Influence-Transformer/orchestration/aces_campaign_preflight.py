"""Offline PWL preflight for a shared scalar ACES CIFAR10 campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from libct.aces_like import (
    ACES_LIKE_EXPERIMENT_CONTROLS,
    ACES_LIKE_PWL_ERROR_METRIC,
    ACES_LIKE_PWL_VALIDATOR_VERSION,
    AcesLikeTransformError,
    build_adaptive_pwl_approximation,
    validate_control_range,
)


def preflight(images: np.ndarray, controls: Sequence[str], *, first_n: int = 100,
              x_min: float = -0.2, x_max: float = 0.2, max_segments: int = 64,
              error_tolerance: float = 1 / 255) -> Dict[str, Any]:
    """Check every requested control/case with the runtime's PWL builder.

    PWL tables depend on image/control/bounds, so models share these checks.
    A rejected case is recorded and never causes automatic budget relaxation.
    """
    if type(first_n) is not int or not 1 <= first_n <= len(images):
        raise ValueError("first_n must be within the local image count")
    if type(max_segments) is not int or max_segments < 1:
        raise ValueError("max_segments must be a positive integer")
    if not math.isfinite(error_tolerance) or error_tolerance <= 0:
        raise ValueError("error_tolerance must be finite and positive")
    if not all(math.isfinite(v) for v in (x_min, x_max)) or not x_min < x_max or not x_min <= 0 <= x_max:
        raise ValueError("ordered finite bounds must include zero")
    if not controls or len(set(controls)) != len(controls) or not set(controls) <= set(ACES_LIKE_EXPERIMENT_CONTROLS):
        raise ValueError("controls must be unique supported scalar ACES controls")
    for control in controls:
        validate_control_range(control, x_min, x_max)
    rows = []
    for control in controls:
        for index in range(first_n):
            started = time.monotonic()
            row = {"control": control, "case_id": index}
            try:
                # Match the float64 image passed by the attack builder.
                approximation = build_adaptive_pwl_approximation(
                    np.asarray(images[index], dtype=np.float64), kind=control,
                    x_min=x_min, x_max=x_max, max_segments=max_segments,
                    error_tolerance=error_tolerance,
                )
                row.update(accepted=True, segments=approximation.segment_count,
                           max_abs_error=float(approximation.max_abs_error))
                if not math.isfinite(row["max_abs_error"]) or row["max_abs_error"] > error_tolerance or row["segments"] > max_segments:
                    raise AcesLikeTransformError("PWL result violates configured limits")
            except (AcesLikeTransformError, ValueError) as exc:
                if not math.isfinite(row.get("max_abs_error", 0.0)):
                    row["max_abs_error"] = None
                row.update(accepted=False, error=str(exc))
            row["seconds"] = time.monotonic() - started
            rows.append(row)
            print("PWL {}/{}: {} ({:.2f}s)".format(control, index,
                  "accepted" if row["accepted"] else row["error"], row["seconds"]), flush=True)
    rejected = sum(not row["accepted"] for row in rows)
    return {"schema_version": "aces-campaign-preflight-v1", "complete": rejected == 0,
            "expected_count": len(controls) * first_n, "checked_count": len(rows),
            "rejected_count": rejected,
            "settings": {"controls": list(controls), "first_n": first_n, "x_min": x_min,
                         "x_max": x_max, "max_segments": max_segments,
                         "error_tolerance": error_tolerance},
            "error_metric": ACES_LIKE_PWL_ERROR_METRIC,
            "validator_version": ACES_LIKE_PWL_VALIDATOR_VERSION, "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", nargs="+", required=True)
    parser.add_argument("--first-n", type=int, default=100)
    parser.add_argument("--x-min", type=float, default=-0.2)
    parser.add_argument("--x-max", type=float, default=0.2)
    parser.add_argument("--max-segments", type=int, default=64)
    parser.add_argument("--error-tolerance", type=float, default=1 / 255)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("preflight output already exists")
    from datasets.keras_cache import get_keras_datasets_dir
    cache = args.dataset_cache or get_keras_datasets_dir() / "cifar-10-batches-py" / "test_batch"
    with cache.open("rb") as handle:
        batch = pickle.load(handle, encoding="bytes")
    pixels = np.asarray(batch[b"data"])
    if pixels.ndim != 2 or pixels.shape[1] != 3072 or pixels.dtype != np.uint8:
        raise ValueError("Invalid local CIFAR10 pixel array")
    images = pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255
    report = preflight(images, args.controls, first_n=args.first_n,
                       x_min=args.x_min, x_max=args.x_max, max_segments=args.max_segments,
                       error_tolerance=args.error_tolerance)
    report.update(dataset_cache=str(cache.resolve()),
                  dataset_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("PWL checked={} rejected={}".format(report["checked_count"], report["rejected_count"]), flush=True)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
