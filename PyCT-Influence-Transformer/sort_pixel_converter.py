from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for SHAP-to-coordinate conversion workflow."""
    parser = argparse.ArgumentParser(
        description="Convert SHAP value tensors to sorted pixel coordinate arrays."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the SHAP values .npy file (shape: samples x H x W x C x classes).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination .npy for sorted coordinates (shape: samples x K x dims).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Optional number of coordinates to retain per sample (defaults to all pixels).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def sort_pixel_coordinates(shap_values: np.ndarray, *, topk: int | None = None) -> np.ndarray:
    """Return per-sample pixel coordinates sorted by descending SHAP magnitude."""
    arr = np.asarray(shap_values, dtype=np.float64)
    if arr.ndim < 3:
        raise ValueError(
            f"Expected SHAP tensor with at least 3 dims (batch, spatial..., classes); got {arr.shape}."
        )

    if arr.shape[-1] == 0:
        raise ValueError("Final axis with class contributions must be non-empty.")

    scores = np.abs(arr).sum(axis=-1)
    spatial_shape = scores.shape[1:]
    if not spatial_shape:
        raise ValueError("SHAP tensor must include spatial dimensions beyond the batch axis.")

    axes = [np.arange(size, dtype=np.int64) for size in spatial_shape]
    coordinate_grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(spatial_shape)
    )

    per_sample = coordinate_grid.shape[0]
    limit = _normalize_topk(topk, per_sample)

    flattened_scores = scores.reshape(scores.shape[0], per_sample)
    sorted_coords = np.empty((scores.shape[0], limit, coordinate_grid.shape[1]), dtype=np.int64)

    for sample_idx in range(scores.shape[0]):
        order = np.argsort(-flattened_scores[sample_idx], kind="mergesort")[:limit]
        sorted_coords[sample_idx] = coordinate_grid[order]

    return sorted_coords


def convert_shap_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    topk: int | None = None,
) -> Path:
    """Load SHAP values from disk, convert them, and save the sorted coordinates."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    shap_values = np.load(in_path)
    sorted_coords = sort_pixel_coordinates(shap_values, topk=topk)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, sorted_coords.astype(np.int64, copy=False))
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point hooking the conversion helpers up to argparse."""
    args = parse_args(argv)
    convert_shap_file(args.input, args.output, topk=args.topk)
    print(f"Saved sorted coordinates to {args.output}")
    return 0


def _normalize_topk(topk: int | None, per_sample: int) -> int:
    """Validate the requested top-k bound and clamp it to available pixels."""
    if topk is None:
        return per_sample
    if topk <= 0:
        raise ValueError("--topk must be a positive integer.")
    return min(topk, per_sample)


if __name__ == "__main__":  # pragma: no cover - CLI hook
    raise SystemExit(main())
