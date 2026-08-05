from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from datasets.common import select_background_per_class
from explainability.input_shap_sign import (
    BOUNDS_MODE_CLIP,
    BOUNDS_MODE_STRICT,
    BOUNDS_MODES,
    TargetClassInputShapProvider,
    build_sign_mask,
    count_clipped_values,
    derive_valid_shift_interval,
    materialize_shifted_input,
)
from explainability.shap_contract import DEFAULT_TARGET_CLASS_SHAP_ROOT


def _parse_case_indices(value: str) -> Tuple[int, ...]:
    try:
        indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case indices must be comma-separated integers") from exc
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("case indices must be non-negative")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("case indices must not contain duplicates")
    return indices


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one shared real-valued shift X where each input element moves "
            "according to the sign of its original-class SHAP attribution."
        )
    )
    parser.add_argument("--dataset", choices=("cifar10",), default="cifar10")
    parser.add_argument("--model-name", default="cifar10_concolic_transformer")
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--first-n", type=int, default=1)
    parser.add_argument("--case-indices", type=_parse_case_indices)
    parser.add_argument("--shift-min", type=float, default=-0.1)
    parser.add_argument("--shift-max", type=float, default=0.1)
    parser.add_argument("--shift-step", type=float, default=0.001)
    parser.add_argument(
        "--bounds-mode",
        choices=BOUNDS_MODES,
        default=BOUNDS_MODE_CLIP,
        help=(
            "clip keeps the requested X interval and clips each shifted value to [0,1]; "
            "strict shrinks the shared interval so clipping is unnecessary."
        ),
    )
    parser.add_argument("--shap-sign-epsilon", type=float, default=0.0)
    parser.add_argument("--background-per-class", type=int, default=3)
    parser.add_argument("--background-seed", type=int, default=2233)
    parser.add_argument("--shap-output-root", default=DEFAULT_TARGET_CLASS_SHAP_ROOT)
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force-refresh-shap", action="store_true")
    return parser.parse_args(argv)


def build_shift_grid(lower: float, upper: float, step: float) -> np.ndarray:
    if not all(math.isfinite(value) for value in (lower, upper, step)):
        raise ValueError("shift grid values must be finite")
    if lower > upper:
        raise ValueError("lower must be <= upper")
    if step <= 0:
        raise ValueError("step must be > 0")
    count = int(math.floor((upper - lower) / step + 1e-12))
    values = [lower + ordinal * step for ordinal in range(count + 1)]
    if not values or values[-1] < upper - 1e-12:
        values.append(upper)
    else:
        values[-1] = min(values[-1], upper)
    if lower <= 0.0 <= upper:
        values.append(0.0)
    deduplicated = sorted(set(round(value, 15) for value in values))
    return np.asarray(deduplicated, dtype=np.float64)


def _prediction_matrix(model: Any, inputs: np.ndarray, *, batch_size: int) -> np.ndarray:
    predictions = np.asarray(model.predict(inputs, batch_size=batch_size, verbose=0))
    if predictions.ndim != 2 or predictions.shape[0] != inputs.shape[0]:
        raise ValueError(
            "model must return one multiclass row per input; "
            f"got predictions={predictions.shape}, inputs={inputs.shape}"
        )
    if predictions.shape[1] < 2:
        raise ValueError("SHAP sign sweep requires a multiclass model")
    if not np.isfinite(predictions).all():
        raise ValueError("model predictions contain NaN or Inf")
    return predictions


def _class_margin(row: np.ndarray, original_class: int) -> float:
    competitors = np.delete(row, original_class)
    return float(row[original_class] - np.max(competitors))


def evaluate_shift_direction(
    model: Any,
    sample: np.ndarray,
    sign_mask: np.ndarray,
    *,
    original_class: int,
    lower: float,
    upper: float,
    step: float,
    batch_size: int,
    bounds_mode: str,
) -> Dict[str, Any]:
    if bounds_mode not in BOUNDS_MODES:
        raise ValueError(f"unsupported bounds mode: {bounds_mode}")
    shifts = build_shift_grid(lower, upper, step)
    shifted_inputs = np.stack(
        [
            materialize_shifted_input(
                sample,
                sign_mask,
                float(shift),
                bounds_mode=bounds_mode,
            )
            for shift in shifts
        ]
    )
    clipped_counts = [
        count_clipped_values(sample, sign_mask, float(shift)) for shift in shifts
    ]
    predictions = _prediction_matrix(model, shifted_inputs, batch_size=batch_size)
    if original_class < 0 or original_class >= predictions.shape[1]:
        raise ValueError(f"original_class {original_class} is outside model output")
    labels = np.argmax(predictions, axis=1).astype(int)
    zero_indices = np.flatnonzero(np.isclose(shifts, 0.0, atol=1e-15, rtol=0.0))
    if len(zero_indices) != 1 or labels[int(zero_indices[0])] != original_class:
        raise ValueError("zero shift does not reproduce the original model label")
    changed = [index for index, label in enumerate(labels) if label != original_class]
    best_index = (
        min(changed, key=lambda index: (abs(shifts[index]), shifts[index]))
        if changed
        else None
    )
    curve = [
        {
            "shift": float(shift),
            "label": int(labels[index]),
            "original_class_score": float(predictions[index, original_class]),
            "margin": _class_margin(predictions[index], original_class),
            "clipped_count": int(clipped_counts[index]),
        }
        for index, shift in enumerate(shifts)
    ]
    best = None
    if best_index is not None:
        best = {
            **curve[best_index],
            "abs_shift": float(abs(shifts[best_index])),
        }
    return {
        "bounds_mode": bounds_mode,
        "grid_count": len(curve),
        "changed_count": len(changed),
        "successful": bool(changed),
        "best": best,
        "curve": curve,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_indices(args: argparse.Namespace) -> Tuple[int, ...]:
    if args.case_indices is not None:
        return tuple(args.case_indices)
    if args.first_n < 1:
        raise ValueError("--first-n must be >= 1")
    return tuple(range(args.first_n))


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.shift_min) or not math.isfinite(args.shift_max):
        raise ValueError("--shift-min and --shift-max must be finite")
    if args.shift_min > args.shift_max:
        raise ValueError("--shift-min must be <= --shift-max")
    if not math.isfinite(args.shift_step) or args.shift_step <= 0:
        raise ValueError("--shift-step must be finite and > 0")
    if args.bounds_mode not in BOUNDS_MODES:
        raise ValueError(f"--bounds-mode must be one of {', '.join(BOUNDS_MODES)}")
    if not math.isfinite(args.shap_sign_epsilon) or args.shap_sign_epsilon < 0:
        raise ValueError("--shap-sign-epsilon must be finite and >= 0")
    if args.background_per_class < 1:
        raise ValueError("--background-per-class must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")


def run_sweep(
    args: argparse.Namespace,
    *,
    dataset: Optional[Any] = None,
    model: Optional[Any] = None,
    provider: Optional[Any] = None,
) -> Dict[str, Any]:
    _validate_args(args)
    indices = _resolve_indices(args)

    if dataset is None:
        from datasets.cifar10 import Cifar10Dataset

        dataset = Cifar10Dataset()
    model_path = Path(args.model_root) / f"{args.model_name}.h5"
    if model is None:
        from modeling.keras_loader import load_model_with_compat

        model = load_model_with_compat(str(model_path))
    if provider is None:
        provider = TargetClassInputShapProvider(
            model_path=model_path,
            output_root=Path(args.shap_output_root),
        )

    background = select_background_per_class(
        dataset.x_test,
        dataset.y_test,
        per_class=args.background_per_class,
        seed=args.background_seed,
    )
    output_dir = Path(
        args.output_dir
        or f"exp/{args.model_name}_shap_sign_sweep_{args.bounds_mode}"
    )
    case_results: List[Dict[str, Any]] = []

    for case_index in indices:
        if case_index >= len(dataset.x_test):
            raise IndexError(f"case index {case_index} is outside the dataset")
        sample = np.asarray(dataset.x_test[case_index], dtype=np.float32)
        original_predictions = _prediction_matrix(
            model,
            sample[np.newaxis, ...],
            batch_size=args.batch_size,
        )[0]
        original_class = int(np.argmax(original_predictions))
        artifact = provider.ensure(
            case_index=case_index,
            sample=sample,
            background=background,
            target_class=original_class,
            force_refresh=args.force_refresh_shap,
        )
        sign_mask = build_sign_mask(
            artifact.values,
            epsilon=args.shap_sign_epsilon,
        )
        if args.bounds_mode == BOUNDS_MODE_STRICT:
            effective_min, effective_max = derive_valid_shift_interval(
                sample,
                sign_mask,
                requested_min=args.shift_min,
                requested_max=args.shift_max,
            )
        else:
            effective_min, effective_max = float(args.shift_min), float(args.shift_max)
        evaluation = evaluate_shift_direction(
            model,
            sample,
            sign_mask,
            original_class=original_class,
            lower=effective_min,
            upper=effective_max,
            step=args.shift_step,
            batch_size=args.batch_size,
            bounds_mode=args.bounds_mode,
        )
        labels = np.asarray(dataset.y_test).reshape(-1)
        result = {
            "case_index": int(case_index),
            "ground_truth": int(labels[case_index]),
            "original_class": original_class,
            "original_scores": [float(value) for value in original_predictions],
            "target_class": int(artifact.target_class),
            "shap_attribution_target": artifact.metadata.get("attribution_target"),
            "shap_cache": str(artifact.cache_path),
            "shap_was_cached": bool(artifact.was_cached),
            "shap_sign_epsilon": float(args.shap_sign_epsilon),
            "sign_counts": {
                "positive": int(np.count_nonzero(sign_mask == 1)),
                "negative": int(np.count_nonzero(sign_mask == -1)),
                "zero": int(np.count_nonzero(sign_mask == 0)),
            },
            "bounds_mode": args.bounds_mode,
            "requested_interval": [float(args.shift_min), float(args.shift_max)],
            "effective_interval": [effective_min, effective_max],
            "evaluation": evaluation,
        }
        _write_json(output_dir / f"case_{case_index}.json", result)
        case_results.append(result)

    successful = [result for result in case_results if result["evaluation"]["successful"]]
    best_shifts = [result["evaluation"]["best"]["abs_shift"] for result in successful]
    zero_only_count = sum(
        abs(result["effective_interval"][0]) <= 1e-15
        and abs(result["effective_interval"][1]) <= 1e-15
        for result in case_results
    )
    clipped_success_count = sum(
        result["evaluation"]["best"]["clipped_count"] > 0 for result in successful
    )
    summary = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "bounds_mode": args.bounds_mode,
        "case_count": len(case_results),
        "successful_count": len(successful),
        "success_rate": len(successful) / len(case_results) if case_results else 0.0,
        "median_min_abs_shift": float(np.median(best_shifts)) if best_shifts else None,
        "zero_only_interval_count": int(zero_only_count),
        "clipped_success_count": int(clipped_success_count),
        "requested_interval": [float(args.shift_min), float(args.shift_max)],
        "shift_step": float(args.shift_step),
        "shap_sign_epsilon": float(args.shap_sign_epsilon),
        "shap_output_root": str(args.shap_output_root),
        "background_per_class": int(args.background_per_class),
        "background_seed": int(args.background_seed),
        "output_dir": str(output_dir),
        "cases": [
            {
                "case_index": result["case_index"],
                "successful": result["evaluation"]["successful"],
                "best": result["evaluation"]["best"],
                "effective_interval": result["effective_interval"],
            }
            for result in case_results
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_sweep(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_shift_grid",
    "evaluate_shift_direction",
    "main",
    "parse_args",
    "run_sweep",
]
