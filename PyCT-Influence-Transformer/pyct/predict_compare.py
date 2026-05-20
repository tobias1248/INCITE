from __future__ import annotations

import argparse
import contextlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


ATOL = 1e-5
RTOL = 1e-5
DEFAULT_LOG_DIR = Path("predict_compare_log")
DATASET_CHOICES = ("mnist", "fashion_mnist", "cifar10")
INPUT_SOURCE_CHOICES = ("dataset", "random")
LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

log = logging.getLogger("ct.predict_compare")


@dataclass(frozen=True)
class DiffMetrics:
    max_abs_diff: float
    mean_abs_diff: float
    relative_l2_diff: float
    close: bool
    shape_match: bool


@dataclass(frozen=True)
class LayerPair:
    keras_layer_index: int
    keras_layer_name: str
    keras_layer_type: str
    my_layer_index: int
    keras_layer: object


@dataclass(frozen=True)
class LayerDiff:
    keras_layer_index: int
    keras_layer_name: str
    keras_layer_type: str
    my_layer_index: int
    metrics: DiffMetrics
    keras_shape: Tuple[int, ...]
    py_shape: Tuple[int, ...]


@dataclass(frozen=True)
class CaseResult:
    idx: int
    keras_class: int
    py_class: int
    class_match: bool
    max_abs_diff: float
    mean_abs_diff: float
    relative_l2_diff: float
    close: bool
    passed: bool
    keras_output: np.ndarray
    py_output: np.ndarray
    first_layer_diff: Optional[LayerDiff] = None


@dataclass(frozen=True)
class CompareSummary:
    compared: int
    passed: int
    failed: int
    class_matches: int
    close_matches: int
    max_abs_diff: float
    worst_case_idx: Optional[int]

    @property
    def ok(self) -> bool:
        return self.failed == 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Keras .h5 predictions with the PyCT Python implementation."
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-name",
        help="Model artifact name under ./model, without the .h5 extension.",
    )
    model_group.add_argument(
        "--model-path",
        help="Explicit path to a .h5 model file.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="mnist",
        help="Dataset to use when --input-source dataset is selected (default: mnist).",
    )
    parser.add_argument(
        "--input-source",
        choices=INPUT_SOURCE_CHOICES,
        default="dataset",
        help="Input source used for comparison (default: dataset).",
    )
    parser.add_argument(
        "--first-n",
        type=_parse_positive_int,
        default=100,
        help="Number of dataset test inputs to compare (default: 100).",
    )
    parser.add_argument(
        "--samples",
        type=_parse_positive_int,
        help="Number of random inputs to compare. Defaults to --first-n.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed used when --input-source random is selected (default: 2024).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed comparison case.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level for comparison output (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        help="Optional path to write comparison logs. Only predict_compare case and summary logs are recorded.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    log_file = resolve_log_file(args)
    configure_logging(args.log_level, log_file=str(log_file))
    print(f"Writing predict comparison log: {log_file}")
    try:
        return run(args)
    except Exception as exc:
        log.error("Prediction comparison failed: %s", exc)
        return 1


def configure_logging(level: str, *, log_file: Optional[str] = None) -> None:
    console_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=console_level,
        format="%(message)s",
    )
    log.setLevel(logging.DEBUG)
    logging.getLogger("ct.model").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)
        log.addHandler(handler)


def run(args: argparse.Namespace) -> int:
    # Keep the reference Keras path on CPU unless the caller explicitly set a device.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    from modeling.keras_loader import load_model_with_compat

    model_path = resolve_model_path(args)
    keras_model = load_model_with_compat(str(model_path))
    inputs = load_inputs(args, keras_model)
    keras_predictions = np.asarray(keras_model.predict(inputs, verbose=0))

    py_model = build_python_model(model_path)
    layer_pairs = collect_layer_pairs(keras_model, py_model)
    keras_layer_outputs = predict_keras_layer_outputs(keras_model, inputs, layer_pairs)
    results = compare_predictions(
        keras_predictions,
        py_model,
        inputs,
        fail_fast=bool(args.fail_fast),
        layer_pairs=layer_pairs,
        keras_layer_outputs=keras_layer_outputs,
    )
    summary = summarize_results(results)
    log_summary(summary)
    return 0 if summary.ok else 1


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path:
        return Path(args.model_path)
    return Path("model") / f"{args.model_name}.h5"


def resolve_log_file(args: argparse.Namespace) -> Path:
    if args.log_file:
        return Path(args.log_file)
    return DEFAULT_LOG_DIR / f"{current_timestamp()}_{resolve_model_label(args)}.log"


def resolve_model_label(args: argparse.Namespace) -> str:
    if args.model_name:
        return sanitize_log_label(args.model_name)
    return sanitize_log_label(Path(args.model_path).stem)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_log_label(label: str) -> str:
    sanitized = []
    for char in label:
        if char.isalnum() or char in {"-", "_"}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_") or "model"


def load_inputs(args: argparse.Namespace, keras_model) -> np.ndarray:
    if args.input_source == "dataset":
        return load_dataset_inputs(args.dataset, args.first_n)
    if args.input_source == "random":
        samples = args.samples if args.samples is not None else args.first_n
        return generate_random_inputs(keras_model, samples=samples, seed=args.seed)
    raise ValueError(f"Unsupported input source: {args.input_source}")


def load_dataset_inputs(dataset: str, first_n: int) -> np.ndarray:
    if dataset == "mnist":
        from datasets.mnist import MnistDataset

        source = MnistDataset()
    elif dataset == "fashion_mnist":
        from datasets.fashion_mnist import FashionMnistDataset

        source = FashionMnistDataset()
    elif dataset == "cifar10":
        from datasets.cifar10 import Cifar10Dataset

        source = Cifar10Dataset()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    inputs = np.asarray(source.x_test[:first_n], dtype=np.float32)
    if len(inputs) < first_n:
        raise ValueError(
            f"Dataset '{dataset}' only has {len(inputs)} test input(s), cannot compare {first_n}."
        )
    return inputs


def generate_random_inputs(keras_model, *, samples: int, seed: int) -> np.ndarray:
    input_shape = extract_single_input_shape(keras_model)
    rng = np.random.default_rng(seed)
    return rng.random((samples, *input_shape), dtype=np.float32)


def extract_single_input_shape(keras_model) -> Tuple[int, ...]:
    shape = getattr(keras_model, "input_shape", None)
    if isinstance(shape, list):
        raise ValueError("Random input generation does not support multiple model inputs.")
    if shape is None:
        raise ValueError("Cannot infer model input shape for random input generation.")

    dims = tuple(shape)
    if dims and dims[0] is None:
        dims = dims[1:]
    if not dims or any(dim is None for dim in dims):
        raise ValueError(f"Random input generation requires a fully-defined input shape, got {shape}.")
    return tuple(int(dim) for dim in dims)


def build_python_model(model_path: Path):
    from engine import predictor_runtime

    with suppress_runtime_output():
        predictor_runtime.init_model(str(model_path))
    if predictor_runtime.myModel is None:
        raise RuntimeError("PyCT predictor runtime did not initialize a model.")
    return predictor_runtime.myModel


def collect_layer_pairs(keras_model, py_model) -> list[LayerPair]:
    keras_layers = collect_comparable_keras_layers(keras_model)
    py_layers = getattr(py_model, "layers", [])
    if not keras_layers or not py_layers:
        return []

    from libct.position import my_layer_number_to_Keras_layer_number

    last_my_layer_by_keras: dict[int, int] = {}
    for my_layer_index in range(len(py_layers)):
        keras_layer_index = my_layer_number_to_Keras_layer_number.get(my_layer_index)
        if keras_layer_index is None:
            continue
        last_my_layer_by_keras[int(keras_layer_index)] = my_layer_index

    pairs = []
    for keras_layer_index, my_layer_index in sorted(last_my_layer_by_keras.items()):
        if keras_layer_index < 0 or keras_layer_index >= len(keras_layers):
            continue
        layer = keras_layers[keras_layer_index]
        pairs.append(
            LayerPair(
                keras_layer_index=keras_layer_index,
                keras_layer_name=str(getattr(layer, "name", f"layer_{keras_layer_index}")),
                keras_layer_type=type(layer).__name__,
                my_layer_index=my_layer_index,
                keras_layer=layer,
            )
        )
    return pairs


def collect_comparable_keras_layers(keras_model) -> list[object]:
    excluded = {"Dropout", "InputLayer", "Embedding"}
    return [
        layer
        for layer in getattr(keras_model, "layers", []) or []
        if type(layer).__name__ not in excluded
    ]


def predict_keras_layer_outputs(
    keras_model,
    inputs: np.ndarray,
    layer_pairs: Sequence[LayerPair],
) -> list[np.ndarray]:
    if not layer_pairs:
        return []

    from keras import Model

    probe_model = Model(
        inputs=keras_model.inputs,
        outputs=[pair.keras_layer.output for pair in layer_pairs],
    )
    outputs = probe_model.predict(inputs, verbose=0)
    if len(layer_pairs) == 1:
        outputs = [outputs]
    return [np.asarray(output) for output in outputs]


def compare_predictions(
    keras_predictions: np.ndarray,
    py_model,
    inputs: np.ndarray,
    *,
    fail_fast: bool = False,
    layer_pairs: Sequence[LayerPair] = (),
    keras_layer_outputs: Sequence[np.ndarray] = (),
) -> list[CaseResult]:
    if len(keras_predictions) != len(inputs):
        raise ValueError(
            f"Keras prediction count {len(keras_predictions)} does not match input count {len(inputs)}."
        )

    results: list[CaseResult] = []
    for idx, input_tensor in enumerate(inputs):
        with suppress_runtime_output():
            py_output = py_model.forward(input_tensor.tolist())
        first_layer_diff = find_first_layer_diff(
            idx,
            py_model,
            layer_pairs=layer_pairs,
            keras_layer_outputs=keras_layer_outputs,
        )
        result = compare_case(idx, keras_predictions[idx], py_output, first_layer_diff)
        results.append(result)
        log_case(result)
        if fail_fast and not result.passed:
            break
    return results


def compare_case(
    idx: int,
    keras_output,
    py_output,
    first_layer_diff: Optional[LayerDiff] = None,
) -> CaseResult:
    keras_values = normalize_output(keras_output)
    py_values = normalize_output(py_output)
    keras_class = predict_class(keras_values)
    py_class = predict_class(py_values)
    class_match = keras_class == py_class
    metrics = calculate_diff_metrics(keras_values, py_values)
    passed = metrics.close and class_match
    return CaseResult(
        idx=idx,
        keras_class=keras_class,
        py_class=py_class,
        class_match=class_match,
        max_abs_diff=metrics.max_abs_diff,
        mean_abs_diff=metrics.mean_abs_diff,
        relative_l2_diff=metrics.relative_l2_diff,
        close=metrics.close,
        passed=passed,
        keras_output=keras_values,
        py_output=py_values,
        first_layer_diff=first_layer_diff,
    )


def normalize_output(output) -> np.ndarray:
    values = np.asarray(output, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Prediction output is empty.")
    return values


def predict_class(output: np.ndarray) -> int:
    if output.size == 1:
        return 1 if float(output[0]) > 0.5 else 0
    return int(np.argmax(output))


def outputs_close(keras_output: np.ndarray, py_output: np.ndarray) -> bool:
    return calculate_diff_metrics(keras_output, py_output).close


def calculate_max_abs_diff(keras_output: np.ndarray, py_output: np.ndarray) -> float:
    return calculate_diff_metrics(keras_output, py_output).max_abs_diff


def calculate_diff_metrics(keras_output: np.ndarray, py_output: np.ndarray) -> DiffMetrics:
    if keras_output.shape != py_output.shape:
        return DiffMetrics(
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            relative_l2_diff=float("inf"),
            close=False,
            shape_match=False,
        )
    diff = keras_output - py_output
    abs_diff = np.abs(diff)
    keras_norm = float(np.linalg.norm(keras_output))
    relative_denominator = max(keras_norm, 1e-12)
    return DiffMetrics(
        max_abs_diff=float(np.max(abs_diff)),
        mean_abs_diff=float(np.mean(abs_diff)),
        relative_l2_diff=float(np.linalg.norm(diff) / relative_denominator),
        close=bool(np.allclose(keras_output, py_output, atol=ATOL, rtol=RTOL)),
        shape_match=True,
    )


def find_first_layer_diff(
    idx: int,
    py_model,
    *,
    layer_pairs: Sequence[LayerPair],
    keras_layer_outputs: Sequence[np.ndarray],
) -> Optional[LayerDiff]:
    if not layer_pairs or not keras_layer_outputs:
        return None
    for output_index, pair in enumerate(layer_pairs):
        if output_index >= len(keras_layer_outputs):
            break
        keras_batch_output = np.asarray(keras_layer_outputs[output_index])
        if idx >= len(keras_batch_output):
            continue
        keras_output = normalize_output(keras_batch_output[idx])
        py_output = normalize_output(py_model.getLayOutput(pair.my_layer_index))
        metrics = calculate_diff_metrics(keras_output, py_output)
        if not metrics.close:
            return LayerDiff(
                keras_layer_index=pair.keras_layer_index,
                keras_layer_name=pair.keras_layer_name,
                keras_layer_type=pair.keras_layer_type,
                my_layer_index=pair.my_layer_index,
                metrics=metrics,
                keras_shape=tuple(np.asarray(keras_batch_output[idx]).shape),
                py_shape=tuple(np.asarray(py_model.getLayOutput(pair.my_layer_index)).shape),
            )
    return None


def log_case(result: CaseResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    log.info(
        (
            "[%s] idx=%s keras_class=%s py_class=%s class_match=%s "
            "max_abs_diff=%.6f mean_abs_diff=%.6f rel_l2_diff=%.6f close=%s"
        ),
        status,
        result.idx,
        result.keras_class,
        result.py_class,
        str(result.class_match).lower(),
        result.max_abs_diff,
        result.mean_abs_diff,
        result.relative_l2_diff,
        str(result.close).lower(),
    )
    if not result.passed:
        log.info("       keras_output=%s", np.array2string(result.keras_output, precision=6))
        log.info("       py_output=%s", np.array2string(result.py_output, precision=6))
        if result.first_layer_diff is not None:
            log_layer_diff(result.first_layer_diff)


def log_layer_diff(layer_diff: LayerDiff) -> None:
    log.info(
        (
            "       first_layer_diff keras_layer=%s name=%s type=%s my_layer=%s "
            "max_abs_diff=%.6f mean_abs_diff=%.6f rel_l2_diff=%.6f "
            "shape_match=%s keras_shape=%s py_shape=%s"
        ),
        layer_diff.keras_layer_index,
        layer_diff.keras_layer_name,
        layer_diff.keras_layer_type,
        layer_diff.my_layer_index,
        layer_diff.metrics.max_abs_diff,
        layer_diff.metrics.mean_abs_diff,
        layer_diff.metrics.relative_l2_diff,
        str(layer_diff.metrics.shape_match).lower(),
        layer_diff.keras_shape,
        layer_diff.py_shape,
    )


def summarize_results(results: Sequence[CaseResult]) -> CompareSummary:
    if not results:
        return CompareSummary(
            compared=0,
            passed=0,
            failed=0,
            class_matches=0,
            close_matches=0,
            max_abs_diff=0.0,
            worst_case_idx=None,
        )

    passed = sum(1 for result in results if result.passed)
    class_matches = sum(1 for result in results if result.keras_class == result.py_class)
    close_matches = sum(1 for result in results if result.close)
    finite_results = [result for result in results if np.isfinite(result.max_abs_diff)]
    if finite_results:
        worst = max(finite_results, key=lambda result: result.max_abs_diff)
    else:
        worst = max(results, key=lambda result: result.max_abs_diff)
    return CompareSummary(
        compared=len(results),
        passed=passed,
        failed=len(results) - passed,
        class_matches=class_matches,
        close_matches=close_matches,
        max_abs_diff=worst.max_abs_diff,
        worst_case_idx=worst.idx,
    )


def log_summary(summary: CompareSummary) -> None:
    log.info("Compared inputs: %s", summary.compared)
    log.info("Passed cases: %s", summary.passed)
    log.info("Failed cases: %s", summary.failed)
    log.info("Class matches: %s/%s", summary.class_matches, summary.compared)
    log.info("Output-close matches: %s/%s", summary.close_matches, summary.compared)
    log.info("Max abs diff: %.6f", summary.max_abs_diff)
    log.info("Worst case idx: %s", summary.worst_case_idx)
    log.info("Result: %s", "PASS" if summary.ok else "FAIL")


def _parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1.")
    return parsed


@contextlib.contextmanager
def suppress_runtime_output():
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            yield


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATOL",
    "RTOL",
    "CaseResult",
    "CompareSummary",
    "DiffMetrics",
    "LayerDiff",
    "LayerPair",
    "DEFAULT_LOG_DIR",
    "calculate_max_abs_diff",
    "calculate_diff_metrics",
    "collect_comparable_keras_layers",
    "collect_layer_pairs",
    "compare_case",
    "compare_predictions",
    "extract_single_input_shape",
    "find_first_layer_diff",
    "generate_random_inputs",
    "load_dataset_inputs",
    "main",
    "normalize_output",
    "parse_args",
    "predict_class",
    "resolve_log_file",
    "resolve_model_label",
    "resolve_model_path",
    "summarize_results",
]
