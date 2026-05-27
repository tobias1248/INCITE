from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ReplayCase:
    case_name: str
    idx: int
    original_label: int
    stored_attack_label: int
    original_prediction: int
    adversarial_prediction: int
    original_matches_label: bool
    still_adversarial: bool
    stored_attack_label_matches_prediction: bool
    adv_input_path: str


def _iter_case_dirs(experiment_root: Path) -> Iterable[Path]:
    for path in sorted(experiment_root.iterdir()):
        if path.is_dir() and path.name.startswith("case_"):
            yield path


def _load_stats(case_dir: Path) -> Dict[str, Any]:
    stats_path = case_dir / "stats.json"
    with stats_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {stats_path}.")
    return payload


def _resolve_model_path(experiment_root: Path, explicit_model_path: Optional[str]) -> Path:
    if explicit_model_path:
        return Path(explicit_model_path)

    first_case = next(iter(_iter_case_dirs(experiment_root)), None)
    if first_case is None:
        raise ValueError(f"No case directories found under {experiment_root}.")

    payload = _load_stats(first_case)
    meta = payload.get("meta") or {}
    model_name = meta.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(
            "Unable to infer model path from stats.json meta.model_name. "
            "Please pass --model-path explicitly."
        )
    return Path("model") / f"{model_name}.h5"


def _prediction_to_labels(predictions: np.ndarray) -> List[int]:
    if predictions.ndim == 1:
        return [int(score > 0.5) for score in predictions.tolist()]
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        return [int(score[0] > 0.5) for score in predictions.tolist()]
    return [int(np.argmax(row)) for row in predictions]


def _predict_labels(model: Any, batch: np.ndarray) -> List[int]:
    outputs = model(batch, training=False)
    if hasattr(outputs, "numpy"):
        outputs = outputs.numpy()
    predictions = np.asarray(outputs)
    return _prediction_to_labels(predictions)


def _batch_predict_labels(model: Any, arrays: Sequence[np.ndarray], batch_size: int) -> List[int]:
    labels: List[int] = []
    for start in range(0, len(arrays), batch_size):
        batch = np.stack(arrays[start : start + batch_size], axis=0)
        labels.extend(_predict_labels(model, batch))
    return labels


def _load_model(model_path: Path) -> Any:
    from modeling.keras_loader import load_model_with_compat

    return load_model_with_compat(str(model_path))


def collect_replay_cases(experiment_root: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    selected: List[Tuple[Path, Dict[str, Any]]] = []
    for case_dir in _iter_case_dirs(experiment_root):
        payload = _load_stats(case_dir)
        meta = payload.get("meta") or {}
        attack_label = meta.get("attack_label")
        original_label = meta.get("original_label")
        if attack_label is None:
            continue
        if not isinstance(original_label, int) or not isinstance(attack_label, int):
            raise ValueError(
                f"{case_dir / 'stats.json'} must contain integer meta.original_label "
                "and meta.attack_label for successful adversarial cases."
            )
        adv_input_path = case_dir / "adv_input.npy"
        ori_input_path = case_dir / "ori_input.npy"
        if not adv_input_path.is_file():
            raise FileNotFoundError(f"Missing adversarial input: {adv_input_path}")
        if not ori_input_path.is_file():
            raise FileNotFoundError(f"Missing original input: {ori_input_path}")
        selected.append((case_dir, payload))
    return selected


def replay_adversarial_cases(
    experiment_root: Path,
    *,
    model_path: Optional[str] = None,
    batch_size: int = 32,
) -> Tuple[Path, List[ReplayCase]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    resolved_model_path = _resolve_model_path(experiment_root, model_path)
    if not resolved_model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {resolved_model_path}")

    selected = collect_replay_cases(experiment_root)
    model = _load_model(resolved_model_path)

    original_arrays: List[np.ndarray] = []
    adversarial_arrays: List[np.ndarray] = []
    case_dirs: List[Path] = []
    payloads: List[Dict[str, Any]] = []

    for case_dir, payload in selected:
        case_dirs.append(case_dir)
        payloads.append(payload)
        original_arrays.append(np.load(case_dir / "ori_input.npy").astype(np.float32, copy=False))
        adversarial_arrays.append(np.load(case_dir / "adv_input.npy").astype(np.float32, copy=False))

    original_predictions = _batch_predict_labels(model, original_arrays, batch_size)
    adversarial_predictions = _batch_predict_labels(model, adversarial_arrays, batch_size)

    records: List[ReplayCase] = []
    for case_dir, payload, original_prediction, adversarial_prediction in zip(
        case_dirs,
        payloads,
        original_predictions,
        adversarial_predictions,
    ):
        meta = payload["meta"]
        original_label = int(meta["original_label"])
        stored_attack_label = int(meta["attack_label"])
        records.append(
            ReplayCase(
                case_name=case_dir.name,
                idx=int(meta.get("idx", case_dir.name.removeprefix("case_"))),
                original_label=original_label,
                stored_attack_label=stored_attack_label,
                original_prediction=original_prediction,
                adversarial_prediction=adversarial_prediction,
                original_matches_label=original_prediction == original_label,
                still_adversarial=adversarial_prediction != original_label,
                stored_attack_label_matches_prediction=adversarial_prediction == stored_attack_label,
                adv_input_path=str(case_dir / "adv_input.npy"),
            )
        )
    return resolved_model_path, records


def _build_summary(model_path: Path, records: Sequence[ReplayCase]) -> Dict[str, Any]:
    total = len(records)
    still_adversarial = sum(1 for record in records if record.still_adversarial)
    original_mismatch = sum(1 for record in records if not record.original_matches_label)
    stored_attack_match = sum(
        1 for record in records if record.stored_attack_label_matches_prediction
    )
    not_adversarial_cases = [
        asdict(record) for record in records if not record.still_adversarial
    ]
    original_mismatch_cases = [
        asdict(record) for record in records if not record.original_matches_label
    ]
    stored_attack_mismatch_cases = [
        asdict(record)
        for record in records
        if not record.stored_attack_label_matches_prediction
    ]
    return {
        "model_path": str(model_path),
        "total_replayed_cases": total,
        "still_adversarial_count": still_adversarial,
        "still_adversarial_rate": (still_adversarial / total) if total else 0.0,
        "original_prediction_match_count": total - original_mismatch,
        "original_prediction_mismatch_count": original_mismatch,
        "stored_attack_label_match_count": stored_attack_match,
        "stored_attack_label_mismatch_count": total - stored_attack_match,
        "not_adversarial_cases": not_adversarial_cases,
        "original_prediction_mismatch_cases": original_mismatch_cases,
        "stored_attack_label_mismatch_cases": stored_attack_mismatch_cases,
        "cases": [asdict(record) for record in records],
    }


def _print_text_summary(experiment_root: Path, summary: Dict[str, Any]) -> None:
    total = int(summary["total_replayed_cases"])
    still_adv = int(summary["still_adversarial_count"])
    original_match = int(summary["original_prediction_match_count"])
    stored_attack_match = int(summary["stored_attack_label_match_count"])

    print(f"Experiment: {experiment_root}")
    print(f"Model: {summary['model_path']}")
    print(f"Replayed adversarial cases: {total}")
    print(f"Still adversarial: {still_adv}/{total}")
    print(f"Original prediction matches stats label: {original_match}/{total}")
    print(f"Stored attack label matches replay prediction: {stored_attack_match}/{total}")

    not_adv = summary["not_adversarial_cases"]
    if not_adv:
        print("")
        print("Cases that are no longer adversarial:")
        for case in not_adv:
            print(
                f"  - {case['case_name']}: original={case['original_label']} "
                f"adv_pred={case['adversarial_prediction']} stored_attack={case['stored_attack_label']}"
            )

    original_mismatch = summary["original_prediction_mismatch_cases"]
    if original_mismatch:
        print("")
        print("Cases whose original input no longer predicts the stored original label:")
        for case in original_mismatch:
            print(
                f"  - {case['case_name']}: stats_original={case['original_label']} "
                f"replay_original_pred={case['original_prediction']}"
            )

    stored_attack_mismatch = summary["stored_attack_label_mismatch_cases"]
    if stored_attack_mismatch:
        print("")
        print("Cases whose adversarial replay prediction differs from stored attack_label:")
        for case in stored_attack_mismatch:
            print(
                f"  - {case['case_name']}: stored_attack={case['stored_attack_label']} "
                f"replay_adv_pred={case['adversarial_prediction']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay saved adversarial samples from an experiment directory against the "
            "corresponding Keras .h5 model and verify whether they are still adversarial."
        )
    )
    parser.add_argument(
        "--experiment-root",
        required=True,
        help="Experiment directory containing case_*/stats.json and adv_input.npy files.",
    )
    parser.add_argument(
        "--model-path",
        help=(
            "Optional explicit .h5 model path. Defaults to model/<meta.model_name>.h5 "
            "based on the first case stats.json."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for Keras inference (default: 32).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the text summary.",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help=(
            "Return exit code 1 if any replayed sample is no longer adversarial or if the "
            "original input no longer matches the stored original_label."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    experiment_root = Path(args.experiment_root)
    model_path, records = replay_adversarial_cases(
        experiment_root,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    summary = _build_summary(model_path, records)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_text_summary(experiment_root, summary)

    has_not_adversarial = bool(summary["not_adversarial_cases"])
    has_original_mismatch = bool(summary["original_prediction_mismatch_cases"])
    if args.fail_on_issues and (has_not_adversarial or has_original_mismatch):
        return 1
    return 0


__all__ = [
    "ReplayCase",
    "build_parser",
    "collect_replay_cases",
    "main",
    "replay_adversarial_cases",
]
