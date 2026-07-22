from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

from pyct.de.artifacts import (
    begin_de_artifact,
    mark_de_artifact_failed,
    sha256_file,
    write_de_artifact,
    write_generation_shard,
)
from pyct.de.audit import audit_case, paired_bootstrap_lift, write_json_atomic
from pyct.de.optimizer import run_de_scout
from pyct.de.types import DeConfig


log = logging.getLogger("ct.de")

DEFAULT_GATE_MODELS = (
    "cifar10_cctlike_single_mha",
    "cifar10_concolic_transformer",
    "cifar10_cctlike_eight_mha",
    "resnet18_cifar10_clean",
    "vgg16_cifar10_clean",
)


def _case_indices(value: str) -> Tuple[int, ...]:
    try:
        indices = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case indices must be comma-separated integers") from exc
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("case indices must contain non-negative integers")
    return indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and analyze PyCT Differential Evolution scouts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scout = subparsers.add_parser("scout", help="Run vectorized one-pixel DE and save generation traces.")
    scout.add_argument("--model-name", required=True)
    scout.add_argument("--dataset", choices=("cifar10",), default="cifar10")
    cases = scout.add_mutually_exclusive_group()
    cases.add_argument("--first-n", type=int, default=100)
    cases.add_argument("--case-indices", type=_case_indices)
    scout.add_argument("--maxiter", type=int, default=75)
    scout.add_argument("--population-size", type=int, default=400)
    scout.add_argument("--seed", type=int, default=2024)
    scout.add_argument(
        "--encoding", choices=("mixed-integer", "legacy-truncate"), default="mixed-integer"
    )
    scout.add_argument(
        "--objective", choices=("margin", "original-confidence"), default="margin"
    )
    scout.add_argument("--case-timeout", type=float)
    scout.add_argument("--output-root", type=Path, default=Path("exp/de_guidance"))
    scout.add_argument("--force", action="store_true")
    scout.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")

    audit = subparsers.add_parser("audit", help="Replay DE pairs and build branch-guidance evidence.")
    audit.add_argument("--model-name", required=True)
    audit.add_argument("--dataset", choices=("cifar10",), default="cifar10")
    audit_cases = audit.add_mutually_exclusive_group()
    audit_cases.add_argument("--first-n", type=int, default=20)
    audit_cases.add_argument("--case-indices", type=_case_indices)
    audit.add_argument("--trace-root", type=Path, default=Path("exp/de_guidance"))
    audit.add_argument("--output-root", type=Path, default=Path("exp/de_guidance_audit"))
    audit.add_argument("--train-pairs", type=int, default=12)
    audit.add_argument("--holdout-pairs", type=int, default=6)
    audit.add_argument("--train-end-generation", type=int, default=60)
    audit.add_argument("--holdout-end-generation", type=int, default=75)
    audit.add_argument("--shrinkage", type=float, default=2.0)
    audit.add_argument("--replay-timeout", type=int, default=120)
    audit.add_argument("--force", action="store_true")
    audit.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")

    gate = subparsers.add_parser("gate", help="Combine model audits and evaluate the go/no-go gate.")
    gate.add_argument("--audit-root", type=Path, default=Path("exp/de_guidance_audit"))
    gate.add_argument("--models", nargs="+", default=DEFAULT_GATE_MODELS)
    gate.add_argument("--bootstrap-samples", type=int, default=10_000)
    gate.add_argument("--seed", type=int, default=2024)
    gate.add_argument("--output", type=Path)
    gate.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def _run_scout(args: argparse.Namespace) -> int:
    if args.first_n is not None and args.first_n < 1:
        raise ValueError("--first-n must be >= 1")
    from datasets.cifar10 import Cifar10Dataset
    from modeling.keras_loader import load_model_with_compat

    model_path = Path("model") / f"{args.model_name}.h5"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    dataset = Cifar10Dataset()
    model = load_model_with_compat(str(model_path))
    indices = args.case_indices if args.case_indices is not None else tuple(range(args.first_n))
    failed_cases = 0
    for offset, case_index in enumerate(indices):
        case_dir = args.output_root / args.model_name / f"case_{case_index}"
        manifest_path = case_dir / "manifest.json"
        if manifest_path.is_file() and not args.force:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("complete") is True:
                log.info("Skip existing DE trace case=%s", case_index)
                continue
        case_seed = int(args.seed) + int(case_index)
        config = DeConfig(
            maxiter=args.maxiter,
            population_size=args.population_size,
            seed=case_seed,
            encoding=args.encoding,
            objective=args.objective,
            case_timeout=args.case_timeout,
        )
        log.info("DE scout case=%s (%s/%s) seed=%s", case_index, offset + 1, len(indices), case_seed)
        case_dir = begin_de_artifact(
            output_root=args.output_root,
            model_name=args.model_name,
            model_path=model_path,
            dataset=args.dataset,
            case_index=case_index,
            config=config,
        )
        try:
            result = run_de_scout(
                model,
                dataset.x_test[case_index],
                config,
                generation_sink=lambda generation, arrays: write_generation_shard(
                    case_dir, generation, arrays
                ),
            )
            output = write_de_artifact(
                output_root=args.output_root,
                model_name=args.model_name,
                model_path=model_path,
                dataset=args.dataset,
                case_index=case_index,
                result=result,
            )
            log.info(
                "DE case=%s success=%s margin=%.6g generations=%s output=%s",
                case_index,
                result.success,
                result.best_margin,
                result.completed_generations,
                output,
            )
        except Exception as exc:
            failed_cases += 1
            mark_de_artifact_failed(case_dir, exc)
            log.exception("DE scout failed for case=%s", case_index)
    return 1 if failed_cases else 0


def _position_parts(position):
    if not isinstance(position, list) or len(position) != 2:
        raise ValueError(f"Invalid branch position: {position!r}")
    layer = int(position[0])
    indices = position[1]
    if isinstance(indices, list) and indices and isinstance(indices[0], list):
        normalized = [tuple(int(part) for part in item) for item in indices]
    elif isinstance(indices, list):
        normalized = tuple(int(part) for part in indices)
    else:
        raise ValueError(f"Invalid branch indices: {indices!r}")
    return layer, normalized


def _run_audit(args: argparse.Namespace) -> int:
    if args.first_n is not None and args.first_n < 1:
        raise ValueError("--first-n must be >= 1")
    if args.train_pairs < 0 or args.holdout_pairs < 0:
        raise ValueError("pair counts must be non-negative")
    if args.train_end_generation < 1 or args.holdout_end_generation <= args.train_end_generation:
        raise ValueError("held-out generations must follow a non-empty training range")
    if args.replay_timeout < 1:
        raise ValueError("--replay-timeout must be >= 1")

    import numpy as np

    from datasets.cifar10 import Cifar10Dataset
    from explainability.shap_calculator import ShapValuesComparator
    from pyct.de.replay import replay_one_pixel_path

    model_path = Path("model") / f"{args.model_name}.h5"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model_sha256 = sha256_file(model_path)
    dataset = Cifar10Dataset()
    model_trace_root = args.trace_root / args.model_name
    case_dirs = []
    for case_dir in sorted(model_trace_root.glob("case_*"), key=lambda path: int(path.name.split("_")[-1])):
        manifest_path = case_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True or manifest.get("success") is True:
            continue
        if manifest.get("model_sha256") != model_sha256:
            raise ValueError(f"DE trace model hash mismatch: {case_dir}")
        case_dirs.append((int(manifest["case_index"]), case_dir))
    if args.case_indices is not None:
        selected_indices = set(args.case_indices)
        case_dirs = [item for item in case_dirs if item[0] in selected_indices]
    else:
        case_dirs = case_dirs[: args.first_n]
    if not case_dirs:
        raise ValueError(f"No complete DE-fail traces found under {model_trace_root}")

    case_records = []
    errors = []
    audit_config = {
        "train_pairs": int(args.train_pairs),
        "holdout_pairs": int(args.holdout_pairs),
        "train_end_generation": int(args.train_end_generation),
        "holdout_end_generation": int(args.holdout_end_generation),
        "shrinkage": float(args.shrinkage),
        "replay_timeout": int(args.replay_timeout),
    }
    for offset, (case_index, case_dir) in enumerate(case_dirs):
        output_path = args.output_root / args.model_name / f"case_{case_index}" / "guidance.json"
        if output_path.is_file() and not args.force:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 2 or payload.get("audit_config") != audit_config:
                raise ValueError(
                    f"Existing guidance config does not match this run: {output_path}; "
                    "use a new output root or --force"
                )
            case_records.append(payload)
            continue
        log.info("Audit replay case=%s (%s/%s)", case_index, offset + 1, len(case_dirs))
        try:
            sample = dataset.x_test[case_index]
            cache_path = Path("shap_value_all_layer") / args.model_name / f"shap_value_{case_index}.json"
            if not cache_path.is_file():
                raise FileNotFoundError(f"Precomputed SHAP cache not found: {cache_path}")
            (
                _in_dict,
                _con_dict,
                input_for_shap,
                background,
            ) = dataset.get_cifar10_test_data_and_set_condict(case_index, [])
            comparator = ShapValuesComparator(
                model_path=str(model_path),
                background_dataset=background,
                input=np.expand_dims(input_for_shap, axis=0),
                idx=case_index,
                shap_value_pre_calculated=True,
            )

            def replay(coordinate, value):
                return replay_one_pixel_path(
                    model_name=args.model_name,
                    case_index=case_index,
                    clean_image=sample,
                    coordinate=coordinate,
                    value=value,
                    model_sha256=model_sha256,
                    timeout=args.replay_timeout,
                )

            def shap_lookup(position):
                layer, indices = _position_parts(position)
                return comparator.get_shap_influence(layer, indices)

            payload = audit_case(
                case_dir=case_dir,
                replay=replay,
                shap_lookup=shap_lookup,
                train_pairs=args.train_pairs,
                holdout_pairs=args.holdout_pairs,
                train_end_generation=args.train_end_generation,
                holdout_end_generation=args.holdout_end_generation,
                shrinkage=args.shrinkage,
                checkpoint_dir=output_path.parent,
                replay_timeout=args.replay_timeout,
                force_checkpoint=args.force,
            )
            write_json_atomic(output_path, payload)
            case_records.append(payload)
        except Exception as exc:
            log.exception("Audit failed for case=%s", case_index)
            errors.append({"case_index": case_index, "error": f"{exc.__class__.__name__}: {exc}"})

    evaluable = [
        case
        for case in case_records
        if all(case.get("ndcg_at_6", {}).get(name) is not None for name in ("de", "shap", "path"))
    ]
    transitions = sum(int(case.get("holdout_attributed_count", 0)) for case in case_records)
    holdout_pairs = sum(int(case.get("holdout_pair_count", 0)) for case in case_records)
    replay_count = sum(int(case.get("replay_count", 0)) for case in case_records)
    partial_replays = sum(int(case.get("partial_replay_count", 0)) for case in case_records)
    replay_duration = sum(
        float(case.get("replay_summary", {}).get("total_duration_seconds", 0.0))
        for case in case_records
    )
    attribution_status_counts = {"train": {}, "holdout": {}}
    for case in case_records:
        for split in ("train", "holdout"):
            for status, count in case.get("attribution_status_counts", {}).get(split, {}).items():
                current = attribution_status_counts[split].get(status, 0)
                attribution_status_counts[split][status] = current + int(count)
    summary = {
        "schema_version": 2,
        "model_name": args.model_name,
        "model_sha256": model_sha256,
        "audit_config": audit_config,
        "requested_case_count": len(case_dirs),
        "completed_case_count": len(case_records),
        "evaluable_case_count": len(evaluable),
        "mapped_holdout_transitions": transitions,
        "holdout_pair_count": holdout_pairs,
        "holdout_attribution_coverage": (
            float(transitions / holdout_pairs) if holdout_pairs else None
        ),
        "replay_count": replay_count,
        "partial_replay_count": partial_replays,
        "partial_replay_rate": (
            float(partial_replays / replay_count) if replay_count else None
        ),
        "total_replay_duration_seconds": float(replay_duration),
        "attribution_status_counts": attribution_status_counts,
        "data_sufficient": len(evaluable) >= 10 and transitions >= 30,
        "case_metrics": [
            {"case_index": case["case_index"], **case["ndcg_at_6"]} for case in evaluable
        ],
        "errors": errors,
    }
    if evaluable:
        de_values = [case["ndcg_at_6"]["de"] for case in evaluable]
        summary["lift_vs_shap"] = paired_bootstrap_lift(
            de_values, [case["ndcg_at_6"]["shap"] for case in evaluable]
        )
        summary["lift_vs_path"] = paired_bootstrap_lift(
            de_values, [case["ndcg_at_6"]["path"] for case in evaluable]
        )
    write_json_atomic(args.output_root / args.model_name / "audit.json", summary)
    log.info(
        "Audit model=%s evaluable=%s transitions=%s sufficient=%s",
        args.model_name,
        len(evaluable),
        transitions,
        summary["data_sufficient"],
    )
    return 1 if errors else 0


def _run_gate(args: argparse.Namespace) -> int:
    model_summaries = []
    pooled = []
    for model_name in args.models:
        path = args.audit_root / model_name / "audit.json"
        if not path.is_file():
            raise FileNotFoundError(f"Model audit not found: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        model_summaries.append(summary)
        pooled.extend({"model_name": model_name, **item} for item in summary.get("case_metrics", []))
    if not pooled:
        raise ValueError("No evaluable case metrics found in model audits")
    de_values = [item["de"] for item in pooled]
    lift_vs_shap = paired_bootstrap_lift(
        de_values,
        [item["shap"] for item in pooled],
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    lift_vs_path = paired_bootstrap_lift(
        de_values,
        [item["path"] for item in pooled],
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    positive_models = sum(
        1
        for summary in model_summaries
        if summary.get("lift_vs_shap", {}).get("point", 0.0) > 0.0
        and summary.get("lift_vs_path", {}).get("point", 0.0) > 0.0
    )
    passed = (
        all(summary.get("data_sufficient") is True for summary in model_summaries)
        and lift_vs_shap["ci95_lower"] > 0.0
        and lift_vs_path["ci95_lower"] > 0.0
        and positive_models >= 3
    )
    payload = {
        "schema_version": 1,
        "models": list(args.models),
        "model_count": len(model_summaries),
        "pooled_case_count": len(pooled),
        "positive_model_count": positive_models,
        "lift_vs_shap": lift_vs_shap,
        "lift_vs_path": lift_vs_path,
        "passed": passed,
        "decision": "go" if passed else "no-go",
    }
    output = args.output or (args.audit_root / "gate.json")
    write_json_atomic(output, payload)
    log.info("DE guidance gate decision=%s output=%s", payload["decision"], output)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s | %(name)s | %(message)s",
    )
    if args.command == "scout":
        return _run_scout(args)
    if args.command == "audit":
        return _run_audit(args)
    if args.command == "gate":
        return _run_gate(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


__all__ = ["build_parser", "main"]
