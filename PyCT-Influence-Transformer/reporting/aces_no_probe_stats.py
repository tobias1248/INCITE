"""Offline audit and ground-truth statistics for scalar ACES-like No Probe runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from libct.aces_like import ACES_LIKE_EXPERIMENT_CONTROLS


TERMINAL_STATUSES = {"success", "timeout", "exhausted", "incomplete"}


def _read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object: {}".format(path))
    return data


def build_report(root: Path, ground_truth: Sequence[int], source_images: np.ndarray) -> Dict[str, Any]:
    """Audit saved metadata/artifacts; predictions are recorded exact Keras labels.

    This does not rerun inference or attest that model weights still match a run.
    Campaign manifests supply expected cases; raw directories have unknown coverage.
    """
    root = Path(root).resolve()
    if len(ground_truth) != len(source_images):
        raise ValueError("Ground truth and source image counts differ")
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    errors, rows, groups = [], [], []
    specifications = manifest.get("experiments")
    if specifications is None:
        discovered = {}
        for path in sorted(root.rglob("stats.json")):
            try:
                meta = _read_json(path).get("meta", {})
                if not isinstance(meta, dict):
                    raise ValueError("Invalid meta mapping")
                key = (meta.get("model_name"), meta.get("global_real_shift_kind"))
                discovered.setdefault(key, []).append(path)
            except (ValueError, OSError) as exc:
                errors.append("{}: {}".format(path, exc))
        specifications = [{"model": m, "control": c, "paths": paths}
                          for (m, c), paths in discovered.items()]
    if not isinstance(specifications, list) or not specifications:
        errors.append("No experiments found")
        specifications = []
    original_predictions = {}
    seen_groups = set()
    for specification in specifications:
        model, control = specification.get("model"), specification.get("control")
        group_errors = []
        if (model, control) in seen_groups:
            group_errors.append("Duplicate experiment")
        seen_groups.add((model, control))
        if not isinstance(model, str) or control not in ACES_LIKE_EXPERIMENT_CONTROLS:
            group_errors.append("Invalid model/control")
        completion = None
        # Locate by recorded keys rather than constructing an untrusted filename.
        for path in root.glob("*.completion.json"):
            try:
                record = _read_json(path)
                if record.get("model") == model and record.get("control") == control:
                    if completion is not None:
                        group_errors.append("Duplicate completion record")
                    completion = record
            except (ValueError, OSError) as exc:
                group_errors.append("Invalid completion {}: {}".format(path, exc))
        result_dir = specification.get("result_dir") or (completion or {}).get("result_dir")
        if specification.get("result_dir") and (completion or {}).get("result_dir"):
            def resolved_directory(value):
                path = Path(value)
                return (path if path.is_absolute() else root / path).resolve()
            if resolved_directory(specification["result_dir"]) != resolved_directory(completion["result_dir"]):
                group_errors.append("Manifest and completion result directories differ")
        launched = bool(completion or result_dir or specification.get("paths"))
        paths = specification.get("paths", [])
        if result_dir:
            directory = Path(result_dir)
            if not directory.is_absolute():
                directory = root / directory
            paths = sorted(directory.rglob("stats.json"))
        if completion and (completion.get("exit_code") != 0 or not completion.get("result_dir")):
            group_errors.append("CLI failed or result directory is ambiguous/missing")
        expected = specification.get("case_ids")
        if expected is not None and (not isinstance(expected, list) or not expected or
                any(type(i) is not int or not 0 <= i < len(ground_truth) for i in expected) or
                len(expected) != len(set(expected))):
            group_errors.append("Invalid expected case IDs")
            expected = []
        found, group_rows = set(), []
        for path in paths:
            path = Path(path)
            try:
                data = _read_json(path)
                meta = data["meta"]
                for section in ("meta", "summary", "solver", "constraints"):
                    if not isinstance(data.get(section, {}), dict):
                        raise ValueError("Invalid {} mapping".format(section))
                index = meta.get("idx")
                if type(index) is not int or not 0 <= index < len(ground_truth):
                    raise ValueError("Invalid/missing case index")
                if index in found:
                    raise ValueError("Duplicate case {}".format(index))
                found.add(index)
                if expected is not None and index not in expected:
                    raise ValueError("Unexpected case {}".format(index))
                if meta.get("model_name") != model or meta.get("global_real_shift_kind") != control:
                    raise ValueError("Model/control mismatch")
                if meta.get("semantic_attack_mode", meta.get("attack_mode")) != "global-real" or meta.get("hybrid_de_status") is not None:
                    raise ValueError("Not standalone PyCT")
                environment = manifest.get("environment", {})
                suffix = environment.get("PYCT_ATTACK_MODE_SUFFIX")
                if suffix:
                    mode = environment.get("PYCT_SMT_PATH_MODE", "full")
                    prefix = "global-real_" + ("smtlast_" if mode == "last" else "") + suffix + "_clip_" + control + "_"
                    if not isinstance(meta.get("attack_mode"), str) or not meta["attack_mode"].startswith(prefix):
                        raise ValueError("Campaign layout/provenance mismatch")
                if meta.get("global_real_probe_enabled") is not False:
                    raise ValueError("Probe is not explicitly disabled")
                if meta.get("global_real_reference_transform") != "exact":
                    raise ValueError("Reference transform is not exact")
                phases = data.get("summary", {}).get("reference_prediction_phase_counts", {})
                if not isinstance(phases, dict):
                    raise ValueError("Invalid reference prediction phase counts")
                if any(value for name, value in phases.items() if "probe" in name):
                    raise ValueError("Probe prediction found")
                if any(value for name, value in meta.items() if
                       ("probe_evaluated_count" in name or "probe_solved" in name)):
                    raise ValueError("Probe evaluation/solution found")
                status = meta.get("status")
                if status not in TERMINAL_STATUSES:
                    raise ValueError("Invalid terminal status: {}".format(status))
                original = meta.get("original_label")
                # Initial-forward timeout legitimately has no original prediction.
                if original is not None and (type(original) is not int or not 0 <= original < 10):
                    raise ValueError("Invalid original label")
                source = np.load(path.parent / "ori_input.npy", allow_pickle=False)
                cached = np.asarray(source_images[index], dtype=np.float32)
                if source.shape != cached.shape or not np.array_equal(source.astype(np.float32), cached):
                    raise ValueError("Source image does not match dataset case")
                key = (model, index)
                if original is not None:
                    if key in original_predictions and original_predictions[key] != original:
                        raise ValueError("Original prediction differs across controls")
                    original_predictions[key] = original
                final = meta.get("attack_label")
                if status == "success":
                    if original is None or type(final) is not int or not 0 <= final < 10 or final == original:
                        raise ValueError("Success is not a prediction flip")
                    adversarial = np.load(path.parent / "adv_input.npy", allow_pickle=False)
                    if adversarial.shape != cached.shape or not np.all(np.isfinite(adversarial)):
                        raise ValueError("Invalid adversarial image")
                elif final is not None:
                    raise ValueError("Attack label on non-success case")
                truth = int(ground_truth[index])
                correct = None if original is None else original == truth
                outcome = "no_flip"
                if status == "success":
                    outcome = "correct_to_wrong" if correct else ("wrong_to_correct" if final == truth else "wrong_to_wrong")
                elif original is None:
                    outcome = "original_unknown"
                settings = manifest.get("settings", {})
                checks = {"path_threshold": "symbolic_path_threshold", "build_timeout": "constraint_build_timeout_seconds",
                          "x_min": "global_real_requested_min", "x_max": "global_real_requested_max",
                          "score_alpha": "score_alpha", "pwl_max_segments": "global_real_pwl_max_segments",
                          "pwl_error": "global_real_pwl_error_tolerance", "timeout": "timeout",
                          "solver_timeout": "solver_run_timeout"}
                for setting, field in checks.items():
                    if setting in settings and (meta.get(field) is None or float(meta[field]) != float(settings[setting])):
                        raise ValueError("Manifest parameter mismatch: {}".format(setting))
                mode = manifest.get("environment", {}).get("PYCT_SMT_PATH_MODE")
                if mode and meta.get("smt_path_mode") != mode:
                    raise ValueError("SMT mode mismatch")
                row = {"model": model, "control": control, "case_id": index, "status": status,
                       "ground_truth": truth, "original_label": original, "attack_label": final,
                       "original_correct": correct, "primary_gt_outcome": outcome, "stats_path": str(path),
                       "source_sha256": hashlib.sha256(source.tobytes()).hexdigest()}
                for name in ("total_wall_time", "total_cpu_time", "solve_constraint_wall_time_total"):
                    value = data.get("summary", {}).get(name)
                    if value is not None and (type(value) not in (float, int) or not math.isfinite(value) or value < 0):
                        raise ValueError("Invalid duration: {}".format(name))
                    row[name] = value
                for name in ("sat", "unsat", "unknown"):
                    value = data.get("solver", {}).get(name, 0)
                    if type(value) is not int or value < 0:
                        raise ValueError("Invalid solver counter: {}".format(name))
                    row[name] = value
                for name in ("generated_total", "solved_total"):
                    value = data.get("constraints", {}).get(name, 0)
                    if type(value) is not int or value < 0:
                        raise ValueError("Invalid constraint counter: {}".format(name))
                    row[name] = value
                group_rows.append(row)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                group_errors.append("{}: {}".format(path, exc))
        missing = sorted(set(expected or []) - found)
        valid_ids = {row["case_id"] for row in group_rows}
        complete = launched and expected is not None and bool(expected) and not missing and valid_ids == set(expected) and not group_errors
        outcomes = Counter(row["primary_gt_outcome"] for row in group_rows)
        denominator = sum(row["original_correct"] is True for row in group_rows)
        unknown = sum(row["original_correct"] is None for row in group_rows)
        groups.append({"model": model, "control": control, "launched": launched, "complete": complete,
                       "expected_count": None if expected is None else len(expected), "observed_count": len(group_rows),
                       "missing_case_ids": missing, "errors": group_errors, "status_counts": dict(Counter(r["status"] for r in group_rows)),
                       "original_correct_count": denominator, "original_unknown_count": unknown,
                       "prediction_flips": sum(r["status"] == "success" for r in group_rows),
                       "correct_to_wrong": outcomes["correct_to_wrong"], "wrong_to_correct": outcomes["wrong_to_correct"],
                       "wrong_to_wrong": outcomes["wrong_to_wrong"],
                       "asr_observed": outcomes["correct_to_wrong"] / denominator if denominator else None,
                       "asr_complete": outcomes["correct_to_wrong"] / denominator if complete and not unknown and denominator else None,
                       "solver_totals": {k: sum(r[k] or 0 for r in group_rows) for k in ("sat", "unsat", "unknown", "generated_total", "solved_total")}})
        rows.extend(group_rows)
        errors.extend("{}/{}: {}".format(model, control, e) for e in group_errors)
    return {"schema_version": "aces-no-probe-report-v1", "root": str(root), "manifest": manifest,
            "complete": bool(groups) and all(g["complete"] for g in groups) and not errors,
            "errors": errors, "groups": groups, "rows": rows,
            "notes": ["ASR = correct-to-wrong / originally correct; ground truth is fixed.",
                      "Solver totals are cumulative recorded iterations; interrupted iterations may be omitted.",
                      "Exact predictions are audited from runtime metadata, not independently replayed."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-cache", type=Path, help="Local CIFAR10 test_batch pickle file; never downloads")
    parser.add_argument("--check-launched", action="store_true", help="Fail if any launched experiment lacks valid terminal results")
    parser.add_argument("--require-complete", action="store_true", help="Fail unless every expected experiment/case is valid")
    args = parser.parse_args()
    from datasets.keras_cache import get_keras_datasets_dir
    cache = args.dataset_cache or get_keras_datasets_dir() / "cifar-10-batches-py" / "test_batch"
    with cache.open("rb") as handle:
        batch = pickle.load(handle, encoding="bytes")
    images = np.asarray(batch[b"data"]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255
    report = build_report(args.path, batch[b"labels"], images)
    report["dataset_cache"] = str(cache.resolve())
    report["dataset_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output_dir / "cases.csv").open("w", newline="") as handle:
        if report["rows"]:
            writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0]))
            writer.writeheader()
            writer.writerows(report["rows"])
    print("groups={} cases={} complete={} errors={}".format(len(report["groups"]), len(report["rows"]), report["complete"], len(report["errors"])))
    failed = bool(report["errors"])
    if args.check_launched:
        failed |= any(g["launched"] and not g["complete"] for g in report["groups"])
    if args.require_complete:
        failed |= not report["complete"]
    return int(failed)
