"""Offline audit and ground-truth statistics for scalar ACES-like No Probe runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import itertools
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from libct.aces_like import ACES_LIKE_EXPERIMENT_CONTROLS
from explainability.shap_contract import fingerprint_array, load_target_class_cache
from datasets.common import select_background_per_class
from libct.searcher.constraint_sensitivity import ALGORITHM_VERSION
from libct.aces_like_multi import MULTI_SHIFT_KIND, MULTI_TRANSFORM_MODE, normalize_controls, normalize_bounds, apply_aces_like_multi_transform
from libct.global_real import validate_global_real_config
from libct.aces_like_pwl_nd import mesh_fingerprint


TERMINAL_STATUSES = {"success", "timeout", "exhausted", "incomplete"}


def _read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object: {}".format(path))
    return data


def _audit_multi(specification, meta, manifest, source, directory, status, *, audit_artifacts=True):
    if not manifest:
        raise ValueError("Multi-control audit requires a campaign manifest")
    controls = normalize_controls(specification.get("controls"))
    bounds = normalize_bounds(controls, specification.get("control_bounds"))
    if (list(controls) != meta.get("global_real_controls")
            or list(controls) != meta.get("global_real_control_order")
            or [list(b) for b in bounds] != meta.get("global_real_control_bounds")):
        raise ValueError("Multi-control combination, order or bounds mismatch")
    settings = manifest.get("settings", {})
    if (settings.get("shap_priority_enabled") != "1" or settings.get("shap_sensitivity_enabled") != "0"
            or meta.get("score_alpha") != 0):
        raise ValueError("Multi-control requires SHAP-only with alpha=0")
    config = {k[len("global_real_"):]: v for k, v in meta.items() if k.startswith("global_real_")}
    config["global_shift_kind"] = meta.get("global_real_shift_kind")
    if config.get("transform_mode") != MULTI_TRANSFORM_MODE:
        raise ValueError("Invalid multi-control transform mode")
    validate_global_real_config(config)
    if mesh_fingerprint(source, config) != config["pwl_mesh_sha256"]:
        raise ValueError("Multi-control mesh fingerprint differs from source/configuration")
    if "pwl_max_regions" in settings and int(settings["pwl_max_regions"]) != config["pwl_max_regions"]:
        raise ValueError("Multi-control region budget differs from manifest")
    if not audit_artifacts:
        return list(controls), [list(b) for b in bounds], None
    sat = directory / "sat_aces_params.npy"
    if not sat.is_file():
        raise ValueError("Missing multi-control SAT parameter array")
    vectors = np.load(sat, allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[1] != len(controls) or not np.isfinite(vectors).all():
        raise ValueError("Invalid multi-control SAT parameter array")
    for column, (lo, hi) in zip(vectors.T, bounds):
        if np.any(column < lo) or np.any(column > hi):
            raise ValueError("SAT parameter outside multi-control bounds")
    sat_images = np.load(directory / "sat_inputs.npy", allow_pickle=False)
    if sat_images.shape != (len(vectors),) + source.shape:
        raise ValueError("SAT images and parameter vectors have different shapes/counts")
    for values, image in zip(vectors, sat_images):
        exact = apply_aces_like_multi_transform(source, values, controls=controls).rgb.astype(np.float32)
        if not np.array_equal(exact, image):
            raise ValueError("SAT image differs from exact multi-control rendering")
    if len(vectors):
        last = vectors[-1].tolist()
        if (meta.get("global_real_last_sat_parameter_vector") != last
                or meta.get("global_real_last_sat_params") != dict(zip(controls, last))):
            raise ValueError("Last SAT parameters differ from parameter artifacts")
    parameters = None
    if status == "success":
        parameters = meta.get("global_real_solved_params")
        if not isinstance(parameters, dict) or set(parameters) != set(controls):
            raise ValueError("Missing or invalid multi-control solved parameters")
        values = [parameters[c] for c in controls]
        if any(type(v) not in (int, float) or not math.isfinite(v) or not lo <= v <= hi
               for v, (lo, hi) in zip(values, bounds)):
            raise ValueError("Invalid multi-control solved parameter value")
        if meta.get("global_real_solved_parameter_vector") != values:
            raise ValueError("Solved parameter vector differs from named parameters")
        if not len(vectors) or vectors[-1].tolist() != values:
            raise ValueError("Successful parameters differ from final SAT vector")
        exact = apply_aces_like_multi_transform(source, values, controls=controls).rgb.astype(np.float32)
        adversarial = np.load(directory / "adv_input.npy", allow_pickle=False)
        if not np.array_equal(exact, adversarial):
            raise ValueError("Adversarial image differs from exact multi-control rendering")
    return list(controls), [list(b) for b in bounds], parameters


def _audit_shap(meta, manifest, model, index, image, background, cache_memo):
    settings = manifest.get("settings", {})
    flags = {}
    for name in ("shap_priority_enabled", "shap_sensitivity_enabled"):
        value = settings.get(name, "0")
        if value not in ("0", "1"):
            raise ValueError("Invalid manifest SHAP flag: {}".format(name))
        flags[name] = value == "1"
        if name in settings and meta.get("global_real_" + name) is not flags[name]:
            raise ValueError("SHAP flag mismatch: {}".format(name))
    if flags["shap_sensitivity_enabled"] and not flags["shap_priority_enabled"]:
        raise ValueError("Sensitivity requires SHAP priority")
    if flags["shap_priority_enabled"]:
        cache = Path(meta["global_real_shap_cache_path"])
        expected = Path(settings["shap_output_root"]) / model / "shap_value_{}.json".format(index)
        if not cache.is_absolute() or cache.resolve() != expected.resolve():
            raise ValueError("SHAP cache path mismatch")
        if cache not in cache_memo:
            metadata, values = load_target_class_cache(cache)
            if not any(key.split("_", 1)[0].isdigit() for key in values):
                raise ValueError("SHAP cache lacks internal-layer values")
            cache_memo[cache] = metadata, hashlib.sha256(cache.read_bytes()).hexdigest()
        metadata, digest = cache_memo[cache]
        if meta.get("global_real_shap_cache_sha256") != digest or meta.get("global_real_shap_cache_metadata") != metadata:
            raise ValueError("SHAP cache provenance mismatch")
        identity = {"case_index": index, "input": fingerprint_array(image[None]),
                    "background": background, "explainer_type": "gradient",
                    "background_per_class": int(settings.get("bg_per_class", 3)),
                    "background_seed": int(settings.get("bg_seed", 2233)), "class_count": 10}
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise ValueError("SHAP cache identity mismatch")
        if metadata["model"]["sha256"] != manifest.get("model_sha256", {}).get(model):
            raise ValueError("SHAP model hash mismatch")
        if meta.get("original_label") is not None and metadata["target_class"] != meta["original_label"]:
            raise ValueError("SHAP target differs from original prediction")
    result = {}
    for name, field in (("shap_lookups", "global_real_shap_lookup_total"),
                        ("shap_nonzero", "global_real_shap_nonzero_total")):
        value = meta.get(field, 0)
        if type(value) is not int or value < 0:
            raise ValueError("Invalid SHAP lookup counter")
        result[name] = value
    if result["shap_nonzero"] > result["shap_lookups"]:
        raise ValueError("Invalid nonzero SHAP count")
    stats = meta.get("global_real_sensitivity_stats")
    if stats is not None:
        if not flags["shap_sensitivity_enabled"] or not isinstance(stats, dict) or stats.get("version") != ALGORITHM_VERSION:
            raise ValueError("Invalid sensitivity telemetry")
        for key, expected in (("step_fraction", .01), ("epsilon", 1e-8), ("max_nodes", 200000), ("wall_budget", 2.0)):
            if stats.get(key) != expected:
                raise ValueError("Sensitivity settings mismatch")
    for key in ("attempted", "evaluated", "fallback", "interrupted", "nodes", "seconds", "forwards"):
        value = (stats or {}).get(key, 0)
        valid_type = type(value) in (int, float) if key == "seconds" else type(value) is int
        if not valid_type or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid sensitivity counter: {}".format(key))
        result["sensitivity_" + key] = value
    if result["sensitivity_attempted"] != sum(result["sensitivity_" + key] for key in ("evaluated", "fallback", "interrupted")):
        raise ValueError("Sensitivity counter totals differ")
    reasons = (stats or {}).get("fallback_reasons", {})
    if not isinstance(reasons, dict) or any(type(v) is not int or v < 0 for v in reasons.values()) or sum(reasons.values()) != result["sensitivity_fallback"]:
        raise ValueError("Invalid sensitivity fallback reasons")
    result["sensitivity_fallback_reasons"] = reasons
    return result


def build_report(root: Path, ground_truth: Sequence[int], source_images: np.ndarray,
                 *, _manifest=None, _audit_context=None, _allow_partial_timeout=False) -> Dict[str, Any]:
    """Audit saved metadata/artifacts; predictions are recorded exact Keras labels.

    This does not rerun inference or attest that model weights still match a run.
    Campaign manifests supply expected cases; raw directories have unknown coverage.
    """
    root = Path(root).resolve()
    if _manifest is None and next(root.rglob("recipe_summary.json"), None) is not None:
        return build_recipe_report(root, ground_truth, source_images)
    if len(ground_truth) != len(source_images):
        raise ValueError("Ground truth and source image counts differ")
    manifest_path = root / "manifest.json"
    manifest = _manifest if _manifest is not None else (_read_json(manifest_path) if manifest_path.exists() else {})
    cache_memo = (_audit_context or {}).get("cache_memo", {})
    settings = manifest.get("settings", {})
    background = (_audit_context or {}).get("background")
    if background is None and settings.get("shap_priority_enabled") == "1":
        background = fingerprint_array(select_background_per_class(
            np.asarray(source_images, dtype=np.float32), np.asarray(ground_truth),
            per_class=int(settings.get("bg_per_class", 3)), seed=int(settings.get("bg_seed", 2233))))
    errors, rows, groups = [], [], []
    specifications = manifest.get("experiments")
    if specifications is None:
        discovered = {}
        for path in sorted(root.rglob("stats.json")):
            try:
                meta = _read_json(path).get("meta", {})
                if not isinstance(meta, dict):
                    raise ValueError("Invalid meta mapping")
                controls = normalize_controls(meta.get("global_real_controls")) if meta.get("global_real_shift_kind") == MULTI_SHIFT_KIND else ()
                key = (meta.get("model_name"), meta.get("global_real_shift_kind"), controls)
                entry = discovered.setdefault(key, {"model": key[0], "control": key[1], "paths": [],
                                                    "controls": list(controls),
                                                    "control_bounds": meta.get("global_real_control_bounds")})
                entry["paths"].append(path)
            except (ValueError, OSError, TypeError) as exc:
                errors.append("{}: {}".format(path, exc))
        specifications = list(discovered.values())
    if not isinstance(specifications, list) or not specifications:
        errors.append("No experiments found")
        specifications = []
    original_predictions = {}
    seen_groups = set()
    for specification in specifications:
        if not isinstance(specification, dict):
            errors.append("Invalid experiment specification")
            continue
        model, control = specification.get("model"), specification.get("control")
        group_errors = []
        if not isinstance(model, str) or not isinstance(control, str):
            errors.append("Invalid model/control")
            continue
        try:
            controls_key = normalize_controls(specification.get("controls")) if control == MULTI_SHIFT_KIND else ()
        except (ValueError, TypeError) as exc:
            errors.append("Invalid experiment controls: {}".format(exc))
            continue
        if (model, control, controls_key) in seen_groups:
            group_errors.append("Duplicate experiment")
        seen_groups.add((model, control, controls_key))
        if not isinstance(model, str) or control not in ACES_LIKE_EXPERIMENT_CONTROLS + (MULTI_SHIFT_KIND,):
            group_errors.append("Invalid model/control")
        completion = None
        # Locate by recorded keys rather than constructing an untrusted filename.
        for path in root.glob("*.completion.json"):
            try:
                record = _read_json(path)
                if (record.get("model") == model and record.get("control") == control
                        and (control != MULTI_SHIFT_KIND or record.get("controls") == list(controls_key))):
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
                cached = np.asarray(source_images[index], dtype=np.float32)
                partial = (_allow_partial_timeout and status == "incomplete"
                           and data.get("summary", {}).get("total_wall_time") is None)
                source_path = path.parent / "ori_input.npy"
                source = cached if partial and not source_path.is_file() else np.load(source_path, allow_pickle=False)
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
                selected, bounds, parameters = None, None, None
                if control == MULTI_SHIFT_KIND:
                    artifacts_complete = all((path.parent / name).is_file() for name in ("sat_aces_params.npy", "sat_inputs.npy"))
                    selected, bounds, parameters = _audit_multi(
                        specification, meta, manifest, cached, path.parent, status,
                        audit_artifacts=not partial or artifacts_complete)
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
                       "controls": selected, "control_bounds": bounds, "parameters": parameters,
                       "ground_truth": truth, "original_label": original, "attack_label": final,
                       "original_correct": correct, "primary_gt_outcome": outcome, "stats_path": str(path),
                       "source_sha256": hashlib.sha256(source.tobytes()).hexdigest()}
                row.update(_audit_shap(meta, manifest, model, index, cached, background, cache_memo))
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
                       "controls": list(controls_key) if controls_key else None,
                       "control_bounds": specification.get("control_bounds") if controls_key else None,
                       "expected_count": None if expected is None else len(expected), "observed_count": len(group_rows),
                       "missing_case_ids": missing, "errors": group_errors, "status_counts": dict(Counter(r["status"] for r in group_rows)),
                       "original_correct_count": denominator, "original_unknown_count": unknown,
                       "prediction_flips": sum(r["status"] == "success" for r in group_rows),
                       "correct_to_wrong": outcomes["correct_to_wrong"], "wrong_to_correct": outcomes["wrong_to_correct"],
                       "wrong_to_wrong": outcomes["wrong_to_wrong"],
                       "asr_observed": outcomes["correct_to_wrong"] / denominator if denominator else None,
                       "asr_complete": outcomes["correct_to_wrong"] / denominator if complete and not unknown and denominator else None,
                       "solver_totals": {k: sum(r[k] or 0 for r in group_rows) for k in ("sat", "unsat", "unknown", "generated_total", "solved_total")}})
        totals = {key: sum(row["sensitivity_" + key] for row in group_rows)
                  for key in ("attempted", "evaluated", "fallback", "interrupted", "nodes", "seconds", "forwards")}
        groups[-1]["sensitivity_totals"] = totals
        groups[-1]["sensitivity_coverage"] = totals["evaluated"] / totals["attempted"] if totals["attempted"] else None
        groups[-1]["shap_lookup_totals"] = {key: sum(row[key] for row in group_rows) for key in ("shap_lookups", "shap_nonzero")}
        groups[-1]["sensitivity_fallback_reasons"] = dict(sum((Counter(row["sensitivity_fallback_reasons"]) for row in group_rows), Counter()))
        rows.extend(group_rows)
        errors.extend("{}/{}: {}".format(model, control, e) for e in group_errors)
    return {"schema_version": "aces-no-probe-report-v1", "root": str(root), "manifest": manifest,
            "complete": bool(groups) and all(g["complete"] for g in groups) and not errors,
            "errors": errors, "groups": groups, "rows": rows,
            "notes": ["ASR = correct-to-wrong / originally correct; ground truth is fixed.",
                      "Solver totals are cumulative recorded iterations; interrupted iterations may be omitted.",
                      "Sensitivity counters cover returned child envelopes; hard-killed forwards may be omitted. Budget/unsupported fallbacks use raw SHAP and may outrank weighted branches.",
                      "Exact predictions are audited from runtime metadata, not independently replayed."]}


def _recipe_duration(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid recipe duration: {}".format(name))
    return float(value)


def _audit_recipe(path, specification, manifest, ground_truth, source_images, context):
    summary = _read_json(path)
    model, index = summary.get("model"), summary.get("case_id")
    if summary.get("schema_version") != "aces-recipes-v1" or model != specification["model"]:
        raise ValueError("Recipe schema/model mismatch")
    if type(index) is not int or not 0 <= index < len(ground_truth):
        raise ValueError("Invalid recipe case index")
    controls = normalize_controls(specification.get("controls"))
    bounds = normalize_bounds(controls, specification.get("control_bounds"))
    if summary.get("controls") != list(controls) or summary.get("control_bounds") != [list(b) for b in bounds]:
        raise ValueError("Recipe pool/order/bounds mismatch")
    if summary.get("policy") != "staged":
        raise ValueError("Invalid recipe policy")
    budgets = summary.get("region_budgets")
    if (not isinstance(budgets, list) or not budgets
            or any(type(n) is not int or n <= 0 for n in budgets)
            or any(b <= a for a, b in zip(budgets, budgets[1:]))):
        raise ValueError("Invalid recipe PWL ladder")
    build_limit = _recipe_duration(summary.get("pwl_build_timeout_seconds"), "PWL build deadline")
    if build_limit <= 0:
        raise ValueError("Invalid recipe PWL build deadline")
    for key in ("aces_pwl_region_budgets", "pwl_region_budgets", "region_budgets"):
        if key in manifest.get("settings", {}) and manifest["settings"][key] != budgets:
            raise ValueError("Recipe PWL ladder differs from manifest")
    for key in ("aces_pwl_build_timeout", "pwl_build_timeout_seconds"):
        if key in manifest.get("settings", {}) and float(manifest["settings"][key]) != build_limit:
            raise ValueError("Recipe PWL build deadline differs from manifest")
    total = _recipe_duration(summary.get("total_budget_seconds"), "total_budget_seconds")
    elapsed = _recipe_duration(summary.get("total_wall_time"), "total_wall_time")
    if total <= 0 or ("timeout" in manifest.get("settings", {}) and total != float(manifest["settings"]["timeout"])):
        raise ValueError("Recipe total budget differs from manifest")
    # Supervising joins allow at most 0.4 seconds of cleanup per recipe.
    if elapsed > total + 5:
        raise ValueError("Recipe exceeded total deadline and cleanup allowance")
    status = summary.get("status")
    if status not in {"success", "timeout", "exhausted", "incomplete", "invalid", "error"}:
        raise ValueError("Recipe lacks terminal status")
    original, final = summary.get("original_label"), summary.get("attack_label")
    for label in (original, final):
        if label is not None and (type(label) is not int or not 0 <= label < 10):
            raise ValueError("Invalid recipe label")
    recipes = list(itertools.combinations(controls, 2))
    if len(controls) == 3:
        recipes.append(controls)
    attempts = summary.get("attempts")
    if not isinstance(attempts, list) or len(attempts) > len(recipes):
        raise ValueError("Invalid recipe attempts")
    spent, previous_end, children = 0., 0., []
    search_count, success_count = 0, 0
    seen_paths = set()
    for ordinal, attempt in enumerate(attempts):
        selected = recipes[ordinal]
        selected_bounds = [list(bounds[controls.index(c)]) for c in selected]
        if (not isinstance(attempt, dict) or type(attempt.get("index")) is not int
                or attempt["index"] != ordinal or attempt.get("controls") != list(selected)
                or attempt.get("control_bounds") != selected_bounds):
            raise ValueError("Recipe attempts are not the deterministic prefix")
        allocation = _recipe_duration(attempt.get("allocated_seconds"), "allocated_seconds")
        wall = _recipe_duration(attempt.get("wall_time"), "attempt wall_time")
        # Remaining wall budget is divided over all unattempted recipes.
        started = _recipe_duration(attempt.get("started_after_seconds", spent), "started_after_seconds")
        expected_allocation = max(0., total - started) / (len(recipes) - ordinal)
        if (started + .01 < previous_end or started >= total or allocation <= 0
                or allocation > expected_allocation + .01 or wall > allocation + 1):
            raise ValueError("Recipe stage allocation/deadline mismatch")
        previous_end = started + wall
        spent += wall
        state = attempt.get("status")
        if state not in {"success", "timeout", "exhausted", "incomplete", "pwl_budget_exhausted", "pwl_build_timeout", "error"}:
            raise ValueError("Invalid recipe attempt status")
        if state in {"success", "error"} and ordinal != len(attempts) - 1:
            raise ValueError("Terminal recipe attempt must be last")
        pwl = attempt.get("pwl")
        if not isinstance(pwl, dict):
            raise ValueError("Missing recipe PWL receipt")
        history = pwl.get("budget_history", pwl.get("pwl_build_history", []))
        if not isinstance(history, list) or len(history) > len(budgets):
            raise ValueError("Invalid recipe PWL history")
        caps = [item.get("budget") if isinstance(item, dict) else item for item in history]
        if caps != budgets[:len(caps)]:
            raise ValueError("Recipe PWL history differs from region ladder")
        for position, receipt in enumerate(history):
            if (not isinstance(receipt, dict) or receipt.get("reason") not in {"accepted", "budget_exhausted", "timeout", "error"}
                    or type(receipt.get("region_count")) is not int or receipt["region_count"] <= 0):
                raise ValueError("Invalid recipe PWL history receipt")
            error = receipt.get("max_abs_error")
            if error is not None and (type(error) not in (int, float) or not math.isfinite(error) or error < 0):
                raise ValueError("Invalid recipe PWL history error")
            if position < len(history) - 1 and receipt["reason"] != "budget_exhausted":
                raise ValueError("Recipe PWL ladder advanced without exhausting previous budget")
        if state == "pwl_budget_exhausted" and (not history or history[-1]["reason"] != "budget_exhausted"):
            raise ValueError("PWL exhaustion lacks final budget-exhausted receipt")
        if state in {"success", "exhausted", "incomplete"} and (not history or history[-1]["reason"] != "accepted"):
            raise ValueError("Recipe search lacks accepted PWL receipt")
        build_time = pwl.get("build_wall_time", pwl.get("pwl_build_wall_time", 0))
        if _recipe_duration(build_time, "PWL build time") > min(build_limit, allocation) + 1:
            raise ValueError("Recipe PWL build exceeded deadline")
        stats_path = attempt.get("stats_path")
        if state in {"pwl_budget_exhausted", "pwl_build_timeout", "error"}:
            if state == "pwl_budget_exhausted" and caps != budgets:
                raise ValueError("PWL budget exhaustion lacks complete region ladder")
            continue
        search_count += 1
        if not isinstance(stats_path, str) or not Path(stats_path).is_absolute():
            if state == "timeout" and stats_path is None:
                if attempt.get("timeout_phase") == "integer_search_budget":
                    available = _recipe_duration(attempt.get("search_seconds_available"), "search_seconds_available")
                    if (available >= 1 or not history or history[-1]["reason"] != "accepted"
                            or wall + available + .05 < allocation):
                        raise ValueError("Invalid fractional search-budget timeout receipt")
                    continue
                if wall + .5 >= allocation:
                    continue
            raise ValueError("Search attempt lacks absolute stats path")
        child_path = Path(stats_path).resolve()
        expected_path = path.parent / "recipes" / ("{:02d}-{}".format(ordinal + 1, "-".join(selected))) / "stats.json"
        if child_path != expected_path.resolve():
            raise ValueError("Recipe child stats path differs from nested attempt directory")
        if child_path in seen_paths:
            raise ValueError("Duplicate recipe child stats path")
        seen_paths.add(child_path)
        child_data = _read_json(child_path)
        child_meta = child_data.get("meta", {})
        if not isinstance(child_meta, dict):
            raise ValueError("Invalid recipe child metadata")
        effective_cap = child_meta.get("global_real_pwl_max_regions")
        if type(effective_cap) is not int or effective_cap not in budgets or not caps or caps[-1] != effective_cap:
            raise ValueError("Effective child PWL cap differs from build history")
        last_receipt = history[-1]
        if (last_receipt["reason"] != "accepted"
                or last_receipt["region_count"] != child_meta.get("global_real_pwl_region_count")
                or last_receipt.get("max_abs_error") != child_meta.get("global_real_pwl_max_abs_error")
                or last_receipt["region_count"] > effective_cap):
            raise ValueError("Final PWL receipt differs from accepted child mesh")
        child_timeout = _recipe_duration(child_meta.get("timeout"), "child timeout")
        if not 0 < child_timeout <= allocation:
            raise ValueError("Child search timeout exceeds recipe stage allocation")
        for field in ("pwl_build_history", "pwl_build_wall_time", "pwl_max_regions", "pwl_region_count", "pwl_max_abs_error", "pwl_error_tolerance"):
            if field in pwl and pwl[field] != child_meta.get("global_real_" + field):
                raise ValueError("Recipe PWL receipt differs from child metadata: {}".format(field))
        child_manifest = dict(manifest)
        child_manifest["settings"] = dict(manifest.get("settings", {}), timeout=child_timeout, pwl_max_regions=effective_cap)
        child_manifest["experiments"] = [{"model": model, "control": MULTI_SHIFT_KIND,
            "controls": list(selected), "control_bounds": selected_bounds, "case_ids": [index], "paths": [child_path]}]
        child = build_report(path.parent, ground_truth, source_images, _manifest=child_manifest, _audit_context=context,
                             _allow_partial_timeout=state == "timeout" and wall + .5 >= allocation)
        if child["errors"] or len(child["rows"]) != 1:
            raise ValueError("Recipe child audit failed: {}".format("; ".join(child["errors"])))
        row = child["rows"][0]
        matching_status = row["status"] == state or (state == "timeout" and row["status"] in {"incomplete", "timeout"})
        if not matching_status or (row["original_label"] is not None and row["original_label"] != original):
            raise ValueError("Recipe child status/original label mismatch")
        if row["status"] == "success":
            success_count += 1
            if final != row["attack_label"]:
                raise ValueError("Recipe winning attack label mismatch")
        children.append(row)
    if spent > elapsed + 1 or previous_end > elapsed + 1:
        raise ValueError("Recipe durations exceed case accounting")
    if original is not None and status not in {"error", "invalid"} and not any(c["original_label"] == original for c in children):
        raise ValueError("Recipe original label lacks verified child prediction")
    winner = summary.get("winning_recipe")
    if status == "success":
        if success_count != 1 or type(winner) is not int or winner != len(attempts) - 1 or original is None or final == original:
            raise ValueError("Recipe success lacks verified final winner")
    elif success_count or final is not None or winner is not None:
        raise ValueError("Non-success recipe has winner/attack label")
    if status == "exhausted" and (len(attempts) != len(recipes)
            or any(a["status"] != "exhausted" for a in attempts)):
        raise ValueError("Recipe exhaustion lacks a complete valid search")
    if attempts and all(a["status"] in {"pwl_budget_exhausted", "pwl_build_timeout"} for a in attempts) and status not in {"invalid", "error"}:
        raise ValueError("All PWL failures must be invalid")
    if not search_count and status not in {"invalid", "error", "timeout"}:
        raise ValueError("All PWL failures must be invalid")
    if status == "incomplete" and (not search_count or len(attempts) != len(recipes)
            or not any(a["status"] in {"incomplete", "pwl_budget_exhausted", "pwl_build_timeout"} for a in attempts)):
        raise ValueError("Recipe incomplete status lacks partial search coverage")
    if status == "timeout" and not any(a["status"] == "timeout" for a in attempts) and elapsed + .01 < total:
        raise ValueError("Recipe timeout lacks a search/global deadline")
    if status == "invalid" and search_count:
        raise ValueError("Invalid recipe contains a completed search")
    if any(a["status"] == "error" for a in attempts) and status != "error":
        raise ValueError("Recipe hides unexpected attempt error")
    parent = _read_json(path.parent / "stats.json")
    meta = parent.get("meta", {})
    if not isinstance(meta, dict) or not isinstance(parent.get("summary"), dict):
        raise ValueError("Invalid recipe parent metadata/summary")
    if (meta.get("semantic_attack_mode") != "aces-recipes" or meta.get("model_name") != model
            or type(meta.get("idx")) is not int or meta["idx"] != index
            or any(meta.get(k) != summary.get(k) for k in ("status", "original_label", "attack_label"))
            or parent.get("summary", {}).get("total_wall_time") != summary["total_wall_time"]):
        raise ValueError("Recipe parent stats mismatch")
    truth = int(ground_truth[index])
    correct = None if original is None else original == truth
    outcome = "original_unknown" if original is None else "no_flip"
    if status == "success":
        outcome = "correct_to_wrong" if correct else ("wrong_to_correct" if final == truth else "wrong_to_wrong")
    row = {"model": model, "control": "aces-recipes", "case_id": index, "status": status,
           "controls": list(controls), "control_bounds": [list(b) for b in bounds],
           "ground_truth": truth, "original_label": original, "attack_label": final,
           "original_correct": correct, "primary_gt_outcome": outcome, "stats_path": str(path.parent / "stats.json"),
           "recipe_summary_path": str(path), "total_wall_time": elapsed, "winning_recipe": winner,
           "attempt_count": len(attempts), "search_attempt_count": search_count}
    for key in ("sat", "unsat", "unknown", "generated_total", "solved_total", "shap_lookups", "shap_nonzero"):
        row[key] = sum(child[key] or 0 for child in children)
    return row


def build_recipe_report(root: Path, ground_truth: Sequence[int], source_images: np.ndarray) -> Dict[str, Any]:
    """Audit staged parents and aggregate one attack case per model/dataset index."""
    root = Path(root).resolve()
    if len(ground_truth) != len(source_images):
        raise ValueError("Ground truth and source image counts differ")
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    errors, rows, groups = [], [], []
    specifications = manifest.get("experiments")
    if not isinstance(specifications, list) or not specifications:
        return {"schema_version": "aces-recipes-report-v1", "root": str(root), "manifest": manifest,
                "complete": False, "errors": ["Recipe audit requires campaign manifest experiments"], "groups": [], "rows": []}
    settings = manifest.get("settings", {})
    context = {"cache_memo": {}}
    if settings.get("shap_priority_enabled") == "1":
        context["background"] = fingerprint_array(select_background_per_class(
            np.asarray(source_images, dtype=np.float32), np.asarray(ground_truth),
            per_class=int(settings.get("bg_per_class", 3)), seed=int(settings.get("bg_seed", 2233))))
    seen = set()
    for spec in specifications:
        group_errors, group_rows, found = [], [], set()
        model = spec.get("model") if isinstance(spec, dict) else None
        expected, launched = [], False
        try:
            if not isinstance(model, str) or spec.get("control") != MULTI_SHIFT_KIND:
                raise ValueError("Invalid recipe experiment model/control")
            normalize_bounds(normalize_controls(spec.get("controls")), spec.get("control_bounds"))
            if settings.get("shap_priority_enabled") != "1" or settings.get("shap_sensitivity_enabled") != "0" or float(settings.get("score_alpha", -1)) != 0:
                raise ValueError("Recipes require SHAP-only priority and alpha=0")
            expected = spec.get("case_ids")
            if (not isinstance(expected, list) or not expected
                    or any(type(i) is not int or not 0 <= i < len(ground_truth) for i in expected)
                    or len(expected) != len(set(expected))):
                expected = []
                raise ValueError("Invalid expected recipe case IDs")
            directory = Path(spec["result_dir"])
            if not directory.is_absolute():
                directory = root / directory
            launched = directory.is_dir()
            for path in sorted(directory.rglob("recipe_summary.json")):
                try:
                    identity = _read_json(path).get("case_id")
                    if type(identity) is not int or identity not in expected:
                        raise ValueError("Unexpected recipe case index")
                    if identity in found or (model, identity) in seen:
                        raise ValueError("Duplicate recipe case")
                    found.add(identity)
                    seen.add((model, identity))
                    row = _audit_recipe(path, spec, manifest, ground_truth, source_images, context)
                    group_rows.append(row)
                    if row["status"] in {"invalid", "error"}:
                        group_errors.append("{}: recipe status {}".format(path, row["status"]))
                except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
                    group_errors.append("{}: {}".format(path, exc))
        except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
            group_errors.append(str(exc))
        missing = sorted(set(expected or []) - found)
        valid = [r for r in group_rows if r["status"] not in {"invalid", "error"}]
        outcomes = Counter(r["primary_gt_outcome"] for r in valid)
        denominator = sum(r["original_correct"] is True for r in valid)
        unknown = sum(r["original_correct"] is None for r in valid)
        complete = launched and bool(expected) and not missing and len(valid) == len(expected) and all(r["status"] != "incomplete" for r in valid) and not group_errors
        groups.append({"model": model, "control": "aces-recipes", "launched": launched, "complete": complete,
            "controls": spec.get("controls") if isinstance(spec, dict) else None,
            "expected_count": len(expected or []), "observed_count": len(group_rows), "missing_case_ids": missing,
            "invalid_count": len(found) - len(valid), "errors": group_errors,
            "status_counts": dict(Counter(r["status"] for r in group_rows)),
            "original_correct_count": denominator, "original_unknown_count": unknown,
            "prediction_flips": sum(r["status"] == "success" for r in valid),
            "correct_to_wrong": outcomes["correct_to_wrong"], "wrong_to_correct": outcomes["wrong_to_correct"],
            "wrong_to_wrong": outcomes["wrong_to_wrong"],
            "asr_observed": outcomes["correct_to_wrong"] / denominator if denominator else None,
            "asr_complete": outcomes["correct_to_wrong"] / denominator if complete and not unknown and denominator else None,
            "solver_totals": {k: sum(r[k] for r in valid) for k in ("sat", "unsat", "unknown", "generated_total", "solved_total")}})
        errors.extend("{}: {}".format(model, e) for e in group_errors)
        rows.extend(group_rows)
    return {"schema_version": "aces-recipes-report-v1", "root": str(root), "manifest": manifest,
            "complete": bool(groups) and all(g["complete"] for g in groups) and not errors,
            "errors": errors, "groups": groups, "rows": rows,
            "notes": ["Each model/case contributes once; recipe attempts are not independent attack cases.",
                      "Invalid/error cases are excluded from ASR denominators; exact predictions use recorded runtime metadata."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", action="store_true", help="Audit staged ACES recipe cases")
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
    builder = build_recipe_report if args.recipes else build_report
    report = builder(args.path, batch[b"labels"], images)
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
