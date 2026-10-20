from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import sys

import numpy as np
import pytest

from reporting.aces_no_probe_stats import build_report, main
from explainability.shap_contract import build_cache_metadata
from datasets.common import select_background_per_class
from libct.aces_like_multi import apply_aces_like_multi_transform
from libct.aces_like_pwl_nd import mesh_fingerprint


def _fixture(tmp_path, case_ids=None, control="aces-exposure"):
    images = np.stack([np.full((32, 32, 3), (i + 1) / 10, dtype=np.float32) for i in range(5)])
    truth = [0, 1, 2, 3, 4]
    root = tmp_path / "campaign"
    result = root / "results"
    result.mkdir(parents=True)
    specification = {"model": "model_a", "control": control, "result_dir": "results", "case_ids": [0, 1, 2, 3] if case_ids is None else case_ids}
    (root / "manifest.json").write_text(json.dumps({"experiments": [specification]}))
    return root, result, truth, images


def _case(result, images, index, original, final=None, status=None, **metadata):
    directory = result / f"case_{index}"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "ori_input.npy", images[index])
    meta = {"model_name": "model_a", "global_real_shift_kind": "aces-exposure", "idx": index,
            "attack_mode": "global-real", "global_real_probe_enabled": False,
            "global_real_reference_transform": "exact", "original_label": original,
            "attack_label": final, "status": status or ("success" if final is not None else "timeout")}
    meta.update(metadata)
    data = {"meta": meta, "summary": {"total_wall_time": 1.0},
            "solver": {"sat": 1, "unsat": 2}, "constraints": {"generated_total": 7, "solved_total": 3}}
    path = directory / "stats.json"
    path.write_text(json.dumps(data))
    if final is not None:
        np.save(directory / "adv_input.npy", images[index] + np.float32(0.01))
    return path


def _shap_fixture(tmp_path):
    root, result, truth, images = _fixture(tmp_path, [0])
    model = tmp_path / "model_a.h5"
    model.write_bytes(b"offline-model-identity")
    background = select_background_per_class(images, np.asarray(truth), per_class=3, seed=2233)
    meta = build_cache_metadata(case_index=0, model_path=model, input_data=images[0][None],
        background_dataset=background, explainer_type="gradient", target_class=0, class_count=10,
        background_per_class=3, background_seed=2233)
    cache_root = root / "shap"
    cache = cache_root / "model_a" / "shap_value_0.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"__meta__": meta, "values": {"0_0": 0.1, "1_0": 0.2}}))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["settings"] = {"shap_priority_enabled": "1", "shap_sensitivity_enabled": "1",
        "shap_output_root": str(cache_root), "bg_per_class": "3", "bg_seed": "2233"}
    manifest["model_sha256"] = {"model_a": meta["model"]["sha256"]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    path = _case(result, images, 0, 0, 1,
        global_real_shap_priority_enabled=True, global_real_shap_sensitivity_enabled=True,
        global_real_shap_cache_path=str(cache),
        global_real_shap_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        global_real_shap_cache_metadata=meta,
        global_real_shap_lookup_total=3, global_real_shap_nonzero_total=2,
        global_real_sensitivity_stats={"version": "target-shap-constraint-sensitivity-v1",
            "attempted": 3, "evaluated": 2, "fallback": 1, "interrupted": 0,
            "fallback_reasons": {"unsupported_operator": 1}, "nodes": 20, "seconds": 0.01,
            "step_fraction": 0.01, "epsilon": 1e-8, "max_nodes": 200000,
            "wall_budget": 2.0, "forwards": 1})
    return root, truth, images, path, cache


def test_report_accepts_opt_in_cached_shap_provenance(tmp_path):
    root, truth, images, _, _ = _shap_fixture(tmp_path)
    report = build_report(root, truth, images)
    assert report["complete"] is True, report["errors"]
    assert report["groups"][0]["asr_complete"] == 1.0
    row = report["rows"][0]
    assert row["sensitivity_attempted"] == 3
    assert row["sensitivity_evaluated"] == 2
    assert row["sensitivity_fallback"] == 1
    assert row["shap_lookups"] == 3
    assert row["shap_nonzero"] == 2
    assert report["groups"][0]["sensitivity_coverage"] == pytest.approx(2 / 3)


def _multi_shap_fixture(tmp_path, controls=("exposure", "contrast")):
    root, truth, images, path, cache = _shap_fixture(tmp_path)
    controls = list(controls)
    bounds = [[-.02, .03] for _ in controls]
    params = [.01 * (i + 1) for i in range(len(controls))]
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["experiments"][0].update(control="aces-multi", controls=controls, control_bounds=bounds)
    manifest["settings"].update(shap_sensitivity_enabled="0", score_alpha="0")
    (root / "manifest.json").write_text(json.dumps(manifest))
    data = json.loads(path.read_text())
    meta = data["meta"]
    meta.pop("global_real_sensitivity_stats")
    meta.update(global_real_shift_kind="aces-multi", global_real_transform_mode="aces-like-pwl-nd",
        global_real_controls=controls, global_real_control_order=controls,
        global_real_control_bounds=bounds, global_real_control_version="ordered-photo-controls-v1",
        global_real_variable_names=["__pyct_aces_" + c for c in controls],
        global_real_pwl_nd_knots=[[lo, 0., hi] for lo, hi in bounds],
        global_real_pwl_region_count=math.factorial(len(controls)) * 2 ** len(controls),
        global_real_pwl_max_regions=512, global_real_pwl_error_tolerance=1 / 255,
        global_real_pwl_max_abs_error=.0001,
        global_real_pwl_error_metric="sampled-max-abs-rgb",
        global_real_pwl_validator_version="conforming-simplex-quarter-v1",
        global_real_bounds_mode="clip", global_real_shap_sensitivity_enabled=False,
        score_alpha=0., global_real_solved_params=dict(zip(controls, params)),
        global_real_solved_parameter_vector=params,
        global_real_last_sat_params=dict(zip(controls, params)),
        global_real_last_sat_parameter_vector=params)
    config = {key[len("global_real_"):]: value for key, value in meta.items() if key.startswith("global_real_")}
    config["global_shift_kind"] = "aces-multi"
    meta["global_real_pwl_mesh_sha256"] = mesh_fingerprint(images[0], config)
    path.write_text(json.dumps(data))
    exact = apply_aces_like_multi_transform(images[0], params, controls=controls).rgb
    np.save(path.parent / "adv_input.npy", exact.astype(np.float32))
    np.save(path.parent / "sat_aces_params.npy", np.asarray([params]))
    np.save(path.parent / "sat_inputs.npy", exact[None].astype(np.float32))
    return root, truth, images, path, cache


@pytest.mark.parametrize("controls", [("exposure", "contrast"), ("exposure", "contrast", "saturation")])
def test_report_accepts_independent_multi_control_success(tmp_path, controls):
    root, truth, images, _, _ = _multi_shap_fixture(tmp_path, controls)
    report = build_report(root, truth, images)
    assert report["complete"], report["errors"]
    assert report["groups"][0]["controls"] == list(controls)
    assert report["groups"][0]["asr_complete"] == 1.
    assert report["rows"][0]["controls"] == list(controls)
    assert report["rows"][0]["sensitivity_attempted"] == 0


def test_report_keeps_different_multi_control_combinations_separate(tmp_path):
    root, truth, images, path, _ = _multi_shap_fixture(tmp_path / "first")
    other_root, _, _, other_path, _ = _multi_shap_fixture(tmp_path / "second", ("exposure", "saturation"))
    # Share the exact same cache provenance so only the selected controls differ.
    data = json.loads(other_path.read_text())
    original = json.loads(path.read_text())["meta"]
    for key in ("global_real_shap_cache_path", "global_real_shap_cache_sha256", "global_real_shap_cache_metadata"):
        data["meta"][key] = original[key]
    other_path.write_text(json.dumps(data))
    manifest = json.loads((root / "manifest.json").read_text())
    other = json.loads((other_root / "manifest.json").read_text())["experiments"][0]
    other["result_dir"] = str(other_path.parent.parent)
    manifest["experiments"].append(other)
    (root / "manifest.json").write_text(json.dumps(manifest))
    report = build_report(root, truth, images)
    assert report["complete"], report["errors"]
    assert {tuple(group["controls"]) for group in report["groups"]} == {
        ("exposure", "contrast"), ("exposure", "saturation")}
    assert [group["observed_count"] for group in report["groups"]] == [1, 1]


def test_multi_control_audit_requires_campaign_manifest(tmp_path):
    root, truth, images, _, _ = _multi_shap_fixture(tmp_path)
    (root / "manifest.json").unlink()
    report = build_report(root, truth, images)
    assert not report["complete"]
    assert any("manifest" in error.lower() for error in report["errors"])
    assert report["rows"] == []


@pytest.mark.parametrize("controls", [None, [["exposure"], "contrast"]])
@pytest.mark.parametrize("raw", [False, True])
def test_report_audits_malformed_multi_controls_without_crashing(tmp_path, controls, raw):
    root, truth, images, path, _ = _multi_shap_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    if raw:
        manifest_path.unlink()
        data = json.loads(path.read_text())
        data["meta"]["global_real_controls"] = controls
        path.write_text(json.dumps(data))
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["experiments"][0]["controls"] = controls
        manifest_path.write_text(json.dumps(manifest))
    report = build_report(root, truth, images)
    assert not report["complete"]
    assert report["errors"]
    assert report["rows"] == []


@pytest.mark.parametrize("damage", [
    "missing_vector", "wrong_vector_shape", "nonfinite_vector", "outside_bounds", "solved_vector_mismatch",
    "order", "bounds", "exact_image", "priority", "sensitivity", "alpha", "probe",
    "empty_sat", "unrelated_sat", "wrong_sat_image", "sat_row_count", "last_sat_params", "last_sat_vector",
    "mesh_knots", "mesh_error",
])
def test_report_rejects_invalid_multi_control_provenance_and_artifacts(tmp_path, damage):
    root, truth, images, path, _ = _multi_shap_fixture(tmp_path)
    data = json.loads(path.read_text())
    meta = data["meta"]
    vector = path.parent / "sat_aces_params.npy"
    if damage == "missing_vector":
        vector.unlink()
    elif damage == "wrong_vector_shape":
        np.save(vector, np.asarray([.01, .02]))
    elif damage == "nonfinite_vector":
        np.save(vector, np.asarray([[float("nan"), .02]]))
    elif damage == "outside_bounds":
        np.save(vector, np.asarray([[.5, .02]]))
    elif damage == "solved_vector_mismatch":
        meta["global_real_solved_parameter_vector"] = [.01, .01]
    elif damage == "empty_sat":
        np.save(vector, np.empty((0, 2)))
        np.save(path.parent / "sat_inputs.npy", np.empty((0,) + images[0].shape))
    elif damage == "unrelated_sat":
        params = [.01, .01]
        np.save(vector, np.asarray([params]))
        exact = apply_aces_like_multi_transform(images[0], params, controls=meta["global_real_controls"]).rgb
        np.save(path.parent / "sat_inputs.npy", exact[None].astype(np.float32))
        meta["global_real_last_sat_params"] = dict(zip(meta["global_real_controls"], params))
        meta["global_real_last_sat_parameter_vector"] = params
    elif damage == "wrong_sat_image":
        np.save(vector, np.asarray([[.01, .01], meta["global_real_solved_parameter_vector"]]))
        exact = np.load(path.parent / "adv_input.npy", allow_pickle=False)
        np.save(path.parent / "sat_inputs.npy", np.stack([images[0], exact]))
    elif damage == "sat_row_count":
        exact = np.load(path.parent / "adv_input.npy", allow_pickle=False)
        np.save(path.parent / "sat_inputs.npy", np.stack([exact, exact]))
    elif damage == "last_sat_params":
        meta["global_real_last_sat_params"]["contrast"] = .01
    elif damage == "last_sat_vector":
        meta["global_real_last_sat_parameter_vector"] = [.01, .01]
    elif damage == "mesh_knots":
        # Keep geometry valid while changing the interior partition.
        meta["global_real_pwl_nd_knots"][0].insert(1, -.01)
        meta["global_real_pwl_region_count"] = 12
    elif damage == "mesh_error":
        meta["global_real_pwl_max_abs_error"] = .0002
    elif damage == "order":
        meta["global_real_control_order"].reverse()
    elif damage == "bounds":
        meta["global_real_control_bounds"][0] = [-.03, .03]
    elif damage == "exact_image":
        np.save(path.parent / "adv_input.npy", images[0])
    elif damage == "priority":
        meta["global_real_shap_priority_enabled"] = False
    elif damage == "sensitivity":
        meta["global_real_shap_sensitivity_enabled"] = True
    elif damage == "alpha":
        meta["score_alpha"] = .5
    else:
        meta["global_real_probe_enabled"] = True
    path.write_text(json.dumps(data))
    report = build_report(root, truth, images)
    assert not report["complete"]
    assert report["errors"]
    assert report["rows"] == []


@pytest.mark.parametrize("damage", ["priority_flag", "sensitivity_flag", "missing_cache", "cache_hash", "input", "case", "target"])
def test_report_rejects_wrong_cached_shap_provenance(tmp_path, damage):
    root, truth, images, path, cache = _shap_fixture(tmp_path)
    data = json.loads(path.read_text())
    meta = data["meta"]
    if damage == "priority_flag":
        meta["global_real_shap_priority_enabled"] = False
    elif damage == "sensitivity_flag":
        meta["global_real_shap_sensitivity_enabled"] = False
    elif damage == "missing_cache":
        cache.unlink()
    elif damage == "cache_hash":
        cache.write_text(cache.read_text() + " ")
    else:
        payload = json.loads(cache.read_text())
        cache_meta = payload["__meta__"]
        if damage == "input":
            cache_meta["input"]["sha256"] = "0" * 64
        elif damage == "case":
            cache_meta["case_index"] = 1
        elif damage == "target":
            cache_meta["target_class"] = 1
        meta["global_real_shap_cache_metadata"] = cache_meta
        cache.write_text(json.dumps(payload))
        meta["global_real_shap_cache_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    path.write_text(json.dumps(data))
    report = build_report(root, truth, images)
    assert report["complete"] is False
    assert report["errors"]
    assert report["rows"] == []


@pytest.mark.parametrize("damage", ["negative", "counter_sum", "reason_sum", "algorithm", "settings"])
def test_report_rejects_malformed_sensitivity_telemetry(tmp_path, damage):
    root, truth, images, path, _ = _shap_fixture(tmp_path)
    data = json.loads(path.read_text())
    stats = data["meta"]["global_real_sensitivity_stats"]
    if damage == "negative":
        stats["seconds"] = -1
    elif damage == "counter_sum":
        stats["attempted"] = 100
    elif damage == "reason_sum":
        stats["fallback_reasons"] = {}
    elif damage == "algorithm":
        stats["version"] = "different-method"
    else:
        stats["step_fraction"] = 0.1
    path.write_text(json.dumps(data))
    report = build_report(root, truth, images)
    assert report["complete"] is False
    assert report["errors"]


def test_report_distinguishes_three_flip_outcomes_and_correct_denominator(tmp_path):
    root, result, truth, images = _fixture(tmp_path)
    _case(result, images, 0, 0, 1)
    _case(result, images, 1, 0, 1)
    _case(result, images, 2, 0, 4)
    _case(result, images, 3, 3, status="exhausted")
    report = build_report(root, truth, images)
    assert report["complete"] is True
    assert report["errors"] == []
    group = report["groups"][0]
    assert group["prediction_flips"] == 3
    assert group["correct_to_wrong"] == group["wrong_to_correct"] == group["wrong_to_wrong"] == 1
    assert group["original_correct_count"] == 2
    assert group["asr_complete"] == group["asr_observed"] == 0.5
    assert group["solver_totals"] == {"sat": 4, "unsat": 8, "unknown": 0, "generated_total": 28, "solved_total": 12}
    assert [row["primary_gt_outcome"] for row in report["rows"]] == ["correct_to_wrong", "wrong_to_correct", "wrong_to_wrong", "no_flip"]


def test_partial_and_unknown_original_predictions_do_not_publish_complete_asr(tmp_path):
    root, result, truth, images = _fixture(tmp_path, [0, 1])
    _case(result, images, 0, 0, 1)
    report = build_report(root, truth, images)
    assert report["complete"] is False
    assert report["groups"][0]["missing_case_ids"] == [1]
    assert report["groups"][0]["asr_observed"] == 1.0
    assert report["groups"][0]["asr_complete"] is None
    _case(result, images, 1, None, status="incomplete")
    report = build_report(root, truth, images)
    assert report["complete"] is True
    assert report["groups"][0]["original_unknown_count"] == 1
    assert report["groups"][0]["asr_complete"] is None
    assert report["rows"][1]["primary_gt_outcome"] == "original_unknown"


@pytest.mark.parametrize("metadata,message", [
    ({"global_real_probe_enabled": True}, "Probe"),
    ({"global_real_reference_transform": "pwl"}, "exact"),
    ({"hybrid_de_status": "de_failed"}, "standalone"),
    ({"attack_mode": "hybrid-de"}, "standalone"),
    ({"status": "running"}, "terminal status"),
    ({"model_name": "different_model"}, "mismatch"),
    ({"global_real_probe_evaluated_count_total": 1}, "Probe"),
    ({"original_label": True}, "original label"),
])
def test_report_flags_wrong_pipeline_or_malformed_metadata(tmp_path, metadata, message):
    root, result, truth, images = _fixture(tmp_path, [0])
    _case(result, images, 0, 0, 1, **metadata)
    report = build_report(root, truth, images)
    assert report["complete"] is False
    assert any(message in error for error in report["errors"])
    assert report["rows"] == []


@pytest.mark.parametrize("artifact", ["source_mismatch", "missing_adv", "nan_adv", "adv_shape"])
def test_report_audits_dataset_identity_and_success_artifacts(tmp_path, artifact):
    root, result, truth, images = _fixture(tmp_path, [0])
    path = _case(result, images, 0, 0, 1)
    if artifact == "source_mismatch":
        np.save(path.parent / "ori_input.npy", images[1])
    elif artifact == "missing_adv":
        (path.parent / "adv_input.npy").unlink()
    elif artifact == "nan_adv":
        np.save(path.parent / "adv_input.npy", images[0] * np.nan)
    else:
        np.save(path.parent / "adv_input.npy", np.zeros((1, 3)))
    report = build_report(root, truth, images)
    assert not report["complete"]
    assert report["errors"]
    assert report["groups"][0]["prediction_flips"] == 0


def test_raw_directory_has_unknown_coverage_and_duplicates_are_flagged(tmp_path):
    root, result, truth, images = _fixture(tmp_path, [0])
    path = _case(result, images, 0, 0, 1)
    raw = build_report(result, truth, images)
    assert raw["groups"][0]["expected_count"] is None
    assert not raw["complete"]
    duplicate = result / "duplicate"
    duplicate.mkdir()
    (duplicate / "stats.json").write_text(path.read_text())
    report = build_report(root, truth, images)
    assert not report["complete"]
    assert any("Duplicate case" in error for error in report["errors"])


def test_manifest_parameter_drift_and_nonzero_completion_fail_audit(tmp_path):
    root, result, truth, images = _fixture(tmp_path, [0])
    _case(result, images, 0, 0, 1, symbolic_path_threshold=8000)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["settings"] = {"path_threshold": "2000"}
    (root / "manifest.json").write_text(json.dumps(manifest))
    report = build_report(root, truth, images)
    assert any("parameter mismatch" in error for error in report["errors"])
    (root / "model_a_exposure.completion.json").write_text(json.dumps({"model": "model_a", "control": "aces-exposure", "result_dir": str(result), "exit_code": 1}))
    report = build_report(root, truth, images)
    assert any("CLI failed" in error for error in report["errors"])


def test_decorated_attack_mode_uses_recorded_semantic_mode(tmp_path):
    root, result, truth, images = _fixture(tmp_path, [0])
    _case(result, images, 0, 0, 1, attack_mode="global-real_smtlast_unit_no_probe", semantic_attack_mode="global-real")
    report = build_report(root, truth, images)
    assert report["complete"]
    assert report["groups"][0]["correct_to_wrong"] == 1


@pytest.mark.parametrize("attack_mode,valid", [
    ("global-real_smtlast_current_clip_aces-exposure_xm0p2_0p2", True),
    ("global-real_smtlast_previous_clip_aces-exposure_xm0p2_0p2", False),
    ("global-real_current_clip_aces-exposure_xm0p2_0p2", False),
    ("global-real_smtlast_current_clip_aces-contrast_xm0p2_0p2", False),
])
def test_manifest_campaign_suffix_checks_full_recorded_attack_prefix(tmp_path, attack_mode, valid):
    root, result, truth, images = _fixture(tmp_path, [0])
    _case(result, images, 0, 0, 1, attack_mode=attack_mode,
          semantic_attack_mode="global-real", smt_path_mode="last")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["environment"] = {"PYCT_ATTACK_MODE_SUFFIX": "current", "PYCT_SMT_PATH_MODE": "last"}
    (root / "manifest.json").write_text(json.dumps(manifest))
    report = build_report(root, truth, images)
    assert report["complete"] is valid
    if valid:
        assert report["errors"] == []
    else:
        assert any("provenance mismatch" in error for error in report["errors"])
        assert report["rows"] == []


def test_conflicting_manifest_and_completion_result_directories_fail(tmp_path):
    root, result, truth, images = _fixture(tmp_path, [0])
    _case(result, images, 0, 0, 1)
    other = root / "other_results"
    other.mkdir()
    (root / "model_a_exposure.completion.json").write_text(json.dumps({
        "model": "model_a", "control": "aces-exposure", "result_dir": str(other), "exit_code": 0,
    }))
    report = build_report(root, truth, images)
    assert not report["complete"]
    assert any("result directories differ" in error for error in report["errors"])


def test_empty_case_list_does_not_mark_unlaunched_group_complete(tmp_path):
    root, _, truth, images = _fixture(tmp_path, [])
    manifest = json.loads((root / "manifest.json").read_text())
    del manifest["experiments"][0]["result_dir"]
    (root / "manifest.json").write_text(json.dumps(manifest))
    report = build_report(root, truth, images)
    assert not report["complete"]
    assert report["groups"][0]["launched"] is False
    assert report["groups"][0]["complete"] is False
    assert report["groups"][0]["asr_complete"] is None
    assert any("Invalid expected case IDs" in error for error in report["errors"])


@pytest.mark.parametrize("field,value", [
    ("meta", None), ("meta", []), ("summary", None),
    ("solver", {"sat": "bad", "unsat": 1}),
    ("solver", {"sat": float("nan"), "unsat": 1}),
])
def test_malformed_nested_stats_and_counters_are_audited_errors(tmp_path, field, value):
    root, result, truth, images = _fixture(tmp_path, [0])
    path = _case(result, images, 0, 0, 1)
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    report = build_report(root, truth, images)
    assert not report["complete"]
    assert report["errors"]
    assert report["rows"] == []


def test_report_rejects_disagreeing_original_predictions_between_controls(tmp_path):
    root, result, truth, images = _fixture(tmp_path, [0])
    _case(result, images, 0, 0, 1)
    other = root / "contrast_results"
    other.mkdir()
    _case(other, images, 0, 2, 1, global_real_shift_kind="aces-contrast")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["experiments"].append({"model": "model_a", "control": "aces-contrast", "result_dir": "contrast_results", "case_ids": [0]})
    (root / "manifest.json").write_text(json.dumps(manifest))
    report = build_report(root, truth, images)
    assert any("Original prediction differs" in error for error in report["errors"])


def test_models_keep_separate_original_accuracy_denominators(tmp_path):
    root, result, truth, images = _fixture(tmp_path, [0, 1])
    _case(result, images, 0, 0, 1)
    _case(result, images, 1, 1, status="timeout")
    other = root / "model_b_results"
    other.mkdir()
    _case(other, images, 0, 2, 1, model_name="model_b")
    _case(other, images, 1, 1, 0, model_name="model_b")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["experiments"].append({"model": "model_b", "control": "aces-exposure", "result_dir": "model_b_results", "case_ids": [0, 1]})
    (root / "manifest.json").write_text(json.dumps(manifest))
    report = build_report(root, truth, images)
    assert report["complete"]
    groups = {group["model"]: group for group in report["groups"]}
    assert groups["model_a"]["original_correct_count"] == 2
    assert groups["model_a"]["asr_complete"] == 0.5
    assert groups["model_b"]["original_correct_count"] == 1
    assert groups["model_b"]["asr_complete"] == 1.0


def test_cli_uses_local_cache_and_writes_csv_without_inference(monkeypatch, tmp_path):
    root, result, truth, images = _fixture(tmp_path, [0])
    pixels = (images * 255).astype(np.uint8)
    images = pixels.astype(np.float32) / 255
    _case(result, images, 0, 0, 1)
    cache = tmp_path / "test_batch"
    with cache.open("wb") as handle:
        pickle.dump({b"data": pixels.transpose(0, 3, 1, 2).reshape(5, -1), b"labels": truth}, handle)
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["aces_stats", "--path", str(root), "--dataset-cache", str(cache), "--output-dir", str(output), "--require-complete"])
    assert main() == 0
    report = json.loads((output / "report.json").read_text())
    assert report["complete"]
    with (output / "cases.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["primary_gt_outcome"] == "correct_to_wrong"
    assert rows[0]["original_correct"] == "True"


def _recipe_fixture(tmp_path, staged=False):
    selected = ("exposure", "saturation") if staged else ("exposure", "contrast")
    root, truth, images, path, _ = _multi_shap_fixture(tmp_path, selected)
    parent = path.parent
    child = parent / "recipes" / ("02-exposure-saturation" if staged else "01-exposure-contrast")
    child.mkdir(parents=True)
    for artifact in list(parent.iterdir()):
        if artifact.is_file():
            artifact.rename(child / artifact.name)
    path = child / "stats.json"
    controls = ["exposure", "contrast", "saturation"] if staged else list(selected)
    bounds = [[-.02, .03] for _ in controls]
    allocation = 1799 / 3 if staged else 1800.
    data = json.loads(path.read_text())
    data["meta"]["timeout"] = allocation
    path.write_text(json.dumps(data))
    attempts = []
    if staged:
        attempts.append({"index": 0, "controls": ["exposure", "contrast"], "control_bounds": bounds[:2],
            "allocated_seconds": 450., "started_after_seconds": 0., "wall_time": 1.,
            "status": "pwl_budget_exhausted", "reason": "region budget", "stats_path": None,
            "pwl": {"budget_history": [{"budget": n, "reason": "budget_exhausted", "region_count": 8, "max_abs_error": .1} for n in (512, 1024, 2048)], "build_wall_time": .5}})
    attempts.append({"index": len(attempts), "controls": list(selected), "control_bounds": bounds[:2],
        "allocated_seconds": allocation, "started_after_seconds": 1. if staged else 0., "wall_time": 1.,
        "status": "success", "reason": None, "stats_path": str(path),
        "pwl": {"budget_history": [{"budget": 512, "reason": "accepted", "region_count": 8, "max_abs_error": .0001}], "build_wall_time": .5}})
    summary = {"schema_version": "aces-recipes-v1", "model": "model_a", "case_id": 0,
        "controls": controls, "control_bounds": bounds, "policy": "staged", "region_budgets": [512, 1024, 2048],
        "pwl_build_timeout_seconds": 30, "total_budget_seconds": 1800., "total_wall_time": 2. if staged else 1.,
        "status": "success", "original_label": 0, "attack_label": 1,
        "winning_recipe": len(attempts) - 1, "attempts": attempts}
    summary_path = parent / "recipe_summary.json"
    summary_path.write_text(json.dumps(summary))
    (parent / "stats.json").write_text(json.dumps({"meta": {"semantic_attack_mode": "aces-recipes",
        "model_name": "model_a", "idx": 0, "status": "success", "original_label": 0, "attack_label": 1},
        "summary": {"total_wall_time": summary["total_wall_time"]}}))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["experiments"][0].update(controls=controls, control_bounds=bounds)
    manifest["settings"]["timeout"] = 1800.
    manifest_path.write_text(json.dumps(manifest))
    return root, truth, images, summary_path, path


@pytest.mark.parametrize("staged", [False, True])
def test_recipe_report_counts_parent_once_and_audits_winning_child(tmp_path, staged):
    from reporting.aces_no_probe_stats import build_recipe_report
    root, truth, images, _, _ = _recipe_fixture(tmp_path, staged)
    report = build_recipe_report(root, truth, images)
    assert report["complete"], report["errors"]
    assert len(report["rows"]) == 1
    group = report["groups"][0]
    assert group["observed_count"] == group["prediction_flips"] == group["original_correct_count"] == 1
    assert group["asr_complete"] == 1.
    assert report["rows"][0]["attempt_count"] == (2 if staged else 1)
    # Automatic routing also excludes all nested normal engine stats.
    assert build_report(root, truth, images) == report


@pytest.mark.parametrize("damage", ["order", "cap", "source", "label", "history", "allocation", "parent", "false_exhausted", "child_path", "receipt_count", "receipt_error", "receipt_reason"])
def test_recipe_report_rejects_invalid_stage_contract_or_artifacts(tmp_path, damage):
    root, truth, images, summary_path, path = _recipe_fixture(tmp_path, True)
    summary = json.loads(summary_path.read_text())
    if damage == "order":
        summary["attempts"][0]["controls"] = ["contrast", "saturation"]
    elif damage == "cap":
        summary["attempts"][-1]["pwl"]["budget_history"].append({"budget": 1024})
    elif damage == "source":
        np.save(path.parent / "ori_input.npy", images[1])
    elif damage == "label":
        summary["original_label"] = 3
    elif damage == "history":
        summary["attempts"][0]["pwl"]["budget_history"] = [{"budget": 1024}]
    elif damage == "allocation":
        summary["attempts"][-1]["allocated_seconds"] = 1800.
    elif damage == "parent":
        parent = json.loads((summary_path.parent / "stats.json").read_text())
        parent["meta"]["idx"] = 1
        (summary_path.parent / "stats.json").write_text(json.dumps(parent))
    elif damage == "false_exhausted":
        summary.update(status="exhausted", attack_label=None, winning_recipe=None)
    elif damage == "child_path":
        summary["attempts"][-1]["stats_path"] = str(tmp_path / "unrelated" / "stats.json")
    elif damage == "receipt_count":
        summary["attempts"][-1]["pwl"]["budget_history"][-1]["region_count"] = 12
    elif damage == "receipt_error":
        summary["attempts"][-1]["pwl"]["budget_history"][-1]["max_abs_error"] = .003
    elif damage == "receipt_reason":
        summary["attempts"][-1]["pwl"]["budget_history"][-1]["reason"] = "budget_exhausted"
    summary_path.write_text(json.dumps(summary))
    report = build_report(root, truth, images)
    assert not report["complete"]
    assert report["errors"]
    assert report["rows"] == []
    assert report["groups"][0]["invalid_count"] == 1
    assert report["groups"][0]["asr_complete"] is None


def test_recipe_report_all_pwl_failures_are_invalid_and_excluded_from_denominator(tmp_path):
    root, truth, images, summary_path, _ = _recipe_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary.update(status="invalid", original_label=None, attack_label=None, winning_recipe=None)
    attempt = summary["attempts"][0]
    attempt.update(status="pwl_budget_exhausted", stats_path=None)
    attempt["pwl"]["budget_history"] = [{"budget": n, "reason": "budget_exhausted", "region_count": 8, "max_abs_error": .1} for n in (512, 1024, 2048)]
    summary_path.write_text(json.dumps(summary))
    parent = json.loads((summary_path.parent / "stats.json").read_text())
    parent["meta"].update(status="invalid", original_label=None, attack_label=None)
    (summary_path.parent / "stats.json").write_text(json.dumps(parent))
    report = build_report(root, truth, images)
    assert not report["complete"]
    assert report["groups"][0]["status_counts"] == {"invalid": 1}
    assert report["groups"][0]["invalid_count"] == 1
    assert report["groups"][0]["original_correct_count"] == 0
    assert report["groups"][0]["asr_observed"] is None
    assert len(report["rows"]) == 1


def test_recipe_report_duplicate_model_case_is_not_aggregated_twice(tmp_path):
    root, truth, images, _, _ = _recipe_fixture(tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["experiments"].append(dict(manifest["experiments"][0]))
    path.write_text(json.dumps(manifest))
    report = build_report(root, truth, images)
    assert not report["complete"]
    assert len(report["rows"]) == 1
    assert any("Duplicate recipe case" in error for error in report["errors"])


@pytest.mark.parametrize("status", ["exhausted", "incomplete", "timeout", "partial-timeout"])
def test_recipe_report_terminal_search_status_and_remaining_engine_timeout(tmp_path, status):
    partial = status == "partial-timeout"
    status = "timeout" if partial else status
    root, truth, images, summary_path, path = _recipe_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary.update(status=status, attack_label=None, winning_recipe=None)
    attempt = summary["attempts"][0]
    attempt["status"] = status
    child = json.loads(path.read_text())
    child["meta"].update(status=status, attack_label=None, timeout=1799.)
    if status == "timeout":
        # Hard shutdown can preserve an earlier nonterminal child snapshot.
        child["meta"]["status"] = "incomplete"
        attempt["wall_time"] = summary["total_wall_time"] = 1800.
        if partial:
            child["summary"]["total_wall_time"] = None
            for artifact in ("ori_input.npy", "sat_inputs.npy", "sat_aces_params.npy"):
                (path.parent / artifact).unlink()
    path.write_text(json.dumps(child))
    summary_path.write_text(json.dumps(summary))
    parent_path = summary_path.parent / "stats.json"
    parent = json.loads(parent_path.read_text())
    parent["meta"].update(status=status, attack_label=None)
    parent["summary"]["total_wall_time"] = summary["total_wall_time"]
    parent_path.write_text(json.dumps(parent))
    report = build_report(root, truth, images)
    assert not report["errors"], report["errors"]
    assert len(report["rows"]) == 1
    assert report["rows"][0]["status"] == status
    assert report["complete"] is (status != "incomplete")
    assert report["groups"][0]["asr_complete"] == (None if status == "incomplete" else 0.)


@pytest.mark.parametrize("expected", [[[0]], None, "0"])
def test_recipe_report_bad_expected_ids_are_reported_without_crashing(tmp_path, expected):
    root, truth, images, _, _ = _recipe_fixture(tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["experiments"][0]["case_ids"] = expected
    path.write_text(json.dumps(manifest))
    report = build_report(root, truth, images)
    assert report["errors"]
    assert not report["complete"]


def test_recipe_report_rejects_case_overall_deadline_overrun(tmp_path):
    root, truth, images, summary_path, _ = _recipe_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary["total_wall_time"] = 1900.
    summary_path.write_text(json.dumps(summary))
    report = build_report(root, truth, images)
    assert not report["complete"]
    assert any("total deadline" in error for error in report["errors"])


def test_recipe_report_all_pwl_deadlines_cannot_be_valid_attack_timeout(tmp_path):
    root, truth, images, summary_path, _ = _recipe_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary.update(status="timeout", original_label=None, attack_label=None, winning_recipe=None, total_wall_time=1800.)
    summary["attempts"][0].update(status="pwl_build_timeout", stats_path=None, wall_time=1800.)
    summary_path.write_text(json.dumps(summary))
    report = build_report(root, truth, images)
    assert not report["complete"]
    assert any("All PWL failures" in error for error in report["errors"])


@pytest.mark.parametrize("damage", [None, "whole_second", "negative", "no_mesh", "unaccepted_mesh", "early"])
def test_recipe_report_fractional_remaining_search_budget_requires_receipt(tmp_path, damage):
    root, truth, images, summary_path, child_path = _recipe_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary.update(status="timeout", original_label=None, attack_label=None, winning_recipe=None,
                   total_budget_seconds=1., total_wall_time=.2)
    attempt = summary["attempts"][0]
    attempt.update(status="timeout", stats_path=None, allocated_seconds=1., wall_time=.2,
                   timeout_phase="integer_search_budget", search_seconds_available=.8,
                   reason="Less than one whole second remains after PWL build")
    attempt["pwl"]["build_wall_time"] = .1
    child_path.unlink()
    if damage == "whole_second":
        attempt["search_seconds_available"] = 1.
    elif damage == "negative":
        attempt["search_seconds_available"] = -.1
    elif damage == "no_mesh":
        attempt["pwl"]["budget_history"] = []
    elif damage == "unaccepted_mesh":
        attempt["pwl"]["budget_history"][-1]["reason"] = "budget_exhausted"
    elif damage == "early":
        attempt["search_seconds_available"] = .1
    summary_path.write_text(json.dumps(summary))
    parent_path = summary_path.parent / "stats.json"
    parent = json.loads(parent_path.read_text())
    parent["meta"].update(status="timeout", original_label=None, attack_label=None)
    parent["summary"]["total_wall_time"] = .2
    parent_path.write_text(json.dumps(parent))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["settings"]["timeout"] = 1.
    manifest_path.write_text(json.dumps(manifest))
    report = build_report(root, truth, images)
    if damage is None:
        assert not report["errors"], report["errors"]
        assert report["complete"]
        assert report["rows"][0]["status"] == "timeout"
        assert report["groups"][0]["asr_complete"] is None
    else:
        assert report["errors"]
        assert not report["complete"]
