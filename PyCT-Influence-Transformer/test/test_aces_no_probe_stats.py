from __future__ import annotations

import csv
import json
import pickle
import sys

import numpy as np
import pytest

from reporting.aces_no_probe_stats import build_report, main


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
