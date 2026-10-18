"""Offline regression coverage for campaign SHAP snapshots."""
import hashlib
import json
import pickle
import numpy as np
import pytest

from datasets.common import select_background_per_class
from explainability.shap_contract import build_cache_metadata
from orchestration import aces_shap_cache as cache


@pytest.fixture
def prepared_inputs(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    models = ["model_a", "model_b"]
    pixels = np.arange(30 * 3072, dtype=np.uint32).reshape(30, 3072).astype(np.uint8)
    # Different images within each class make seed/background checks meaningful.
    pixels[:, 0] = np.arange(30, dtype=np.uint8)
    labels = np.repeat(np.arange(10), 3)
    dataset = tmp_path / "test_batch"
    dataset.write_bytes(pickle.dumps({b"data": pixels, b"labels": labels.tolist()}))
    images = pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255
    background = select_background_per_class(images, labels, per_class=1, seed=2233)
    source = tmp_path / "source with spaces"
    for model in models:
        model_path = model_root / (model + ".h5")
        model_path.write_bytes(model.encode())
        (source / model).mkdir(parents=True)
        for index in range(2):
            metadata = build_cache_metadata(
                case_index=index, model_path=model_path, input_data=images[index:index + 1],
                background_dataset=background, explainer_type="gradient", target_class=index,
                class_count=10, background_per_class=1, background_seed=2233,
            )
            (source / model / "shap_value_{}.json".format(index)).write_text(
                json.dumps({"__meta__": metadata, "values": {"input_0_0_0": 0.1, "1_0": -0.2}})
            )
    return dict(mode="reuse", output_root=tmp_path / "campaign" / "shap_target_class",
                manifest=tmp_path / "campaign" / "shap_cache_manifest.json", models=models,
                first_n=2, model_root=model_root, dataset_cache=dataset,
                background_per_class=1, background_seed=2233, source_root=source)


def _source_file(settings):
    return settings["source_root"] / "model_a" / "shap_value_0.json"


def test_reuse_copies_exact_bytes_and_records_provenance_without_source_writes(prepared_inputs):
    settings = prepared_inputs
    original = {str(p.relative_to(settings["source_root"])): p.read_bytes()
                for p in settings["source_root"].rglob("*.json")}
    receipt = cache.prepare_cache(**settings)
    assert receipt["mode"] == "reuse"
    assert receipt["source_root"] == str(settings["source_root"].resolve())
    assert len(receipt["files"]) == 4
    for relative, data in original.items():
        destination = settings["output_root"] / relative
        assert destination.read_bytes() == data
        assert not destination.is_symlink()
        assert destination.stat().st_ino != (settings["source_root"] / relative).stat().st_ino
        assert receipt["files"][relative] == hashlib.sha256(data).hexdigest()
        assert (settings["source_root"] / relative).read_bytes() == data
    # The snapshot remains independent of subsequent changes to the old campaign.
    _source_file(settings).unlink()
    assert cache.prepare_cache(**dict(settings, mode="verify")) == receipt


@pytest.mark.parametrize("corruption", [
    "missing", "json", "input_only", "nan", "model", "input", "background",
    "case_index", "explainer_type", "background_seed", "class_count", "target_class",
])
def test_invalid_source_cache_stops_without_success_receipt(prepared_inputs, corruption):
    path = _source_file(prepared_inputs)
    payload = json.loads(path.read_text())
    if corruption == "missing":
        path.unlink()
    elif corruption == "json":
        path.write_text("{")
    else:
        if corruption == "input_only":
            payload["values"] = {"input_0_0_0": 1.0}
        elif corruption == "nan":
            payload["values"]["1_0"] = float("nan")
        elif corruption in ("model", "input", "background"):
            payload["__meta__"][corruption]["sha256"] = "wrong"
        else:
            payload["__meta__"][corruption] = {
                "case_index": 99, "explainer_type": "kernel", "background_seed": 99,
                "class_count": 2, "target_class": 99,
            }[corruption]
        path.write_text(json.dumps(payload))
    with pytest.raises((OSError, ValueError)):
        cache.prepare_cache(**prepared_inputs)
    assert not prepared_inputs["manifest"].exists()
    assert not prepared_inputs["output_root"].exists()


@pytest.mark.parametrize("change", ["cache", "missing", "model", "dataset", "background", "receipt_files"])
def test_verify_rejects_changed_snapshot_or_identity(prepared_inputs, change):
    cache.prepare_cache(**prepared_inputs)
    settings = dict(prepared_inputs, mode="verify")
    path = settings["output_root"] / "model_b" / "shap_value_1.json"
    if change == "cache":
        path.write_text(path.read_text() + " ")
    elif change == "missing":
        path.unlink()
    elif change == "model":
        (settings["model_root"] / "model_b.h5").write_bytes(b"changed model")
    elif change == "dataset":
        with settings["dataset_cache"].open("ab") as handle:
            handle.write(b"changed data")
    elif change == "background":
        settings["background_seed"] = 42
    else:
        receipt = json.loads(settings["manifest"].read_text())
        receipt["files"].pop("model_b/shap_value_1.json")
        settings["manifest"].write_text(json.dumps(receipt))
    with pytest.raises((OSError, ValueError)):
        cache.prepare_cache(**settings)


def test_partial_copy_failure_never_publishes_receipt(prepared_inputs):
    (prepared_inputs["source_root"] / "model_b/shap_value_1.json").unlink()
    with pytest.raises(ValueError):
        cache.prepare_cache(**prepared_inputs)
    assert (prepared_inputs["output_root"] / "model_a/shap_value_0.json").exists()
    assert not prepared_inputs["manifest"].exists()


def test_cache_changing_during_validation_is_rejected(prepared_inputs, monkeypatch):
    load = cache.load_target_class_cache

    def change_after_validation(path, **kwargs):
        result = load(path, **kwargs)
        path.write_text(path.read_text() + " ")
        return result

    monkeypatch.setattr(cache, "load_target_class_cache", change_after_validation)
    with pytest.raises(ValueError, match="changed during preparation"):
        cache.prepare_cache(**prepared_inputs)
    assert not prepared_inputs["manifest"].exists()


def test_reuse_refuses_existing_output_and_overlapping_roots(prepared_inputs):
    settings = prepared_inputs
    with pytest.raises(ValueError, match="overlap"):
        cache.prepare_cache(**dict(settings, output_root=settings["source_root"]))
    with pytest.raises(ValueError, match="overlap"):
        cache.prepare_cache(**dict(settings, output_root=settings["source_root"] / "child"))
    settings["output_root"].mkdir(parents=True)
    marker = settings["output_root"] / "preserved"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        cache.prepare_cache(**settings)
    assert marker.read_text() == "keep"


def test_refresh_validates_existing_generated_cache_without_rewriting(prepared_inputs):
    settings = dict(prepared_inputs, mode="refresh", output_root=prepared_inputs["source_root"], source_root=None)
    path = _source_file(prepared_inputs)
    original = path.read_bytes()
    receipt = cache.prepare_cache(**settings)
    assert receipt["mode"] == "refresh"
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="already exists"):
        cache.prepare_cache(**settings)


def test_snapshot_refuses_symlinks_and_receipt_inside_source(prepared_inputs):
    settings = prepared_inputs
    with pytest.raises(ValueError, match="outside the source"):
        cache.prepare_cache(**dict(settings, manifest=settings["source_root"] / "receipt.json"))
    link = settings["source_root"].parent / "linked snapshot"
    link.symlink_to(settings["source_root"], target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        cache.prepare_cache(**dict(settings, output_root=link))
    cache.prepare_cache(**settings)
    path = settings["output_root"] / "model_a/shap_value_0.json"
    path.unlink()
    path.symlink_to(_source_file(settings))
    with pytest.raises(ValueError, match="changed"):
        cache.prepare_cache(**dict(settings, mode="verify"))


def test_cli_failure_never_downloads_or_writes_receipt(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PYCT_ALLOW_DATASET_DOWNLOAD", "1")
    with pytest.raises(SystemExit) as error:
        cache.main(["--mode", "reuse", "--source-root", str(tmp_path / "missing"),
                    "--output-root", str(tmp_path / "out"), "--manifest", str(tmp_path / "receipt.json"),
                    "--models", "model_a"])
    assert error.value.code == 1
    assert "source is missing" in capsys.readouterr().err
    assert not (tmp_path / "receipt.json").exists()
