from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyct.doctor as doctor_mod


def _present_spec(_module_name):
    return object()


def test_run_checks_passes_with_local_prerequisites(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "demo.h5").write_text("placeholder")
    dataset_cache = tmp_path / "keras" / "datasets"
    dataset_cache.mkdir(parents=True)
    shap_root = tmp_path / "shap_value_all_layer"
    (shap_root / "demo").mkdir(parents=True)
    output_dir = tmp_path / "exp"
    output_dir.mkdir()

    args = doctor_mod.parse_args(
        [
            "--solver",
            "cvc5",
            "--model-dir",
            str(model_dir),
            "--model-name",
            "demo",
            "--dataset-cache",
            str(dataset_cache),
            "--shap-root",
            str(shap_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    results = doctor_mod.run_checks(
        args,
        find_spec_fn=_present_spec,
        which_fn=lambda command: f"/usr/bin/{command}",
    )

    assert all(result.ok for result in results)
    assert {result.name for result in results} >= {
        "solver",
        "model",
        "dataset-cache",
        "shap-root",
        "output-dir",
    }


def test_main_returns_nonzero_when_required_solver_is_missing(tmp_path, capsys) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "demo.h5").write_text("placeholder")
    output_dir = tmp_path / "exp"
    output_dir.mkdir()

    exit_code = doctor_mod.main(
        [
            "--solver",
            "missing-solver",
            "--model-dir",
            str(model_dir),
            "--model-name",
            "demo",
            "--dataset-cache",
            str(tmp_path / "missing-cache"),
            "--shap-root",
            str(tmp_path / "missing-shap"),
            "--output-dir",
            str(output_dir),
            "--skip-runtime-packages",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[FAIL] solver" in captured.out
    assert "PyCT doctor failed." in captured.out


def test_main_json_output_is_machine_readable(tmp_path, monkeypatch, capsys) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "demo.h5").write_text("placeholder")
    dataset_cache = tmp_path / "keras" / "datasets"
    dataset_cache.mkdir(parents=True)
    output_dir = tmp_path / "exp"
    output_dir.mkdir()

    monkeypatch.setattr(doctor_mod.shutil, "which", lambda command: f"/usr/bin/{command}")

    exit_code = doctor_mod.main(
        [
            "--model-dir",
            str(model_dir),
            "--model-name",
            "demo",
            "--dataset-cache",
            str(dataset_cache),
            "--shap-root",
            str(tmp_path / "missing-shap"),
            "--output-dir",
            str(output_dir),
            "--skip-runtime-packages",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert any(check["name"] == "solver" for check in payload["checks"])


def test_missing_model_file_is_required_failure(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    result = doctor_mod.check_model_dir(model_dir, "missing")

    assert result.name == "model"
    assert result.status == "fail"
    assert not result.ok
