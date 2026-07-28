from __future__ import annotations

from pathlib import Path
import importlib
import json
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyct.shap as shap_mod


def test_parse_args_accepts_required_dataset_options() -> None:
    args = shap_mod.parse_args(["--dataset", "mnist"])

    assert args.dataset == "mnist"
    assert args.first_n == 100
    assert args.explainer_type == "gradient"
    assert args.output_root == "shap_target_class"


def test_importing_pyct_shap_does_not_import_task_builders() -> None:
    sys.modules.pop("tasks.builders.cifar10", None)
    sys.modules.pop("tasks.builders.fashion_mnist", None)
    sys.modules.pop("tasks.builders.mnist", None)

    reloaded = importlib.reload(shap_mod)

    assert "tasks.builders.cifar10" not in sys.modules
    assert "tasks.builders.fashion_mnist" not in sys.modules
    assert "tasks.builders.mnist" not in sys.modules
    assert hasattr(reloaded, "main")


def test_main_requires_dataset_before_heavy_imports() -> None:
    sys.modules.pop("tasks.builders.cifar10", None)
    sys.modules.pop("tasks.builders.fashion_mnist", None)
    sys.modules.pop("tasks.builders.mnist", None)

    with pytest.raises(SystemExit):
        shap_mod.main(["--first-n", "1"])

    assert "tasks.builders.cifar10" not in sys.modules
    assert "tasks.builders.fashion_mnist" not in sys.modules
    assert "tasks.builders.mnist" not in sys.modules


def _install_fake_builder_modules(monkeypatch, calls):
    def _make_module(name: str) -> ModuleType:
        module = ModuleType(name)

        def _handler(model_name, **kwargs):
            calls.append((name, model_name, kwargs))
            return [
                {
                    "idx": 0,
                    "was_cached": False,
                    "computed": True,
                    "compute_seconds": 2.0,
                    "output_path": f"{name}.json",
                },
                {
                    "idx": 1,
                    "was_cached": True,
                    "computed": False,
                    "compute_seconds": 0.0,
                    "output_path": f"{name}_cached.json",
                }
            ]

        if name.endswith("fashion_mnist"):
            module.fashion_mnist_transformer_shap_calculate_all = _handler
        elif name.endswith("mnist"):
            module.mnist_transformer_shap_calculate_all = _handler
        else:
            module.cifar10_cal_shap_specs = _handler
        return module

    monkeypatch.setitem(sys.modules, "tasks.builders.mnist", _make_module("tasks.builders.mnist"))
    monkeypatch.setitem(
        sys.modules,
        "tasks.builders.fashion_mnist",
        _make_module("tasks.builders.fashion_mnist"),
    )
    monkeypatch.setitem(sys.modules, "tasks.builders.cifar10", _make_module("tasks.builders.cifar10"))


def test_main_rejects_first_n_lt_1_before_heavy_imports() -> None:
    sys.modules.pop("tasks.builders.cifar10", None)
    sys.modules.pop("tasks.builders.fashion_mnist", None)
    sys.modules.pop("tasks.builders.mnist", None)

    with pytest.raises(ValueError, match="--first-n must be >= 1"):
        shap_mod.main(["--dataset", "mnist", "--first-n", "0"])

    assert "tasks.builders.cifar10" not in sys.modules
    assert "tasks.builders.fashion_mnist" not in sys.modules
    assert "tasks.builders.mnist" not in sys.modules


def test_main_rejects_negative_sleep_before_heavy_imports() -> None:
    sys.modules.pop("tasks.builders.cifar10", None)
    sys.modules.pop("tasks.builders.fashion_mnist", None)
    sys.modules.pop("tasks.builders.mnist", None)

    with pytest.raises(ValueError, match="--sleep-seconds must be non-negative"):
        shap_mod.main(["--dataset", "mnist", "--sleep-seconds", "-1"])

    assert "tasks.builders.cifar10" not in sys.modules
    assert "tasks.builders.fashion_mnist" not in sys.modules
    assert "tasks.builders.mnist" not in sys.modules


def test_main_sets_background_env_vars_before_dispatch(monkeypatch, capsys) -> None:
    calls = []
    _install_fake_builder_modules(monkeypatch, calls)
    monkeypatch.setattr(shap_mod.time, "sleep", lambda *_args, **_kwargs: None)

    shap_mod.main(
        [
            "--dataset",
            "mnist",
            "--model-name",
            "demo",
            "--first-n",
            "2",
            "--background-per-class",
            "5",
            "--background-seed",
            "77",
            "--sleep-seconds",
            "0",
            "--force-refresh",
        ]
    )

    captured = capsys.readouterr()
    assert calls == [
        (
            "tasks.builders.mnist",
            "demo",
            {
                "first_n_img": 2,
                "force_refresh": True,
                "explainer_type": "gradient",
                "output_root": "shap_target_class",
            },
        )
    ]
    assert shap_mod.os.environ["PYCT_BG_PER_CLASS"] == "5"
    assert shap_mod.os.environ["PYCT_BG_SEED"] == "77"
    assert "processed inputs: 2" in captured.out
    summary = _extract_summary_json(captured.out)
    assert summary == {
        "cached": 1,
        "computed": 1,
        "attribution_target": "original_prediction",
        "dataset": "mnist",
        "explainer_type": "gradient",
        "max_seconds": 2.0,
        "mean_seconds": 2.0,
        "median_seconds": 2.0,
        "min_seconds": 2.0,
        "model": "demo",
        "output_root": "shap_target_class",
        "schema_version": 2,
        "total_compute_seconds": 2.0,
        "total_inputs": 2,
    }


def test_main_dispatches_cifar10_handler_with_expected_kwargs(monkeypatch) -> None:
    calls = []
    _install_fake_builder_modules(monkeypatch, calls)
    monkeypatch.setattr(shap_mod.time, "sleep", lambda *_args, **_kwargs: None)

    shap_mod.main(
        [
            "--dataset",
            "cifar10",
            "--model-name",
            "demo-cifar",
            "--first-n",
            "3",
            "--explainer-type",
            "kernel",
            "--output-root",
            "artifacts",
            "--sleep-seconds",
            "0",
        ]
    )

    assert calls == [
        (
            "tasks.builders.cifar10",
            "demo-cifar",
            {
                "first_n_img": 3,
                "force_refresh": False,
                "explainer_type": "kernel",
                "output_root": "artifacts",
            },
        )
    ]


def _extract_summary_json(output: str) -> dict:
    for line in output.splitlines():
        if line.startswith("[SHAP-SUMMARY-JSON] "):
            return json.loads(line.removeprefix("[SHAP-SUMMARY-JSON] "))
    raise AssertionError("missing SHAP summary JSON line")


def test_build_timing_summary_handles_all_cached_artifacts() -> None:
    summary = shap_mod.build_timing_summary(
        [
            {"idx": 0, "was_cached": True, "computed": False, "compute_seconds": 0.0},
            {"idx": 1, "was_cached": True, "computed": False, "compute_seconds": 0.0},
        ],
        model_name="demo",
        dataset="mnist",
        explainer_type="gradient",
        output_root="shap_value_all_layer",
    )

    assert summary["total_inputs"] == 2
    assert summary["computed"] == 0
    assert summary["cached"] == 2
    assert summary["total_compute_seconds"] == 0.0
    assert summary["mean_seconds"] is None
    assert summary["median_seconds"] is None
    assert summary["min_seconds"] is None
    assert summary["max_seconds"] is None
