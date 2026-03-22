from __future__ import annotations

from pathlib import Path
import pytest
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from start_cli import parse_args


def test_parse_args_accepts_patch_shap_for_cifar10_single_ton() -> None:
    args = parse_args(
        [
            "--attack-mode",
            "shap",
            "--dataset",
            "cifar10",
            "--pixel-search",
            "1",
            "--pixel-selector",
            "patch-shap",
            "--score-alpha",
            "0.8",
        ]
    )

    assert args.pixel_selector == "patch-shap"


def test_parse_args_rejects_patch_shap_for_non_cifar10() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--attack-mode",
                "shap",
                "--dataset",
                "mnist",
                "--pixel-search",
                "1",
                "--pixel-selector",
                "patch-shap",
                "--score-alpha",
                "0.8",
            ]
        )


def test_parse_args_rejects_patch_shap_for_non_shap_attack() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--attack-mode",
                "random",
                "--dataset",
                "cifar10",
                "--pixel-search",
                "1",
                "--pixel-selector",
                "patch-shap",
                "--score-alpha",
                "0.8",
            ]
        )


def test_parse_args_rejects_patch_shap_for_multi_ton() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--attack-mode",
                "shap",
                "--dataset",
                "cifar10",
                "--pixel-search",
                "1,2",
                "--pixel-selector",
                "patch-shap",
                "--score-alpha",
                "0.8",
            ]
        )
