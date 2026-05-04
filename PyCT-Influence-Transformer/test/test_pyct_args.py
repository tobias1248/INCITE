from __future__ import annotations

from pathlib import Path
import pytest
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyct.args import parse_args


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


def test_parse_args_accepts_token_shap_for_cifar10_single_ton() -> None:
    args = parse_args(
        [
            "--attack-mode",
            "shap",
            "--dataset",
            "cifar10",
            "--pixel-search",
            "1",
            "--pixel-selector",
            "token-shap",
            "--score-alpha",
            "0.8",
        ]
    )

    assert args.pixel_selector == "token-shap"


def test_parse_args_rejects_token_shap_for_non_cifar10() -> None:
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
                "token-shap",
                "--score-alpha",
                "0.8",
            ]
        )


def test_parse_args_rejects_token_shap_for_non_shap_attack() -> None:
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
                "token-shap",
                "--score-alpha",
                "0.8",
            ]
        )


def test_parse_args_rejects_token_shap_for_multi_ton() -> None:
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
                "token-shap",
                "--score-alpha",
                "0.8",
            ]
        )


def test_parse_args_defaults_ternary_flags() -> None:
    args = parse_args(["--attack-mode", "queue"])

    assert args.ternary_simplification is False
    assert args.ternary_threshold_scale == pytest.approx(0.75)
    assert args.error_retry_limit == 2


def test_parse_args_accepts_custom_ternary_threshold_scale() -> None:
    args = parse_args(
        [
            "--attack-mode",
            "queue",
            "--ternary-simplification",
            "--ternary-threshold-scale",
            "1.5",
        ]
    )

    assert args.ternary_simplification is True
    assert args.ternary_threshold_scale == pytest.approx(1.5)


def test_parse_args_accepts_zero_error_retry_limit() -> None:
    args = parse_args(
        [
            "--attack-mode",
            "queue",
            "--error-retry-limit",
            "0",
        ]
    )

    assert args.error_retry_limit == 0


@pytest.mark.parametrize("value", ["0", "-0.1"])
def test_parse_args_rejects_non_positive_ternary_threshold_scale(value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--attack-mode",
                "queue",
                "--ternary-threshold-scale",
                value,
            ]
        )


def test_parse_args_rejects_negative_error_retry_limit() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--attack-mode",
                "queue",
                "--error-retry-limit",
                "-1",
            ]
        )
