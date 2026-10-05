from __future__ import annotations

from pathlib import Path
import pytest
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyct.args import parse_args


def test_parse_args_accepts_global_real_configuration() -> None:
    args = parse_args(
        [
            "--attack-mode",
            "global-real",
            "--dataset",
            "cifar10",
            "--score-alpha",
            "0.8",
            "--global-x-min",
            "-0.2",
            "--global-x-max",
            "0.05",
            "--global-x-bounds-mode",
            "strict",
            "--shap-sign-epsilon",
            "0.001",
            "--global-shift-kind",
            "contrast",
        ]
    )

    assert args.global_x_min == pytest.approx(-0.2)
    assert args.global_x_max == pytest.approx(0.05)
    assert args.global_x_bounds_mode == "strict"
    assert args.shap_sign_epsilon == pytest.approx(0.001)
    assert args.global_shift_kind == "contrast"


@pytest.mark.parametrize("shift_kind", ["brightness", "contrast", "aces-brightness", "aces-contrast"])
def test_parse_args_accepts_hybrid_de_for_global_real_kinds(shift_kind: str) -> None:
    args = parse_args(
        [
            "--attack-mode",
            "hybrid-de",
            "--dataset",
            "cifar10",
            "--score-alpha",
            "0.8",
            "--global-shift-kind",
            shift_kind,
        ]
    )

    assert args.attack_mode == "hybrid-de"
    assert args.global_shift_kind == shift_kind
    assert args.de_maxiter == 75
    assert args.de_population_size == 400


@pytest.mark.parametrize("shift_kind", ["brightness", "aces-brightness", "aces-contrast"])
def test_hybrid_margin_schedule_does_not_require_shap_score_alpha(shift_kind) -> None:
    args = parse_args(
        [
            "--attack-mode", "hybrid-de",
            "--dataset", "cifar10",
            "--global-shift-kind", shift_kind,
        ]
    )

    assert args.score_alpha is None


@pytest.mark.parametrize(
    "extra",
    [
        ["--dataset", "mnist"],
        ["--global-x-min", "0.01"],
        ["--global-x-max", "-0.01"],
        ["--shap-sign-epsilon", "-1"],
    ],
)
def test_parse_args_rejects_invalid_global_real_configuration(extra) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--attack-mode",
                "global-real",
                "--dataset",
                "cifar10",
                "--score-alpha",
                "0.8",
                *extra,
            ]
        )


@pytest.mark.parametrize(
    "extra",
    [
        ["--dataset", "mnist"],
        ["--global-shift-kind", "shap-sign"],
        ["--global-x-bounds-mode", "strict"],
        ["--global-x-min", "0.1", "--global-x-max", "0.1"],
        ["--de-population-size", "4"],
    ],
)
def test_parse_args_rejects_invalid_hybrid_de_configuration(extra) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--attack-mode",
                "hybrid-de",
                "--dataset",
                "cifar10",
                "--score-alpha",
                "0.8",
                "--global-shift-kind",
                "brightness",
                *extra,
            ]
        )


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
    assert args.ternary_fallback is False
    assert args.ternary_threshold_scale == pytest.approx(0.75)
    assert args.error_retry_limit == 2
    assert args.norm_01 is True


def test_parse_args_rejects_disabling_norm_01_for_image_datasets() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--attack-mode", "queue", "--dataset", "cifar10", "--no-norm-01"])


def test_parse_args_accepts_case_indices() -> None:
    args = parse_args(
        [
            "--attack-mode",
            "shap",
            "--score-alpha",
            "0.8",
            "--case-indices",
            "7,3,7,11",
        ]
    )

    assert args.case_indices == (7, 3, 11)


def test_parse_args_rejects_negative_case_indices() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--attack-mode",
                "shap",
                "--score-alpha",
                "0.8",
                "--case-indices",
                "1,-2",
            ]
        )


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


def test_parse_args_accepts_zero_ternary_threshold_scale() -> None:
    args = parse_args(
        [
            "--attack-mode",
            "queue",
            "--ternary-simplification",
            "--ternary-threshold-scale",
            "0",
        ]
    )

    assert args.ternary_simplification is True
    assert args.ternary_threshold_scale == pytest.approx(0.0)


def test_parse_args_accepts_ternary_fallback_for_queue() -> None:
    args = parse_args(["--attack-mode", "queue", "--ternary-fallback"])

    assert args.ternary_fallback is True
    assert args.ternary_simplification is False


def test_parse_args_accepts_ternary_fallback_for_shap() -> None:
    args = parse_args(["--attack-mode", "shap", "--score-alpha", "0.8", "--ternary-fallback"])

    assert args.ternary_fallback is True


def test_parse_args_rejects_combined_ternary_simplification_and_fallback() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--attack-mode",
                "queue",
                "--ternary-simplification",
                "--ternary-fallback",
            ]
        )


def test_parse_args_rejects_ternary_fallback_for_random_assign() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--attack-mode",
                "random-assign",
                "--score-alpha",
                "0.8",
                "--ternary-fallback",
            ]
        )


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


@pytest.mark.parametrize("value", ["-0.1"])
def test_parse_args_rejects_negative_ternary_threshold_scale(value: str) -> None:
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


def test_parse_args_defaults_aces_pwl_to_32_segments() -> None:
    args = parse_args(
        [
            "--attack-mode",
            "global-real",
            "--dataset",
            "cifar10",
            "--score-alpha",
            "0.8",
            "--global-shift-kind",
            "aces-brightness",
        ]
    )

    assert args.aces_pwl_max_segments == 32


def test_parse_args_rejects_aces_like_strict_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--attack-mode",
                "global-real",
                "--dataset",
                "cifar10",
                "--score-alpha",
                "0.8",
                "--global-shift-kind",
                "aces-contrast",
                "--global-x-bounds-mode",
                "strict",
            ]
        )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--global-x-min", "nan"),
        ("--global-x-max", "inf"),
        ("--aces-pwl-error-tolerance", "nan"),
        ("--aces-pwl-error-tolerance", "inf"),
        ("--aces-pwl-error-tolerance", "-inf"),
    ],
)
def test_parse_args_rejects_non_finite_aces_values(
    option: str, value: str
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--attack-mode",
                "global-real",
                "--dataset",
                "cifar10",
                "--score-alpha",
                "0.8",
                "--global-shift-kind",
                "aces-brightness",
                option,
                value,
            ]
        )


def test_parse_args_accepts_aces_like_global_real_configuration():
    args = parse_args([
        "--attack-mode", "global-real", "--dataset", "cifar10",
        "--score-alpha", "0.8", "--global-shift-kind", "aces-contrast",
        "--aces-pwl-max-segments", "12", "--aces-pwl-error-tolerance", "0.002",
    ])
    assert args.global_shift_kind == "aces-contrast"
    assert args.aces_pwl_max_segments == 12
    assert args.aces_pwl_error_tolerance == pytest.approx(0.002)


def test_parse_args_defaults_global_real_probe_controls() -> None:
    args = parse_args(
        [
            "--attack-mode",
            "global-real",
            "--dataset",
            "cifar10",
            "--score-alpha",
            "0.8",
            "--global-shift-kind",
            "aces-brightness",
        ]
    )

    assert args.global_real_probe is True
    assert args.global_real_probe_points == 17
    assert args.global_real_probe_refinements == 8
    assert args.global_real_probe_tolerance_fraction == pytest.approx(1.0 / 1024.0)


def test_global_shift_kind_help_mentions_aces_variants(capsys) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--help"])

    assert "aces-brightness" in capsys.readouterr().out
