import json
from types import SimpleNamespace

import numpy as np
import pytest

from explainability.input_shap_sign import TargetClassInputShap
from pyct import shap_sign_sweep


class _ThresholdModel:
    def predict(self, inputs, batch_size, verbose):
        del batch_size, verbose
        means = np.asarray(inputs).reshape(len(inputs), -1).mean(axis=1)
        return np.stack([1.0 - means, means], axis=1)


def test_build_shift_grid_includes_zero_and_both_bounds() -> None:
    grid = shap_sign_sweep.build_shift_grid(-0.08, 0.09, 0.05)

    assert grid.tolist() == pytest.approx([-0.08, -0.03, 0.0, 0.02, 0.07, 0.09])


@pytest.mark.parametrize("bounds_mode", ["strict", "clip"])
def test_evaluate_shift_direction_finds_smallest_label_flip(bounds_mode) -> None:
    result = shap_sign_sweep.evaluate_shift_direction(
        _ThresholdModel(),
        np.full((2, 1), 0.5, dtype=np.float32),
        np.ones((2, 1), dtype=np.int8),
        original_class=0,
        lower=-0.1,
        upper=0.1,
        step=0.05,
        batch_size=16,
        bounds_mode=bounds_mode,
    )

    assert result["successful"] is True
    assert result["best"]["shift"] == pytest.approx(0.05)
    assert result["best"]["label"] == 1
    assert result["best"]["clipped_count"] == 0


def test_evaluate_clip_mode_reports_clipped_values() -> None:
    result = shap_sign_sweep.evaluate_shift_direction(
        _ThresholdModel(),
        np.array([[0.95], [0.95]], dtype=np.float32),
        np.ones((2, 1), dtype=np.int8),
        original_class=1,
        lower=-0.1,
        upper=0.1,
        step=0.1,
        batch_size=16,
        bounds_mode="clip",
    )

    positive_endpoint = next(
        row for row in result["curve"] if row["shift"] == pytest.approx(0.1)
    )
    assert positive_endpoint["clipped_count"] == 2


def test_evaluate_shift_direction_rejects_zero_shift_label_mismatch() -> None:
    class _AlwaysOneModel:
        def predict(self, inputs, batch_size, verbose):
            del batch_size, verbose
            return np.tile([0.1, 0.9], (len(inputs), 1))

    with pytest.raises(ValueError, match="zero shift"):
        shap_sign_sweep.evaluate_shift_direction(
            _AlwaysOneModel(),
            np.full((1, 1), 0.5, dtype=np.float32),
            np.ones((1, 1), dtype=np.int8),
            original_class=0,
            lower=-0.1,
            upper=0.1,
            step=0.1,
            batch_size=4,
            bounds_mode="strict",
        )


def test_parse_args_rejects_duplicate_case_indices() -> None:
    with pytest.raises(SystemExit):
        shap_sign_sweep.parse_args(["--case-indices", "1,1"])


def test_parse_args_defaults_to_clip_and_canonical_cache() -> None:
    args = shap_sign_sweep.parse_args([])

    assert args.bounds_mode == "clip"
    assert args.shap_output_root == "shap_target_class"


@pytest.mark.parametrize(
    "bounds_mode, expected_interval",
    [
        ("strict", [-0.1, 0.0]),
        ("clip", [-0.1, 0.1]),
    ],
)
def test_run_sweep_writes_bounds_metadata(
    tmp_path,
    bounds_mode,
    expected_interval,
) -> None:
    class _Dataset:
        x_test = np.array([[[[0.1], [1.0]]]], dtype=np.float32)
        y_test = np.array([[1]], dtype=np.int64)

    class _Provider:
        def __init__(self):
            self.calls = []

        def ensure(self, **kwargs):
            self.calls.append(kwargs)
            return TargetClassInputShap(
                values=np.ones_like(kwargs["sample"]),
                target_class=kwargs["target_class"],
                cache_path=tmp_path / "target_shap.json",
                was_cached=True,
                metadata={"attribution_target": "original_prediction"},
            )

    provider = _Provider()
    args = shap_sign_sweep.parse_args(
        [
            "--case-indices",
            "0",
            "--shift-step",
            "0.05",
            "--bounds-mode",
            bounds_mode,
            "--output-dir",
            str(tmp_path / bounds_mode),
        ]
    )

    summary = shap_sign_sweep.run_sweep(
        args,
        dataset=_Dataset(),
        model=_ThresholdModel(),
        provider=provider,
    )

    assert summary["bounds_mode"] == bounds_mode
    assert summary["case_count"] == 1
    assert summary["successful_count"] == 1
    assert provider.calls[0]["target_class"] == 1
    case_payload = json.loads(
        (tmp_path / bounds_mode / "case_0.json").read_text(encoding="utf-8")
    )
    summary_payload = json.loads(
        (tmp_path / bounds_mode / "summary.json").read_text(encoding="utf-8")
    )
    assert case_payload["effective_interval"] == pytest.approx(expected_interval)
    assert case_payload["shap_attribution_target"] == "original_prediction"
    assert summary_payload["bounds_mode"] == bounds_mode


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"shift_min": 0.2, "shift_max": 0.1}, "shift-min"),
        ({"shift_step": 0.0}, "shift-step"),
        ({"bounds_mode": "invalid"}, "bounds-mode"),
        ({"shap_sign_epsilon": -1.0}, "shap-sign-epsilon"),
        ({"background_per_class": 0}, "background-per-class"),
        ({"batch_size": 0}, "batch-size"),
    ],
)
def test_run_sweep_validates_runtime_arguments(updates, message) -> None:
    args = vars(shap_sign_sweep.parse_args([]))
    args.update(updates)

    with pytest.raises(ValueError, match=message):
        shap_sign_sweep.run_sweep(
            SimpleNamespace(**args),
            dataset=object(),
            model=object(),
            provider=object(),
        )
