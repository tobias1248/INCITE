from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tasks.paths as paths


def test_get_save_dir_from_save_exp_formats_expected_layout(monkeypatch):
    monkeypatch.setenv('PYCT_TIMEOUT', '1800')
    monkeypatch.setenv('PYCT_CONSTRAINT_BUILD_TIMEOUT_ENABLED', '1')
    monkeypatch.setenv('PYCT_CONSTRAINT_BUILD_TIMEOUT_SECONDS', '45')
    monkeypatch.setenv('PYCT_SCORE_ALPHA', '0.8')
    monkeypatch.setenv('PYCT_SYMBOLIC_PATH_THRESHOLD', '2000')
    monkeypatch.setenv('PYCT_TERNARY_SIMPLIFICATION', '1')
    monkeypatch.setenv('PYCT_TERNARY_THRESHOLD_SCALE', '1.5')

    save_dir = paths.get_save_dir_from_save_exp(
        {'input_name': 'case_3', 'idx': 3},
        'transformer_fashion_mnist',
        'shap_solver60s',
    )

    assert save_dir == str(
        ROOT / 'exp' / 'transformer_fashion_mnist_shap_solver60s_1800_45_a08_2000_t1_1.5' / 'case_3'
    )


def test_get_save_dir_from_save_exp_uses_only_first_forward_flag():
    save_dir = paths.get_save_dir_from_save_exp(
        {'input_name': 'case_9', 'idx': 9},
        'mnist_model',
        'queue',
        only_first_forward=True,
        timeout=10,
        constraint_build_timeout=False,
        score_alpha=0.5,
        symbolic_path_threshold=100,
        ternary_simplification=False,
        ternary_threshold_scale=0.75,
    )

    assert save_dir == str(
        ROOT / 'exp' / 'mnist_model_only_first_forward_queue_10_0_a05_100_t0' / 'case_9'
    )


def test_get_save_dir_from_save_exp_omits_ternary_scale_when_disabled(monkeypatch):
    monkeypatch.setenv('PYCT_TIMEOUT', '1800')
    monkeypatch.setenv('PYCT_CONSTRAINT_BUILD_TIMEOUT_ENABLED', '1')
    monkeypatch.setenv('PYCT_CONSTRAINT_BUILD_TIMEOUT_SECONDS', '30')
    monkeypatch.setenv('PYCT_SCORE_ALPHA', '0.8')
    monkeypatch.setenv('PYCT_SYMBOLIC_PATH_THRESHOLD', '2000')
    monkeypatch.setenv('PYCT_TERNARY_SIMPLIFICATION', '0')
    monkeypatch.setenv('PYCT_TERNARY_THRESHOLD_SCALE', '0.75')

    save_dir = paths.get_save_dir_from_save_exp(
        {'input_name': 'case_0', 'idx': 0},
        'cifar10_concolic_transformer',
        'shap_solver60s',
    )

    assert save_dir == str(
        ROOT / 'exp' / 'cifar10_concolic_transformer_shap_solver60s_1800_30_a08_2000_t0' / 'case_0'
    )


def test_get_repo_output_subdir_creates_directory_under_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)

    output_dir = paths.get_repo_output_subdir("popped_constraint_position", "demo_model", "shap_1")

    assert output_dir == tmp_path / "popped_constraint_position" / "demo_model" / "shap_1"
    assert output_dir.is_dir()


def test_shared_serialization_uses_separate_output_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("PYCT_SMT_SERIALIZATION", raising=False)
    config = {"input_name": "case_3", "idx": 3}
    raw = Path(paths.get_save_dir_from_save_exp(config, "model", "queue"))
    monkeypatch.setenv("PYCT_SMT_SERIALIZATION", "shared")
    shared = Path(paths.get_save_dir_from_save_exp(config, "model", "queue"))
    assert shared.parent.name == raw.parent.name + "_smtshared"
    assert shared.name == raw.name == "case_3"


def test_output_directory_rejects_invalid_serialization_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("PYCT_SMT_SERIALIZATION", "typo")
    with pytest.raises(ValueError, match="PYCT_SMT_SERIALIZATION"):
        paths.get_save_dir_from_save_exp({"idx": 0}, "model", "queue")


@pytest.mark.parametrize("serialization", ["raw", "shared"])
def test_compact_representation_isolates_output_directories(tmp_path, monkeypatch, serialization):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("PYCT_SMT_SERIALIZATION", serialization)
    config = {"input_name": "case_3", "idx": 3}
    monkeypatch.setenv("PYCT_SYMBOLIC_REPRESENTATION", "legacy")
    legacy = Path(paths.get_save_dir_from_save_exp(config, "model", "queue"))
    monkeypatch.setenv("PYCT_SYMBOLIC_REPRESENTATION", "compact-affine")
    compact = Path(paths.get_save_dir_from_save_exp(config, "model", "queue"))
    assert compact.parent.name == legacy.parent.name + "_sym-affine"
    assert compact.name == legacy.name == "case_3"


def test_output_directory_rejects_invalid_symbolic_representation(monkeypatch):
    monkeypatch.setenv("PYCT_SYMBOLIC_REPRESENTATION", "typo")
    with pytest.raises(ValueError, match="PYCT_SYMBOLIC_REPRESENTATION"):
        paths.get_save_dir_from_save_exp({"idx": 0}, "model", "queue")
