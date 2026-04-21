from __future__ import annotations

from pathlib import Path
import sys

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
        ROOT / 'exp' / 'mnist_model_only_first_forward_queue_10_0_a05_100_t0_0.75' / 'case_9'
    )


def test_get_repo_output_subdir_creates_directory_under_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)

    output_dir = paths.get_repo_output_subdir("popped_constraint_position", "demo_model", "shap_1")

    assert output_dir == tmp_path / "popped_constraint_position" / "demo_model" / "shap_1"
    assert output_dir.is_dir()
