from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.paths import get_save_dir_from_save_exp


def test_get_save_dir_from_save_exp_formats_expected_layout(monkeypatch):
    monkeypatch.setenv('PYCT_TIMEOUT', '1800')
    monkeypatch.setenv('PYCT_CONSTRAINT_BUILD_TIMEOUT_ENABLED', '1')
    monkeypatch.setenv('PYCT_CONSTRAINT_BUILD_TIMEOUT_SECONDS', '45')
    monkeypatch.setenv('PYCT_SCORE_ALPHA', '0.8')
    monkeypatch.setenv('PYCT_SYMBOLIC_PATH_THRESHOLD', '2000')

    save_dir = get_save_dir_from_save_exp(
        {'input_name': 'case_3', 'idx': 3},
        'transformer_fashion_mnist',
        'shap_solver60s',
    )

    assert save_dir == 'exp/transformer_fashion_mnist_shap_solver60s_1800_45_a08_2000/case_3'


def test_get_save_dir_from_save_exp_uses_only_first_forward_flag():
    save_dir = get_save_dir_from_save_exp(
        {'input_name': 'case_9', 'idx': 9},
        'mnist_model',
        'queue',
        only_first_forward=True,
        timeout=10,
        constraint_build_timeout=False,
        score_alpha=0.5,
        symbolic_path_threshold=100,
    )

    assert save_dir == 'exp/mnist_model_only_first_forward_queue_10_0_a05_100/case_9'
