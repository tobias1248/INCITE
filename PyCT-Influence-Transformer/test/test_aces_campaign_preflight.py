from __future__ import annotations

import hashlib
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from orchestration import aces_campaign_preflight as campaign


CONTROLS = ['aces-exposure', 'aces-highlights', 'aces-shadows', 'aces-saturation']


def test_real_pwl_checks_every_control_case_within_requested_limits():
    images = np.array([[[[0.1, 0.4, 0.7]]], [[[0.8, 0.5, 0.2]]]], dtype=np.float32)
    report = campaign.preflight(images, CONTROLS, first_n=2)
    assert report['complete'] is True
    assert report['checked_count'] == report['expected_count'] == 8
    assert report['rejected_count'] == 0
    assert {(row['control'], row['case_id']) for row in report['rows']} == {
        (control, index) for control in CONTROLS for index in range(2)
    }
    assert all(row['segments'] <= 64 and row['max_abs_error'] <= 1 / 255
               for row in report['rows'])


@pytest.mark.parametrize('settings', [
    {'first_n': 0}, {'first_n': 3}, {'first_n': True},
    {'max_segments': 0}, {'max_segments': 2.5},
    {'x_min': 0.1}, {'x_min': 0.2, 'x_max': 0.2}, {'x_max': float('nan')},
    {'error_tolerance': 0}, {'error_tolerance': float('inf')},
])
def test_invalid_settings_fail_before_any_pwl_call(monkeypatch, settings):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid settings reached PWL construction')
    monkeypatch.setattr(campaign, 'build_adaptive_pwl_approximation', forbidden)
    arguments = {'first_n': 2, **settings}
    with pytest.raises(ValueError):
        campaign.preflight(np.zeros((2, 1, 1, 3)), CONTROLS, **arguments)


@pytest.mark.parametrize('controls', [[], ['aces-exposure', 'aces-exposure'], ['unsupported']])
def test_invalid_controls_fail_before_pwl(monkeypatch, controls):
    monkeypatch.setattr(campaign, 'build_adaptive_pwl_approximation',
                        lambda *args, **kwargs: pytest.fail('PWL should not run'))
    with pytest.raises(ValueError):
        campaign.preflight(np.zeros((1, 1, 1, 3)), controls, first_n=1)


def _cache(tmp_path):
    cache = tmp_path / 'test_batch'
    pixels = np.arange(3072, dtype=np.uint16).astype(np.uint8)[None, :]
    cache.write_bytes(pickle.dumps({b'data': pixels}))
    return cache, pixels


def test_cli_normalizes_uint8_float32_then_passes_float64_and_records_identity(tmp_path, monkeypatch):
    cache, pixels = _cache(tmp_path)
    received = []
    def build(image, **settings):
        received.append(image)
        return SimpleNamespace(segment_count=2, max_abs_error=0.001)
    monkeypatch.setattr(campaign, 'build_adaptive_pwl_approximation', build)
    output = tmp_path / 'out' / 'report.json'
    assert campaign.main(['--controls', 'aces-exposure', '--first-n', '1',
                          '--dataset-cache', str(cache), '--output', str(output)]) == 0
    expected = (pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                .astype(np.float32) / 255).astype(np.float64)[0]
    assert received[0].dtype == np.float64
    np.testing.assert_array_equal(received[0], expected)
    report = json.loads(output.read_text())
    assert report['dataset_cache'] == str(cache.resolve())
    assert report['dataset_sha256'] == hashlib.sha256(cache.read_bytes()).hexdigest()


def test_cli_records_failed_cases_and_nonzero_result_without_relaxing_budget(tmp_path, monkeypatch):
    cache, _ = _cache(tmp_path)
    calls = []
    def fail(image, **settings):
        calls.append(settings)
        raise campaign.AcesLikeTransformError('cap cannot meet tolerance')
    monkeypatch.setattr(campaign, 'build_adaptive_pwl_approximation', fail)
    output = tmp_path / 'rejected.json'
    assert campaign.main(['--controls', 'aces-exposure', 'aces-shadows', '--first-n', '1',
                          '--dataset-cache', str(cache), '--output', str(output)]) == 1
    report = json.loads(output.read_text())
    assert report['complete'] is False
    assert report['rejected_count'] == report['checked_count'] == 2
    assert all('cap cannot meet tolerance' in row['error'] for row in report['rows'])
    assert all(call['max_segments'] == 64 and call['error_tolerance'] == 1 / 255
               for call in calls)


def test_cli_uses_only_local_default_cache_and_missing_cache_writes_nothing(tmp_path, monkeypatch):
    import datasets.keras_cache
    monkeypatch.setattr(datasets.keras_cache, 'get_keras_datasets_dir', lambda: tmp_path)
    monkeypatch.setenv('PYCT_ALLOW_DATASET_DOWNLOAD', '1')
    output = tmp_path / 'report.json'
    with pytest.raises(FileNotFoundError):
        campaign.main(['--controls', 'aces-exposure', '--first-n', '1', '--output', str(output)])
    assert not output.exists()


def test_cli_refuses_output_reuse_before_loading_dataset(tmp_path):
    output = tmp_path / 'report.json'
    output.write_text('preserved')
    with pytest.raises(SystemExit):
        campaign.main(['--controls', 'aces-exposure', '--output', str(output),
                       '--dataset-cache', str(tmp_path / 'missing')])
    assert output.read_text() == 'preserved'


@pytest.mark.parametrize('result', [
    SimpleNamespace(segment_count=65, max_abs_error=0.001),
    SimpleNamespace(segment_count=2, max_abs_error=0.1),
    SimpleNamespace(segment_count=2, max_abs_error=float('nan')),
])
def test_preflight_rejects_builder_result_outside_limits(monkeypatch, result):
    monkeypatch.setattr(campaign, 'build_adaptive_pwl_approximation', lambda *a, **kw: result)
    report = campaign.preflight(np.zeros((1, 1, 1, 3)), ['aces-exposure'], first_n=1)
    assert not report['complete']
    assert report['rows'][0]['accepted'] is False


def test_cli_serializes_rejected_nonfinite_metric_as_valid_failure_report(tmp_path, monkeypatch):
    cache, _ = _cache(tmp_path)
    monkeypatch.setattr(campaign, 'build_adaptive_pwl_approximation', lambda *a, **kw:
                        SimpleNamespace(segment_count=2, max_abs_error=float('nan')))
    output = tmp_path / 'report.json'
    assert campaign.main(['--controls', 'aces-exposure', '--first-n', '1',
                          '--dataset-cache', str(cache), '--output', str(output)]) == 1
    report = json.loads(output.read_text(), parse_constant=lambda value: pytest.fail('nonfinite JSON'))
    assert report['complete'] is False
    assert report['rejected_count'] == 1
