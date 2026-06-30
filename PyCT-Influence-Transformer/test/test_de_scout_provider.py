from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explainability.de_scout_provider import DeScoutPixelProvider


FIXTURE = ROOT / "test" / "fixtures" / "de_scout" / "cifar10_demo.json"


def _provider(path: Path = FIXTURE) -> DeScoutPixelProvider:
    return DeScoutPixelProvider(
        path=str(path),
        dataset="cifar10",
        model_name="demo_model",
        coordinate_bounds=(32, 32, 3),
    )


def _write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "de_scout.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_de_scout_provider_sorts_ranked_candidates_and_deduplicates() -> None:
    provider = _provider()

    assert provider.top_pixels(88, 2) == [(0, 1, 0), (4, 5, 2)]


def test_de_scout_provider_preserves_unranked_candidate_order() -> None:
    provider = _provider()

    assert provider.top_pixels(7, 2) == [(2, 3, 1), (8, 9, 0)]


def test_de_scout_provider_rejects_dataset_mismatch() -> None:
    with pytest.raises(ValueError, match="dataset mismatch"):
        DeScoutPixelProvider(
            path=str(FIXTURE),
            dataset="mnist",
            model_name="demo_model",
            coordinate_bounds=(32, 32, 3),
        )


def test_de_scout_provider_rejects_missing_case() -> None:
    with pytest.raises(ValueError, match="no candidate list for case 99"):
        _provider().top_pixels(99, 1)


def test_de_scout_provider_rejects_too_few_unique_candidates() -> None:
    with pytest.raises(ValueError, match="only 2 unique candidates"):
        _provider().top_pixels(88, 3)


def test_de_scout_provider_rejects_out_of_bounds_coord(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "dataset": "cifar10",
        "model_name": "demo_model",
        "candidates": {"0": [{"coord": [32, 0, 0]}]},
    }

    with pytest.raises(ValueError, match="out of bounds"):
        _provider(_write_payload(tmp_path, payload)).top_pixels(0, 1)


def test_de_scout_provider_rejects_non_integer_coord(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "dataset": "cifar10",
        "model_name": "demo_model",
        "candidates": {"0": [{"coord": [1.5, 0, 0]}]},
    }

    with pytest.raises(ValueError, match="must be an integer"):
        _provider(_write_payload(tmp_path, payload)).top_pixels(0, 1)
