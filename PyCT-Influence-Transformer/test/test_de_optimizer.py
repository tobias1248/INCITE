from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import differential_evolution

from pyct.de.artifacts import (
    begin_de_artifact,
    load_de_artifact,
    mark_de_artifact_failed,
    write_de_artifact,
    write_generation_shard,
)
from pyct.de.objective import VectorizedOnePixelObjective, canonicalize_candidates
from pyct.de.optimizer import PopulationTraceRecorder, run_de_scout
from pyct.de.types import DeConfig


class _ThresholdModel:
    def __init__(self) -> None:
        self.batch_sizes = []

    def predict(self, batch, verbose=0):
        self.batch_sizes.append(len(batch))
        values = np.asarray(batch)[:, -1, -1, -1]
        return np.column_stack([values, 1.0 - values])


def test_mixed_integer_canonicalization_reaches_upper_coordinate_and_channel() -> None:
    canonical = canonicalize_candidates(
        np.array([[31.0, 31.0, 2.0, 1.0]]),
        (32, 32, 3),
        "mixed-integer",
    )

    assert canonical.tolist() == [[31.0, 31.0, 2.0, 1.0]]


def test_legacy_canonicalization_reproduces_truncation() -> None:
    canonical = canonicalize_candidates(
        np.array([[30.9, 30.9, 1.9, 254.9]]),
        (32, 32, 3),
        "legacy-truncate",
    )

    assert canonical[0, :3].tolist() == [30.0, 30.0, 1.0]
    assert canonical[0, 3] == pytest.approx(254.0 / 255.0)


def test_canonicalization_rejects_non_finite_candidates() -> None:
    with pytest.raises(ValueError, match="finite"):
        canonicalize_candidates(
            np.array([[0.0, 0.0, 0.0, np.nan]]),
            (32, 32, 3),
            "mixed-integer",
        )


def test_scipy_integrality_delivers_explicit_upper_bound_candidate() -> None:
    image = np.ones((32, 32, 3), dtype=np.float32)
    model = _ThresholdModel()
    recorder = PopulationTraceRecorder(8)
    objective = VectorizedOnePixelObjective(
        model=model,
        image=image,
        original_class=0,
        encoding="mixed-integer",
        objective="margin",
        observer=recorder.observe,
    )
    initial = np.array(
        [
            [31, 31, 2, 0.0],
            [0, 0, 0, 0.2],
            [1, 1, 1, 0.3],
            [2, 2, 2, 0.4],
            [3, 3, 0, 0.5],
            [4, 4, 1, 0.6],
            [5, 5, 2, 0.7],
            [6, 6, 0, 0.8],
        ],
        dtype=np.float64,
    )

    differential_evolution(
        objective,
        [(0, 31), (0, 31), (0, 2), (0.0, 1.0)],
        init=initial,
        maxiter=1,
        seed=4,
        polish=False,
        updating="deferred",
        integrality=[True, True, True, False],
        vectorized=True,
    )

    assert recorder.initial is not None
    assert [31.0, 31.0, 2.0] in recorder.initial.raw[:, :3].tolist()
    assert np.equal(recorder.initial.raw[:, :3], np.rint(recorder.initial.raw[:, :3])).all()


def test_run_de_scout_tracks_vectorized_population_and_is_deterministic() -> None:
    image = np.ones((2, 2, 3), dtype=np.float32)
    config = DeConfig(maxiter=2, population_size=8, seed=7)

    first = run_de_scout(_ThresholdModel(), image, config)
    second = run_de_scout(_ThresholdModel(), image, config)

    assert first.success is True
    assert first.completed_generations >= 1
    assert first.model_evaluations == 8 * (first.completed_generations + 1)
    assert first.auxiliary_model_evaluations == 2
    assert first.total_model_evaluations == first.model_evaluations + 2
    assert first.scipy_nfev == first.completed_generations + 1
    assert np.array_equal(first.best_raw, second.best_raw)
    assert np.array_equal(first.trace_arrays["accepted"], second.trace_arrays["accepted"])


def test_de_config_rejects_population_not_divisible_by_genome_size() -> None:
    with pytest.raises(ValueError, match="divisible by 4"):
        DeConfig(population_size=10).validate()


def test_de_artifact_round_trip_and_schema_validation(tmp_path: Path) -> None:
    model_path = tmp_path / "model.h5"
    model_path.write_bytes(b"model")
    result = run_de_scout(
        _ThresholdModel(),
        np.ones((2, 2, 3), dtype=np.float32),
        DeConfig(maxiter=1, population_size=8, seed=3),
    )
    case_dir = write_de_artifact(
        output_root=tmp_path / "out",
        model_name="demo",
        model_path=model_path,
        dataset="cifar10",
        case_index=4,
        result=result,
    )

    manifest, arrays = load_de_artifact(case_dir)

    assert manifest["case_index"] == 4
    assert manifest["config"]["encoding"] == "mixed-integer"
    assert manifest["auxiliary_model_evaluations"] == 2
    assert manifest["total_model_evaluations"] == manifest["model_evaluations"] + 2
    assert arrays["trial_raw"].shape[1:] == (8, 4)

    manifest_path = case_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["complete"] = False
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        load_de_artifact(case_dir)


def test_partial_artifact_remains_incomplete_after_failure(tmp_path: Path) -> None:
    model_path = tmp_path / "model.h5"
    model_path.write_bytes(b"model")
    config = DeConfig(maxiter=1, population_size=8)
    case_dir = begin_de_artifact(
        output_root=tmp_path / "out",
        model_name="demo",
        model_path=model_path,
        dataset="cifar10",
        case_index=2,
        config=config,
    )
    shard = write_generation_shard(case_dir, 0, {"initial_raw": np.zeros((8, 4))})
    mark_de_artifact_failed(case_dir, RuntimeError("boom"))

    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["status"] == "error"
    assert shard.is_file()
    with pytest.raises(ValueError, match="incomplete"):
        load_de_artifact(case_dir)

    (case_dir / "trace.npz").write_bytes(b"stale")
    restarted = begin_de_artifact(
        output_root=tmp_path / "out",
        model_name="demo",
        model_path=model_path,
        dataset="cifar10",
        case_index=2,
        config=config,
    )
    assert restarted == case_dir
    assert not shard.exists()
    assert not (case_dir / "trace.npz").exists()
