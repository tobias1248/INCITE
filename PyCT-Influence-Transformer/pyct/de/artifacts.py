from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from pyct.de.types import DeRunResult


SCHEMA_VERSION = 2


@lru_cache(maxsize=16)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def begin_de_artifact(
    *,
    output_root: Path,
    model_name: str,
    model_path: Path,
    dataset: str,
    case_index: int,
    config: Any,
) -> Path:
    case_dir = output_root / model_name / f"case_{int(case_index)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(case_dir / ".partial", ignore_errors=True)
    (case_dir / "trace.npz").unlink(missing_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "model_name": model_name,
        "model_sha256": sha256_file(model_path),
        "case_index": int(case_index),
        "config": asdict(config),
        "complete": False,
        "status": "running",
    }
    _atomic_json(case_dir / "manifest.json", manifest)
    return case_dir


def write_generation_shard(
    case_dir: Path,
    generation: int,
    arrays: Dict[str, np.ndarray],
) -> Path:
    partial_dir = case_dir / ".partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    path = partial_dir / f"generation_{int(generation):03d}.npz"
    _atomic_npz(path, arrays)
    return path


def mark_de_artifact_failed(case_dir: Path, exc: BaseException) -> None:
    manifest_path = case_dir / "manifest.json"
    payload: Dict[str, Any] = {}
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "complete": False,
            "status": "error",
            "error_type": exc.__class__.__name__,
            "error_reason": str(exc),
        }
    )
    _atomic_json(manifest_path, payload)


def write_de_artifact(
    *,
    output_root: Path,
    model_name: str,
    model_path: Path,
    dataset: str,
    case_index: int,
    result: DeRunResult,
) -> Path:
    case_dir = output_root / model_name / f"case_{int(case_index)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    trace_path = case_dir / "trace.npz"
    _atomic_npz(trace_path, result.trace_arrays)
    config = asdict(result.config)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "model_name": model_name,
        "model_sha256": sha256_file(model_path),
        "case_index": int(case_index),
        "config": config,
        "original_class": int(result.original_class),
        "clean_probabilities": result.clean_probabilities.tolist(),
        "best_raw": result.best_raw.tolist(),
        "best_canonical": result.best_canonical.tolist(),
        "best_probabilities": result.best_probabilities.tolist(),
        "best_margin": float(result.best_margin),
        "predicted_class": int(result.predicted_class),
        "success": bool(result.success),
        "stop_reason": result.stop_reason,
        "duration_seconds": float(result.duration_seconds),
        "scipy_nfev": int(result.scipy_nfev),
        "model_evaluations": int(result.model_evaluations),
        "auxiliary_model_evaluations": int(result.auxiliary_model_evaluations),
        "total_model_evaluations": int(result.total_model_evaluations),
        "completed_generations": int(result.completed_generations),
        "trace_file": trace_path.name,
        "complete": True,
    }
    _atomic_json(case_dir / "manifest.json", manifest)
    shutil.rmtree(case_dir / ".partial", ignore_errors=True)
    return case_dir


def load_de_artifact(case_dir: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    manifest_path = case_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported DE trace schema_version {payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if payload.get("complete") is not True:
        raise ValueError(f"DE trace is incomplete: {case_dir}")
    trace_file = payload.get("trace_file")
    if not isinstance(trace_file, str):
        raise ValueError(f"DE trace manifest has no trace_file: {case_dir}")
    with np.load(case_dir / trace_file, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return payload, arrays


__all__ = [
    "SCHEMA_VERSION",
    "begin_de_artifact",
    "load_de_artifact",
    "mark_de_artifact_failed",
    "sha256_file",
    "write_de_artifact",
    "write_generation_shard",
]
