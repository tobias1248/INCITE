from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from libct.branch_trace import BranchTraceEvent
from pyct.de.replay import BranchReplay


CHECKPOINT_SCHEMA_VERSION = 1


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _value_identity(value: float) -> tuple[str, str]:
    value_hex = float(value).hex()
    digest = hashlib.sha256(value_hex.encode("ascii")).hexdigest()
    return value_hex, digest


class ReplayCheckpointStore:
    def __init__(
        self,
        root: Path,
        fingerprint: Mapping[str, Any],
        *,
        force: bool = False,
    ) -> None:
        self.root = Path(root)
        self.state_path = self.root / "audit_state.json"
        self.replay_dir = self.root / "replays"
        self.fingerprint = json.loads(json.dumps(fingerprint, sort_keys=True))
        if force and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_initialize_state()

    def _load_or_initialize_state(self) -> Dict[str, Any]:
        if not self.state_path.is_file():
            state = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "fingerprint": self.fingerprint,
                "status": "running",
                "resume_count": 0,
            }
            _atomic_json(self.state_path, state)
            return state
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported audit checkpoint schema: "
                f"{state.get('schema_version')!r}"
            )
        if state.get("fingerprint") != self.fingerprint:
            raise ValueError(
                "Audit checkpoint configuration does not match this run; "
                "use a new output root or --force"
            )
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
        state["status"] = "running"
        _atomic_json(self.state_path, state)
        return state

    def _paths(self, value: float) -> tuple[str, Path, Path]:
        value_hex, identity = _value_identity(value)
        return (
            value_hex,
            self.replay_dir / f"{identity}.json",
            self.replay_dir / f"{identity}.npz",
        )

    def set_plan(self, values: list[float]) -> None:
        planned = [float(value).hex() for value in values]
        existing = self.state.get("planned_value_hex")
        if existing is not None and existing != planned:
            raise ValueError(
                "Audit checkpoint replay plan does not match this run; "
                "use a new output root or --force"
            )
        self.state["planned_value_hex"] = planned
        self.state["planned_replay_count"] = len(planned)
        self._refresh_progress()

    def _refresh_progress(self, *, last_value_hex: Optional[str] = None) -> None:
        complete_or_partial = 0
        errors = 0
        for value_hex in self.state.get("planned_value_hex", []):
            identity = hashlib.sha256(value_hex.encode("ascii")).hexdigest()
            metadata_path = self.replay_dir / f"{identity}.json"
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status = metadata.get("status")
            if status in {"complete", "partial"}:
                complete_or_partial += 1
            elif status == "error":
                errors += 1
        self.state["completed_replay_count"] = complete_or_partial
        self.state["error_replay_count"] = errors
        if last_value_hex is not None:
            self.state["last_completed_value_hex"] = last_value_hex
        _atomic_json(self.state_path, self.state)

    def load(self, value: float) -> Optional[BranchReplay]:
        value_hex, metadata_path, trace_path = self._paths(value)
        if not metadata_path.is_file() or not trace_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported replay checkpoint: {metadata_path}")
        if metadata.get("value_hex") != value_hex:
            raise ValueError(f"Replay checkpoint value mismatch: {metadata_path}")
        if metadata.get("status") not in {"complete", "partial"}:
            return None
        with np.load(trace_path, allow_pickle=False) as archive:
            site_digest = archive["site_digest"]
            outcomes = archive["observed_outcome"]
            depths = archive["depth"]
            positions = archive["position_json"]
        lengths = {len(site_digest), len(outcomes), len(depths), len(positions)}
        if len(lengths) != 1:
            raise ValueError(f"Replay checkpoint arrays have inconsistent lengths: {trace_path}")
        events = tuple(
            BranchTraceEvent(
                site_digest=str(site_digest[index]),
                observed_outcome=bool(outcomes[index]),
                depth=int(depths[index]),
                position=json.loads(str(positions[index])),
            )
            for index in range(len(site_digest))
        )
        return BranchReplay(
            events=events,
            complete=metadata["status"] == "complete",
            event_type=metadata.get("event_type"),
            duration_seconds=float(metadata.get("duration_seconds", 0.0)),
            timeout_seconds=(
                int(metadata["timeout_seconds"])
                if metadata.get("timeout_seconds") is not None
                else None
            ),
        )

    def save(self, value: float, replay: BranchReplay) -> None:
        value_hex, metadata_path, trace_path = self._paths(value)
        arrays = {
            "site_digest": np.asarray(
                [event.site_digest for event in replay.events], dtype="U64"
            ),
            "observed_outcome": np.asarray(
                [event.observed_outcome for event in replay.events], dtype=np.bool_
            ),
            "depth": np.asarray([event.depth for event in replay.events], dtype=np.int64),
            "position_json": np.asarray(
                [
                    json.dumps(event.position, sort_keys=True, separators=(",", ":"))
                    for event in replay.events
                ],
                dtype=str,
            ),
        }
        _atomic_npz(trace_path, arrays)
        _atomic_json(
            metadata_path,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "value": float(value),
                "value_hex": value_hex,
                "status": "complete" if replay.complete else "partial",
                "event_type": replay.event_type,
                "event_count": len(replay.events),
                "duration_seconds": float(replay.duration_seconds),
                "timeout_seconds": replay.timeout_seconds,
                "trace_file": trace_path.name,
            },
        )
        self._refresh_progress(last_value_hex=value_hex)

    def save_error(self, value: float, exc: BaseException) -> None:
        value_hex, metadata_path, _trace_path = self._paths(value)
        _atomic_json(
            metadata_path,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "value": float(value),
                "value_hex": value_hex,
                "status": "error",
                "error_type": exc.__class__.__name__,
                "error_reason": str(exc),
            },
        )
        self._refresh_progress()

    def mark_complete(self) -> None:
        self.state["status"] = "complete"
        _atomic_json(self.state_path, self.state)

    @property
    def resume_count(self) -> int:
        return int(self.state.get("resume_count", 0))


__all__ = ["CHECKPOINT_SCHEMA_VERSION", "ReplayCheckpointStore"]
