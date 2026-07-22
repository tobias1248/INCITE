from __future__ import annotations

import json
from pathlib import Path

import pytest

from libct.branch_trace import BranchTraceEvent
from pyct.de.checkpoints import ReplayCheckpointStore
from pyct.de.replay import BranchReplay


def test_replay_checkpoint_round_trip_and_resume(tmp_path: Path) -> None:
    fingerprint = {"model_sha256": "abc", "case_index": 2, "replay_timeout": 60}
    store = ReplayCheckpointStore(tmp_path / "case_2", fingerprint)
    store.set_plan([0.25, 0.75])
    replay = BranchReplay(
        events=(BranchTraceEvent("a", True, 3, [1, [2, 3]]),),
        complete=False,
        event_type="soft_timeout",
        duration_seconds=60.1,
        timeout_seconds=60,
    )
    store.save(0.25, replay)

    resumed = ReplayCheckpointStore(tmp_path / "case_2", fingerprint)
    loaded = resumed.load(0.25)

    assert loaded == replay
    assert resumed.resume_count == 1
    state = json.loads((tmp_path / "case_2" / "audit_state.json").read_text())
    assert state["planned_replay_count"] == 2
    assert state["completed_replay_count"] == 1


def test_replay_checkpoint_rejects_incompatible_config(tmp_path: Path) -> None:
    ReplayCheckpointStore(tmp_path / "case_2", {"replay_timeout": 60})

    with pytest.raises(ValueError, match="does not match"):
        ReplayCheckpointStore(tmp_path / "case_2", {"replay_timeout": 1800})


def test_replay_checkpoint_rejects_incompatible_plan(tmp_path: Path) -> None:
    fingerprint = {"replay_timeout": 60}
    store = ReplayCheckpointStore(tmp_path / "case_2", fingerprint)
    store.set_plan([0.25])

    resumed = ReplayCheckpointStore(tmp_path / "case_2", fingerprint)
    with pytest.raises(ValueError, match="replay plan does not match"):
        resumed.set_plan([0.75])
