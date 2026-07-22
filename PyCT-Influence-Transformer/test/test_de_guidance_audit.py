from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from libct.branch_trace import BranchTraceEvent
from pyct.de.audit import (
    aggregate_branch_utility,
    audit_case,
    attribute_first_divergence,
    compare_branch_replays,
    ndcg,
    paired_bootstrap_lift,
    select_same_coordinate_pairs,
    select_same_coordinate_pairs_with_diagnostics,
)
from pyct.de.cli import main
from pyct.de.replay import BranchReplay


def _event(site: str, outcome: bool, depth: int = 1) -> BranchTraceEvent:
    return BranchTraceEvent(site, outcome, depth, [2, [3, 4]])


def test_first_divergence_credits_trial_outcome_only_at_same_site() -> None:
    transition = attribute_first_divergence(
        [_event("a", True), _event("b", False, 2)],
        [_event("a", True), _event("b", True, 2)],
        0.4,
    )

    assert transition is not None
    assert transition.transition_key == "b:1"
    assert transition.improvement == pytest.approx(0.4)

    assert attribute_first_divergence(
        [_event("a", True)],
        [_event("different", False)],
        0.4,
    ) is None


def test_pair_selection_filters_coordinate_value_and_generation() -> None:
    parent = np.zeros((75, 2, 4), dtype=np.float64)
    trial = np.zeros_like(parent)
    parent[:, :, :3] = [1, 2, 2]
    trial[:, :, :3] = [1, 2, 2]
    parent[:, :, 3] = 0.8
    trial[:, :, 3] = 0.2
    parent_energy = np.ones((75, 2), dtype=np.float64)
    trial_energy = np.full((75, 2), 0.5, dtype=np.float64)
    trial[10, 0, :3] = [9, 9, 0]
    arrays = {
        "parent_canonical": parent,
        "trial_canonical": trial,
        "parent_energy": parent_energy,
        "trial_energy": trial_energy,
    }

    selected = select_same_coordinate_pairs(
        arrays,
        (1, 2, 2),
        generation_start=1,
        generation_end=60,
        count=12,
    )

    assert len(selected) == 12
    assert all(1 <= pair.generation <= 60 for pair in selected)
    assert all(pair.improvement == pytest.approx(0.5) for pair in selected)
    assert selected == sorted(
        selected, key=lambda pair: (pair.generation, pair.candidate_index)
    )


def test_pair_selection_refills_missing_generation_bands() -> None:
    parent = np.tile([1.0, 2.0, 2.0, 0.8], (6, 3, 1))
    trial = np.tile([9.0, 9.0, 0.0, 0.2], (6, 3, 1))
    trial[4:, :, :3] = [1.0, 2.0, 2.0]
    selection = select_same_coordinate_pairs_with_diagnostics(
        {
            "parent_canonical": parent,
            "trial_canonical": trial,
            "parent_energy": np.ones((6, 3)),
            "trial_energy": np.zeros((6, 3)),
        },
        (1, 2, 2),
        generation_start=1,
        generation_end=6,
        count=3,
    )

    assert len(selection.refs) == 3
    assert selection.eligible_count == 6
    assert selection.band_selected_counts == (0, 0, 1)
    assert selection.refill_count == 2


def test_replay_comparison_distinguishes_attribution_and_censoring() -> None:
    parent = BranchReplay((_event("a", True), _event("b", False)), complete=False)
    trial = BranchReplay((_event("a", True), _event("b", True)), complete=False)

    attributed = compare_branch_replays(parent, trial, 0.4)
    censored = compare_branch_replays(
        BranchReplay((_event("a", True),), complete=False),
        BranchReplay((_event("a", True),), complete=True),
        0.4,
    )

    assert attributed.status == "attributed"
    assert attributed.transition is not None
    assert attributed.transition.transition_key == "b:1"
    assert censored.status == "censored_parent_partial"
    assert censored.transition is None


def test_branch_utility_uses_support_shrinkage() -> None:
    transitions = [
        attribute_first_divergence([_event("a", False)], [_event("a", True)], 0.6),
        attribute_first_divergence([_event("a", False)], [_event("a", True)], 0.4),
    ]

    utility = aggregate_branch_utility([item for item in transitions if item is not None])

    assert utility["a:1"]["support"] == 2
    assert utility["a:1"]["utility"] == pytest.approx(0.25)


def test_ndcg_and_paired_bootstrap_detect_better_ordering() -> None:
    relevance = [3.0, 2.0, 1.0]

    guided = ndcg(relevance, [3.0, 2.0, 1.0])
    baseline = ndcg(relevance, [1.0, 2.0, 3.0])
    lift = paired_bootstrap_lift([guided] * 12, [baseline] * 12, samples=500, seed=9)

    assert guided == pytest.approx(1.0)
    assert baseline is not None and baseline < guided
    assert lift["ci95_lower"] > 0.0


def test_audit_case_accepts_exact_divergence_from_partial_replays(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_00000"
    case_dir.mkdir()
    parent = np.tile([1.0, 2.0, 2.0, 0.8], (2, 2, 1))
    trial = np.tile([1.0, 2.0, 2.0, 0.2], (2, 2, 1))
    np.savez_compressed(
        case_dir / "trace.npz",
        parent_canonical=parent,
        trial_canonical=trial,
        parent_energy=np.ones((2, 2)),
        trial_energy=np.full((2, 2), 0.5),
    )
    (case_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "complete": True,
                "trace_file": "trace.npz",
                "case_index": 0,
                "model_sha256": "model-hash",
                "best_canonical": [1, 2, 2, 0.2],
            }
        ),
        encoding="utf-8",
    )

    def replay(_coordinate, value: float) -> BranchReplay:
        outcome = value < 0.5
        return BranchReplay((_event("site", outcome),), complete=False, event_type="soft_timeout")

    result = audit_case(
        case_dir=case_dir,
        replay=replay,
        shap_lookup=lambda _position: 0.0,
        train_pairs=1,
        holdout_pairs=1,
        train_end_generation=1,
        holdout_end_generation=2,
    )

    assert result["train_attributed_count"] == 1
    assert result["holdout_attributed_count"] == 1
    assert result["partial_replay_count"] == 2
    assert result["audit_config"]["train_end_generation"] == 1
    assert result["branch_utilities"]["site:1"]["utility"] == pytest.approx(1.0 / 6.0)


def test_audit_case_resumes_persisted_replays(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_00000"
    case_dir.mkdir()
    parent = np.tile([1.0, 2.0, 2.0, 0.8], (2, 1, 1))
    trial = np.tile([1.0, 2.0, 2.0, 0.2], (2, 1, 1))
    np.savez_compressed(
        case_dir / "trace.npz",
        parent_canonical=parent,
        trial_canonical=trial,
        parent_energy=np.ones((2, 1)),
        trial_energy=np.zeros((2, 1)),
    )
    (case_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "complete": True,
                "trace_file": "trace.npz",
                "case_index": 0,
                "model_sha256": "model-hash",
                "best_canonical": [1, 2, 2, 0.2],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def replay(_coordinate, value: float) -> BranchReplay:
        calls.append(value)
        return BranchReplay(
            (_event("site", value < 0.5),),
            complete=False,
            event_type="soft_timeout",
            duration_seconds=3.0,
            timeout_seconds=5,
        )

    kwargs = {
        "case_dir": case_dir,
        "replay": replay,
        "shap_lookup": lambda _position: 0.0,
        "train_pairs": 1,
        "holdout_pairs": 1,
        "train_end_generation": 1,
        "holdout_end_generation": 2,
        "checkpoint_dir": tmp_path / "audit" / "case_0",
        "replay_timeout": 5,
    }
    first = audit_case(**kwargs)
    second = audit_case(**kwargs)

    assert len(calls) == 2
    assert first["replay_summary"]["checkpoint_hits"] == 0
    assert second["replay_summary"]["checkpoint_hits"] == 2
    assert second["replay_summary"]["resume_count"] == 1
    assert second["pair_diagnostics"] == first["pair_diagnostics"]


def test_gate_cli_requires_all_conditions_and_writes_decision(tmp_path: Path) -> None:
    models = ["m1", "m2", "m3", "m4", "m5"]
    for model in models:
        model_dir = tmp_path / model
        model_dir.mkdir()
        payload = {
            "data_sufficient": True,
            "lift_vs_shap": {"point": 0.4},
            "lift_vs_path": {"point": 0.3},
            "case_metrics": [
                {"case_index": index, "de": 1.0, "shap": 0.4, "path": 0.5}
                for index in range(10)
            ],
        }
        (model_dir / "audit.json").write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "decision.json"

    result = main(
        [
            "gate",
            "--audit-root",
            str(tmp_path),
            "--models",
            *models,
            "--bootstrap-samples",
            "500",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8"))["decision"] == "go"
