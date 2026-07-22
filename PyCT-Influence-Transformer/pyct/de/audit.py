from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from libct.branch_trace import BranchTraceEvent
from pyct.de.artifacts import load_de_artifact, sha256_file
from pyct.de.checkpoints import ReplayCheckpointStore
from pyct.de.replay import BranchReplay


TraceReplay = Callable[[Sequence[int], float], BranchReplay]
ShapLookup = Callable[[Any], float]
log = logging.getLogger("ct.de")


@dataclass(frozen=True)
class PairRef:
    generation: int
    candidate_index: int
    parent_value: float
    trial_value: float
    improvement: float


@dataclass(frozen=True)
class AttributedTransition:
    transition_key: str
    improvement: float
    depth: int
    position: Any


@dataclass(frozen=True)
class PairSelection:
    refs: Tuple[PairRef, ...]
    eligible_count: int
    band_selected_counts: Tuple[int, ...]
    refill_count: int


@dataclass(frozen=True)
class AttributionResult:
    status: str
    transition: Optional[AttributedTransition]
    compared_event_count: int


def _partition_ranges(start: int, end: int, parts: int = 3) -> List[Tuple[int, int]]:
    values = np.array_split(np.arange(start, end + 1), parts)
    return [(int(part[0]), int(part[-1])) for part in values if len(part)]


def select_same_coordinate_pairs(
    arrays: Mapping[str, np.ndarray],
    coordinate: Sequence[int],
    *,
    generation_start: int,
    generation_end: int,
    count: int,
) -> List[PairRef]:
    return list(
        select_same_coordinate_pairs_with_diagnostics(
            arrays,
            coordinate,
            generation_start=generation_start,
            generation_end=generation_end,
            count=count,
        ).refs
    )


def select_same_coordinate_pairs_with_diagnostics(
    arrays: Mapping[str, np.ndarray],
    coordinate: Sequence[int],
    *,
    generation_start: int,
    generation_end: int,
    count: int,
) -> PairSelection:
    if count < 0:
        raise ValueError("pair count must be non-negative")
    parent = np.asarray(arrays["parent_canonical"])
    trial = np.asarray(arrays["trial_canonical"])
    parent_energy = np.asarray(arrays["parent_energy"])
    trial_energy = np.asarray(arrays["trial_energy"])
    coord = np.asarray(tuple(int(part) for part in coordinate), dtype=np.int64)
    if parent.ndim != 3 or parent.shape[-1] != 4 or trial.shape != parent.shape:
        raise ValueError("DE trace parent/trial canonical arrays must have shape (G, P, 4)")
    if generation_end > parent.shape[0]:
        generation_end = parent.shape[0]
    if generation_start < 1 or generation_start > generation_end or count == 0:
        return PairSelection((), 0, (), 0)

    ranges = _partition_ranges(generation_start, generation_end)
    base, remainder = divmod(count, len(ranges))
    selected: List[PairRef] = []
    eligible_by_band: List[List[PairRef]] = []
    band_selected_counts: List[int] = []
    for partition_index, (range_start, range_end) in enumerate(ranges):
        partition_count = base + (1 if partition_index < remainder else 0)
        eligible: List[PairRef] = []
        for generation in range(range_start, range_end + 1):
            generation_index = generation - 1
            for candidate_index in range(parent.shape[1]):
                parent_candidate = parent[generation_index, candidate_index]
                trial_candidate = trial[generation_index, candidate_index]
                if not np.array_equal(parent_candidate[:3].astype(np.int64), coord):
                    continue
                if not np.array_equal(trial_candidate[:3].astype(np.int64), coord):
                    continue
                parent_value = float(parent_candidate[3])
                trial_value = float(trial_candidate[3])
                if not np.isfinite([parent_value, trial_value]).all() or np.isclose(
                    parent_value, trial_value
                ):
                    continue
                improvement = float(
                    parent_energy[generation_index, candidate_index]
                    - trial_energy[generation_index, candidate_index]
                )
                if not math.isfinite(improvement) or improvement <= 0.0:
                    continue
                eligible.append(
                    PairRef(
                        generation=generation,
                        candidate_index=candidate_index,
                        parent_value=parent_value,
                        trial_value=trial_value,
                        improvement=improvement,
                    )
                )
        eligible.sort(key=lambda pair: (-pair.improvement, pair.generation, pair.candidate_index))
        eligible_by_band.append(eligible)
        band_selected = eligible[:partition_count]
        selected.extend(band_selected)
        band_selected_counts.append(len(band_selected))

    selected_keys = {(pair.generation, pair.candidate_index) for pair in selected}
    remaining = sorted(
        (
            pair
            for eligible in eligible_by_band
            for pair in eligible
            if (pair.generation, pair.candidate_index) not in selected_keys
        ),
        key=lambda pair: (-pair.improvement, pair.generation, pair.candidate_index),
    )
    refill = remaining[: max(0, count - len(selected))]
    selected.extend(refill)
    return PairSelection(
        refs=tuple(sorted(selected, key=lambda pair: (pair.generation, pair.candidate_index))),
        eligible_count=sum(len(eligible) for eligible in eligible_by_band),
        band_selected_counts=tuple(band_selected_counts),
        refill_count=len(refill),
    )


def attribute_first_divergence(
    parent_trace: Sequence[BranchTraceEvent],
    trial_trace: Sequence[BranchTraceEvent],
    improvement: float,
) -> Optional[AttributedTransition]:
    if improvement <= 0.0 or not math.isfinite(improvement):
        return None
    for parent_event, trial_event in zip(parent_trace, trial_trace):
        if parent_event.site_digest != trial_event.site_digest:
            return None
        if parent_event.observed_outcome == trial_event.observed_outcome:
            continue
        return AttributedTransition(
            transition_key=trial_event.transition_key,
            improvement=float(improvement),
            depth=int(trial_event.depth),
            position=trial_event.position,
        )
    return None


def compare_branch_replays(
    parent_replay: BranchReplay,
    trial_replay: BranchReplay,
    improvement: float,
) -> AttributionResult:
    if improvement <= 0.0 or not math.isfinite(improvement):
        return AttributionResult("invalid_improvement", None, 0)
    for index, (parent_event, trial_event) in enumerate(
        zip(parent_replay.events, trial_replay.events)
    ):
        if parent_event.site_digest != trial_event.site_digest:
            return AttributionResult("site_mismatch", None, index + 1)
        if parent_event.observed_outcome == trial_event.observed_outcome:
            continue
        transition = AttributedTransition(
            transition_key=trial_event.transition_key,
            improvement=float(improvement),
            depth=int(trial_event.depth),
            position=trial_event.position,
        )
        return AttributionResult("attributed", transition, index + 1)
    compared = min(len(parent_replay.events), len(trial_replay.events))
    if parent_replay.complete and trial_replay.complete:
        status = "no_divergence_complete"
    elif not parent_replay.complete and not trial_replay.complete:
        status = "censored_both_partial"
    elif not parent_replay.complete:
        status = "censored_parent_partial"
    else:
        status = "censored_trial_partial"
    return AttributionResult(status, None, compared)


def aggregate_branch_utility(
    transitions: Iterable[AttributedTransition],
    *,
    shrinkage: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    if shrinkage < 0:
        raise ValueError("shrinkage must be non-negative")
    grouped: Dict[str, Dict[str, Any]] = {}
    for transition in transitions:
        item = grouped.setdefault(
            transition.transition_key,
            {
                "support": 0,
                "improvement_sum": 0.0,
                "depths": [],
                "position": transition.position,
            },
        )
        item["support"] += 1
        item["improvement_sum"] += float(transition.improvement)
        item["depths"].append(int(transition.depth))
    for item in grouped.values():
        item["utility"] = item["improvement_sum"] / (item["support"] + shrinkage)
        item["median_depth"] = float(np.median(item.pop("depths")))
    return grouped


def ndcg(relevance: Sequence[float], scores: Sequence[float], k: int = 6) -> Optional[float]:
    rel = np.asarray(relevance, dtype=np.float64)
    predicted = np.asarray(scores, dtype=np.float64)
    if rel.size < 2 or rel.shape != predicted.shape or not np.isfinite(rel).all():
        return None
    predicted = np.nan_to_num(predicted, nan=-np.inf)
    limit = min(int(k), rel.size)
    discounts = np.log2(np.arange(2, limit + 2))
    order = np.argsort(-predicted, kind="stable")[:limit]
    ideal = np.argsort(-rel, kind="stable")[:limit]
    dcg = float(np.sum(rel[order] / discounts))
    ideal_dcg = float(np.sum(rel[ideal] / discounts))
    if ideal_dcg <= 0.0:
        return None
    return dcg / ideal_dcg


def paired_bootstrap_lift(
    values: Sequence[float],
    baselines: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 2024,
) -> Dict[str, float]:
    guided = np.asarray(values, dtype=np.float64)
    baseline = np.asarray(baselines, dtype=np.float64)
    if guided.shape != baseline.shape or guided.size == 0:
        raise ValueError("paired bootstrap requires non-empty, equally sized samples")
    differences = guided - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    means = differences[indices].mean(axis=1)
    return {
        "point": float(differences.mean()),
        "ci95_lower": float(np.percentile(means, 2.5)),
        "ci95_upper": float(np.percentile(means, 97.5)),
    }


def audit_case(
    *,
    case_dir: Path,
    replay: TraceReplay,
    shap_lookup: ShapLookup,
    train_pairs: int = 12,
    holdout_pairs: int = 6,
    train_end_generation: int = 60,
    holdout_end_generation: int = 75,
    shrinkage: float = 2.0,
    checkpoint_dir: Optional[Path] = None,
    replay_timeout: Optional[int] = None,
    force_checkpoint: bool = False,
) -> Dict[str, Any]:
    manifest, arrays = load_de_artifact(case_dir)
    coordinate = tuple(int(value) for value in manifest["best_canonical"][:3])
    train_selection = select_same_coordinate_pairs_with_diagnostics(
        arrays,
        coordinate,
        generation_start=1,
        generation_end=train_end_generation,
        count=train_pairs,
    )
    holdout_selection = select_same_coordinate_pairs_with_diagnostics(
        arrays,
        coordinate,
        generation_start=train_end_generation + 1,
        generation_end=holdout_end_generation,
        count=holdout_pairs,
    )
    train_refs = train_selection.refs
    holdout_refs = holdout_selection.refs
    checkpoint_store: Optional[ReplayCheckpointStore] = None
    if checkpoint_dir is not None:
        if replay_timeout is None or replay_timeout < 1:
            raise ValueError("replay_timeout is required when checkpointing audit replays")
        trace_path = case_dir / str(manifest["trace_file"])
        checkpoint_store = ReplayCheckpointStore(
            checkpoint_dir,
            {
                "model_sha256": manifest["model_sha256"],
                "case_index": int(manifest["case_index"]),
                "coordinate": list(coordinate),
                "de_trace_sha256": sha256_file(trace_path),
                "replay_timeout": int(replay_timeout),
                "train_pairs": int(train_pairs),
                "holdout_pairs": int(holdout_pairs),
                "train_end_generation": int(train_end_generation),
                "holdout_end_generation": int(holdout_end_generation),
                "shrinkage": float(shrinkage),
            },
            force=force_checkpoint,
        )
    replay_cache: Dict[str, BranchReplay] = {}
    checkpoint_hits = 0
    replay_progress = 0
    planned_values: List[float] = []
    planned_keys = set()
    for ref in (*train_refs, *holdout_refs):
        for value in (ref.parent_value, ref.trial_value):
            key = float(value).hex()
            if key not in planned_keys:
                planned_keys.add(key)
                planned_values.append(float(value))
    if checkpoint_store is not None:
        checkpoint_store.set_plan(planned_values)

    def trace_for(value: float) -> BranchReplay:
        nonlocal checkpoint_hits, replay_progress
        key = float(value).hex()
        if key not in replay_cache:
            checkpoint = checkpoint_store.load(value) if checkpoint_store is not None else None
            if checkpoint is not None:
                replay_cache[key] = checkpoint
                checkpoint_hits += 1
            else:
                try:
                    replay_cache[key] = replay(coordinate, value)
                except Exception as exc:
                    if checkpoint_store is not None:
                        checkpoint_store.save_error(value, exc)
                    raise
                if checkpoint_store is not None:
                    checkpoint_store.save(value, replay_cache[key])
            replay_progress += 1
            item = replay_cache[key]
            log.info(
                "Audit replay case=%s replay=%s/%s status=%s events=%s duration=%.3fs checkpoint=%s",
                manifest["case_index"],
                replay_progress,
                len(planned_values),
                "complete" if item.complete else "partial",
                len(item.events),
                item.duration_seconds,
                checkpoint is not None,
            )
        return replay_cache[key]

    for planned_value in planned_values:
        trace_for(planned_value)

    def attribute(
        refs: Sequence[PairRef],
    ) -> Tuple[List[AttributedTransition], List[Dict[str, Any]], Dict[str, int]]:
        attributed: List[AttributedTransition] = []
        diagnostics: List[Dict[str, Any]] = []
        status_counts: Dict[str, int] = {}
        for ref in refs:
            parent_replay = trace_for(ref.parent_value)
            trial_replay = trace_for(ref.trial_value)
            result = compare_branch_replays(
                parent_replay,
                trial_replay,
                ref.improvement,
            )
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
            if result.transition is not None:
                attributed.append(result.transition)
            diagnostics.append(
                {
                    "generation": int(ref.generation),
                    "candidate_index": int(ref.candidate_index),
                    "parent_value": float(ref.parent_value),
                    "trial_value": float(ref.trial_value),
                    "improvement": float(ref.improvement),
                    "status": result.status,
                    "compared_event_count": int(result.compared_event_count),
                    "transition_key": (
                        result.transition.transition_key
                        if result.transition is not None
                        else None
                    ),
                }
            )
        return attributed, diagnostics, status_counts

    train_transitions, train_diagnostics, train_status_counts = attribute(train_refs)
    utilities = aggregate_branch_utility(train_transitions, shrinkage=shrinkage)
    holdout_transitions, holdout_diagnostics, holdout_status_counts = attribute(holdout_refs)
    holdout_rows = []
    for transition in holdout_transitions:
        utility = utilities.get(transition.transition_key, {}).get("utility", 0.0)
        shap_score = float(shap_lookup(transition.position))
        if not math.isfinite(shap_score):
            shap_score = -1e300
        holdout_rows.append(
            {
                "transition_key": transition.transition_key,
                "relevance": transition.improvement,
                "de_score": float(utility),
                "shap_score": shap_score,
                "path_score": -float(transition.depth),
                "depth": int(transition.depth),
            }
        )
    relevance = [row["relevance"] for row in holdout_rows]
    metrics = {
        name: ndcg(relevance, [row[f"{name}_score"] for row in holdout_rows])
        for name in ("de", "shap", "path")
    }
    if checkpoint_store is not None:
        checkpoint_store.mark_complete()
    replay_values = list(replay_cache.values())
    event_counts = [len(item.events) for item in replay_values]
    replay_durations = [float(item.duration_seconds) for item in replay_values]

    def selection_payload(selection: PairSelection) -> Dict[str, Any]:
        return {
            "eligible_count": int(selection.eligible_count),
            "selected_count": len(selection.refs),
            "band_selected_counts": list(selection.band_selected_counts),
            "refill_count": int(selection.refill_count),
        }

    return {
        "schema_version": 2,
        "case_index": int(manifest["case_index"]),
        "model_sha256": manifest["model_sha256"],
        "coordinate": list(coordinate),
        "audit_config": {
            "train_pairs": int(train_pairs),
            "holdout_pairs": int(holdout_pairs),
            "train_end_generation": int(train_end_generation),
            "holdout_end_generation": int(holdout_end_generation),
            "shrinkage": float(shrinkage),
            "replay_timeout": int(replay_timeout) if replay_timeout is not None else None,
        },
        "train_pair_count": len(train_refs),
        "train_attributed_count": len(train_transitions),
        "holdout_pair_count": len(holdout_refs),
        "holdout_attributed_count": len(holdout_transitions),
        "replay_count": len(replay_cache),
        "partial_replay_count": sum(not replay.complete for replay in replay_cache.values()),
        "replay_summary": {
            "complete_count": sum(item.complete for item in replay_values),
            "partial_count": sum(not item.complete for item in replay_values),
            "total_duration_seconds": float(sum(replay_durations)),
            "median_duration_seconds": (
                float(np.median(replay_durations)) if replay_durations else None
            ),
            "event_count_min": min(event_counts) if event_counts else None,
            "event_count_median": float(np.median(event_counts)) if event_counts else None,
            "event_count_max": max(event_counts) if event_counts else None,
            "checkpoint_hits": int(checkpoint_hits),
            "resume_count": (
                checkpoint_store.resume_count if checkpoint_store is not None else 0
            ),
        },
        "pair_selection": {
            "train": selection_payload(train_selection),
            "holdout": selection_payload(holdout_selection),
        },
        "attribution_status_counts": {
            "train": train_status_counts,
            "holdout": holdout_status_counts,
        },
        "pair_diagnostics": {
            "train": train_diagnostics,
            "holdout": holdout_diagnostics,
        },
        "branch_utilities": utilities,
        "holdout": holdout_rows,
        "ndcg_at_6": metrics,
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


__all__ = [
    "AttributionResult",
    "AttributedTransition",
    "PairRef",
    "PairSelection",
    "aggregate_branch_utility",
    "attribute_first_divergence",
    "audit_case",
    "compare_branch_replays",
    "ndcg",
    "paired_bootstrap_lift",
    "select_same_coordinate_pairs",
    "select_same_coordinate_pairs_with_diagnostics",
    "write_json_atomic",
]
