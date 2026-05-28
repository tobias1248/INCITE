from __future__ import annotations

import heapq
import logging
import math
from typing import Any, Optional, Tuple

from libct.constraint import Constraint
from libct.position import summarize_indices, summarize_position
from libct.searcher.base import Searcher
from libct.state import ConstraintWorkItem


log = logging.getLogger("ct.explore")


class ConstraintScheduler:
    """Compatibility facade for legacy constraint worklist scheduling."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def push_constraint(self, constraint: Constraint, position: Any) -> None:
        shap_value = 0.0
        if (
            position is not None
            and hasattr(self._engine, "comparator")
            and self._engine.comparator is not None
        ):
            layer_number, indices = position
            shap_value = self._engine.comparator.get_shap_influence(layer_number, indices)

        layer_number = None
        index_summary = "None"
        if isinstance(position, tuple) and len(position) == 2:
            layer_number = position[0]
            index_summary = summarize_indices(position[1])

        path_len = getattr(constraint, "height", None)
        if self._engine.constraints_collection_type == "priority_queue":
            score, _assert_num = self._compute_priority_score(shap_value, constraint)
            item = ConstraintWorkItem.from_constraint(
                constraint,
                position=position,
                shap_value=shap_value,
                score=score,
            )
            self._push_work_item(item)
            self._record_queue_size()
            if self._constraint_log_enabled():
                log.info(
                    "[PUSH] idx=%s layer=%s position=%s shap=%.3e path_len=%s queue_size=%s",
                    self._engine.idx,
                    layer_number,
                    index_summary,
                    abs(shap_value),
                    path_len,
                    len(self._engine.constraints_to_solve),
                )
            return

        item = ConstraintWorkItem.from_constraint(
            constraint,
            position=position,
            shap_value=shap_value,
        )
        self._push_work_item(item)
        self._record_queue_size()
        if self._constraint_log_enabled():
            log.info(
                "[PUSH] idx=%s queue=%s position=%s shap=%.3e path_len=%s total=%s",
                self._engine.idx,
                self._engine.constraints_collection_type,
                summarize_position(position),
                abs(shap_value),
                path_len,
                len(self._engine.constraints_to_solve),
            )

    def pop_constraint(self) -> Any:
        if isinstance(self._engine.constraints_to_solve, Searcher):
            return self._pop_modular_constraint()
        return self._pop_legacy_constraint()

    def _compute_priority_score(self, shap_value: float, constraint: Constraint) -> Tuple[float, int]:
        if self._engine.shap_score_alpha is None:
            raise ValueError(
                "shap_score_alpha is required when collect_constraints_with='priority_queue'; pass via --score-alpha"
            )
        path_len = int(getattr(constraint, "height", 0) or 0)
        alpha = self._engine.shap_score_alpha
        score = (1 - alpha) * math.log10(abs(shap_value) + self._engine.SHAP_SCORE_EPS)
        score -= alpha * math.log10(path_len + 1)
        return score, path_len

    def _push_work_item(self, item: ConstraintWorkItem) -> None:
        worklist = self._engine.constraints_to_solve
        if isinstance(worklist, Searcher):
            worklist.push(item)
        elif self._engine.constraints_collection_type == "priority_queue":
            heapq.heappush(
                worklist,
                (-item.score, item.constraint.id, item.position, item.constraint, item.shap_value),
            )
        else:
            worklist.append(item.constraint)

    def _record_queue_size(self) -> None:
        recorder = self._get_recorder()
        if recorder is None:
            return
        current_size = len(self._engine.constraints_to_solve)
        recorder.queue_last = current_size
        if current_size > getattr(recorder, "queue_max", 0):
            recorder.queue_max = current_size

    def _log_pop_event(
        self,
        *,
        queue_mode: str,
        remaining: int,
        layer: Any = None,
        indices: Any = None,
        shap_value: Optional[float] = None,
        path_len: Optional[int] = None,
    ) -> None:
        if not self._constraint_log_enabled():
            return
        attack_mode = getattr(self._engine, "popped_log_attack_mode", "unknown")
        sample_idx = getattr(self._engine, "idx", "unknown")
        if queue_mode == "priority":
            log.info(
                "[POP] idx=%s attack=%s queue=%s layer=%s position=%s shap=%s path_len=%s remaining=%d",
                sample_idx,
                attack_mode,
                queue_mode,
                layer,
                summarize_indices(indices),
                shap_value,
                path_len,
                remaining,
            )
            return
        log.info(
            "[POP] idx=%s attack=%s queue=%s path_len=%s remaining=%d",
            sample_idx,
            attack_mode,
            queue_mode,
            path_len,
            remaining,
        )

    def _pop_modular_constraint(self) -> Any:
        item = self._engine.constraints_to_solve.pop()
        constraint = item.constraint
        if self._engine.constraints_collection_type == "priority_queue":
            position = item.position
            layer_number = None
            indices = None
            if isinstance(position, tuple) and len(position) == 2:
                layer_number, indices = position
            self._log_pop_event(
                queue_mode="priority",
                remaining=len(self._engine.constraints_to_solve),
                layer=layer_number,
                indices=indices,
                shap_value="{:.3e}".format(abs(item.shap_value)),
                path_len=getattr(constraint, "height", None),
            )
            log.debug(
                "Popped constraint from queue (position=%s shap_value=%s constraint_id=%s)",
                summarize_position(position),
                item.shap_value,
                constraint.id,
            )
            return constraint, item.shap_value, position

        queue_mode = "stack" if self._engine.constraints_collection_type == "stack" else "queue"
        self._log_pop_event(
            queue_mode=queue_mode,
            remaining=len(self._engine.constraints_to_solve),
            path_len=getattr(constraint, "height", None),
        )
        return constraint

    def _pop_legacy_constraint(self) -> Any:
        if self._engine.constraints_collection_type == "stack":
            constraint = self._engine.constraints_to_solve.pop()
            self._log_pop_event(
                queue_mode="stack",
                remaining=len(self._engine.constraints_to_solve),
                path_len=getattr(constraint, "height", None),
            )
            return constraint

        if self._engine.constraints_collection_type == "queue":
            constraint = self._engine.constraints_to_solve.popleft()
            self._log_pop_event(
                queue_mode="queue",
                remaining=len(self._engine.constraints_to_solve),
                path_len=getattr(constraint, "height", None),
            )
            return constraint

        if self._engine.constraints_collection_type == "priority_queue":
            score, constraint_id, position, constraint, shap_value = heapq.heappop(
                self._engine.constraints_to_solve
            )
            layer_number, indices = position
            self._log_pop_event(
                queue_mode="priority",
                remaining=len(self._engine.constraints_to_solve),
                layer=layer_number,
                indices=indices,
                shap_value="{:.3e}".format(abs(shap_value)),
                path_len=getattr(constraint, "height", None),
            )
            log.debug(
                "Popped constraint from queue (position=%s shap_value=%s constraint_id=%s)",
                summarize_position(position),
                shap_value,
                constraint_id,
            )
            return constraint, shap_value, position

        raise ValueError(
            "Unsupported constraint collection type: {}".format(
                self._engine.constraints_collection_type
            )
        )

    def _constraint_log_enabled(self) -> bool:
        return bool(getattr(self._engine, "constraint_log_enabled", False))

    def _get_recorder(self) -> Any:
        get_recorder = getattr(self._engine, "_get_recorder", None)
        if callable(get_recorder):
            return get_recorder()
        return None
