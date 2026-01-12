from __future__ import annotations

import gc
import json
import logging
import queue
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Literal, Tuple
import time
from pathlib import Path

from libct.random_assign_attack import (
    run_random_assign_step,
    write_combined_log,
    write_experiment_artifacts,
)
from libct.shapInfl import ShapValuesCalculator

log = logging.getLogger("ct.runner")


__all__ = [
    "QueueRunner",
    "ShapRunner",
    "run_attack_with_shap",
    "run_attack_with_queue",
    "run_attack_with_random_assign",
    "update_ton_progress_stats",
]


def _derive_ton_outcome(recorder: Any) -> Tuple[bool, str]:
    attack_label = getattr(recorder, "attack_label", None)
    solved_all = getattr(recorder, "solve_all_ctr", False)
    is_timeout = getattr(recorder, "is_timeout", False)
    if attack_label is not None:
        return False, "adv_found"
    if solved_all:
        return True, "solve_all_ctr"
    if is_timeout:
        return True, "timeout"
    return False, "incomplete"


def update_ton_progress_stats(
    stats_path: Path,
    *,
    current_ton: int,
    status: str,
    reason: str,
    next_ton: Optional[int] = None,
) -> bool:
    if not stats_path.is_file():
        return False
    try:
        with stats_path.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    progress = {
        "current": current_ton,
        "next": next_ton,
        "stop_at": current_ton if status == "stop" else None,
        "status": status,
        "reason": reason,
    }
    meta = stats.setdefault("meta", {})
    meta["ton_progress"] = progress
    stats.pop("ton_progress", None)
    stats.pop("progress", None)
    stats.pop("ton_sequence", None)
    meta.pop("progress", None)
    meta.pop("ton_sequence", None)
    meta.pop("finished", None)

    try:
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle)
    except OSError:
        return False
    try:
        history_path = stats_path.with_name("stats_history.jsonl")
        with history_path.open("a", encoding="utf-8") as handle:
            json.dump(stats, handle)
            handle.write("\n")
    except OSError:
        return False
    return True


@dataclass
class BaseRunner:
    timeout: int
    norm: bool
    collect_constraints_with: Literal["priority_queue", "queue"]
    constraint_build_timeout: bool = True
    solver_run_timeout: Optional[int] = None

    def run_tasks(self, tasks: Sequence[Dict[str, Any]]) -> None:
        for payload in tasks:
            try:
                log.info(
                    "[PAYLOAD-START] idx=%s attack=%s mode=%s",
                    payload.get("idx"),
                    payload.get("popped_log_attack_mode"),
                    payload.get("solve_order_stack"),
                )
                result = self._run_single(payload)
                self._log_payload_end(payload, result)
            except Exception:
                log.exception(
                    "[PAYLOAD-ERROR] idx=%s attack=%s",
                    payload.get("idx"),
                    payload.get("popped_log_attack_mode"),
                )
                raise
            finally:
                self._cleanup(payload)

    def _run_single(self, payload: Dict[str, Any]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _log_payload_end(self, payload: Dict[str, Any], result: Any) -> None:
        recorder = None
        if isinstance(result, tuple) and len(result) >= 2:
            recorder = result[1]
        if recorder is not None and getattr(recorder, "is_timeout", False):
            log.warning(
                "[PAYLOAD-TIMEOUT] idx=%s attack=%s total_iter=%s",
                payload.get("idx"),
                payload.get("popped_log_attack_mode"),
                getattr(recorder, "total_iter", "?"),
            )
        else:
            log.info(
                "[PAYLOAD-END] idx=%s attack=%s",
                payload.get("idx"),
                payload.get("popped_log_attack_mode"),
            )

    def _cleanup(self, payload: Dict[str, Any]) -> None:
        payload.clear()
        gc.collect()

    def _execute_attack(self, payload: Dict[str, Any]) -> Any:
        import run_dnnct

        return run_dnnct.run(
            **payload,
            norm=self.norm,
            max_iter=0,
            total_timeout=self.timeout,
            single_timeout=self.timeout,
            timeout=self.timeout,
            constraint_build_timeout=self.constraint_build_timeout,
            collect_constraints_with=self.collect_constraints_with,
            solver_run_timeout=self.solver_run_timeout,
        )

    @staticmethod
    def _write_ton_sequence(
        recorder: Any, ton_sequence: Sequence[int], current_ton: Optional[int] = None
    ) -> None:
        if not recorder:
            return
        if current_ton is None:
            current_ton = ton_sequence[-1] if ton_sequence else None
        if current_ton is None:
            return
        save_dir = getattr(recorder, "save_dir", None)
        if not save_dir:
            return
        stats_path = Path(save_dir) / "stats.json"
        if not stats_path.is_file():
            return
        should_continue, reason = _derive_ton_outcome(recorder)
        status = "continue" if should_continue else "stop"
        next_ton = None
        try:
            idx = list(ton_sequence).index(current_ton)
        except ValueError:
            idx = None
        if isinstance(idx, int) and idx + 1 < len(ton_sequence):
            next_ton = ton_sequence[idx + 1]
        update_ton_progress_stats(
            stats_path,
            current_ton=current_ton,
            status=status,
            reason=reason,
            next_ton=next_ton,
        )


class QueueRunner(BaseRunner):
    def __init__(
        self,
        timeout: int,
        norm: bool,
        constraint_build_timeout: bool = True,
        solver_run_timeout: Optional[int] = None,
        collect_constraints_with: Literal["priority_queue", "queue"] = "queue",
    ) -> None:
        super().__init__(
            timeout=timeout,
            norm=norm,
            collect_constraints_with=collect_constraints_with,
            constraint_build_timeout=constraint_build_timeout,
            solver_run_timeout=solver_run_timeout,
        )
        self.collect_constraints_with = collect_constraints_with

    def _run_single(self, payload: Dict[str, Any]) -> None:
        ton_plans = payload.pop("ton_plans", None)
        if not ton_plans:
            return self._execute_attack(payload)

        base_payload = dict(payload)
        last_result = None
        ton_sequence = [plan.get("ton") for plan in ton_plans if "ton" in plan]

        for plan in ton_plans:
            plan_payload = dict(base_payload)
            plan_payload["con_dict"] = plan["con_dict"]
            plan_payload["save_exp"] = plan["save_exp"]

            start_time = time.monotonic()
            result = self._execute_attack(plan_payload)
            last_result = result

            recorder = None
            if isinstance(result, tuple) and len(result) >= 2:
                recorder = result[1]
            if recorder is not None:
                recorder.attack_wall_time = time.monotonic() - start_time  # type: ignore[attr-defined]
                self._write_ton_sequence(recorder, ton_sequence, plan.get("ton"))
                should_continue, reason = _derive_ton_outcome(recorder)
                if reason == "adv_found":
                    break  # success
                if should_continue:
                    continue  # solved all or timed out, move to next ton
                break

        return last_result


class ShapRunner(BaseRunner):
    """Execute SHAP-guided attacks while respecting CLI-provided options."""

    def __init__(
        self,
        timeout: int,
        norm: bool,
        *,
        model_type: str = "transformer",
        collect_constraints_with: Literal["priority_queue", "queue"] = "priority_queue",
        constraint_build_timeout: bool = True,
        solver_run_timeout: Optional[int] = None,
    ) -> None:
        super().__init__(
            timeout=timeout or 0,
            norm=norm,
            collect_constraints_with=collect_constraints_with,
            constraint_build_timeout=constraint_build_timeout,
            solver_run_timeout=solver_run_timeout,
        )
        self.collect_constraints_with = collect_constraints_with
        self.model_type = model_type

    def _run_single(self, payload: Dict[str, Any]) -> None:
        ton_plans = payload.pop("ton_plans", None)
        if not ton_plans:
            return self._execute_attack(payload)

        base_payload = dict(payload)
        last_result = None
        ton_sequence = [plan.get("ton") for plan in ton_plans if "ton" in plan]

        for plan in ton_plans:
            plan_payload = dict(base_payload)
            plan_payload["con_dict"] = plan["con_dict"]
            plan_payload["save_exp"] = plan["save_exp"]

            start_time = time.monotonic()
            result = self._execute_attack(plan_payload)
            last_result = result

            recorder = None
            if isinstance(result, tuple) and len(result) >= 2:
                recorder = result[1]
            if recorder is not None:
                recorder.attack_wall_time = time.monotonic() - start_time  # type: ignore[attr-defined]
                self._write_ton_sequence(recorder, ton_sequence, plan.get("ton"))
                should_continue, reason = _derive_ton_outcome(recorder)
                if reason == "adv_found":
                    break  # success
                if should_continue:
                    continue  # solved all or timed out, move to next ton
                break

        return last_result


# class _ShapPrefetchRunner(BaseRunner):
#     """Pre-compute SHAP values to warm caches prior to attack execution."""

#     def __init__(self, timeout: int = 0) -> None:
#         super().__init__(timeout=timeout or 0, norm=False)

#     def _run_single(self, payload: Dict[str, Any]) -> None:
#         model_name = payload.get("model_name")
#         if model_name is None:
#             raise KeyError("Expected 'model_name' in payload for SHAP computation.")

#         calculator = ShapValuesCalculator(
#             model_path=f"./model/{model_name}.h5",
#             background_dataset=payload["background_dataset_for_shap"],
#             input_data=payload["input_for_shap"],
#             idx=payload["idx"],
#             explainer_type=payload.get("explainer_type", "gradient"),
#         )
#         assume_cached = bool(payload.get("shap_value_pre_calculated"))
#         calculator.ensure(
#             assume_cached=assume_cached,
#             force_refresh=not assume_cached,
#         )


class RandomAssignRunner(BaseRunner):
    """Execute baseline attacks that randomly assign values to selected pixels."""

    def __init__(
        self,
        timeout: int,
        norm: bool,
        *,
        pixel_source: str,
        base_seed: int,
        model_type: str = "transformer",
        collect_constraints_with: Literal["priority_queue", "queue"] = "priority_queue",
        constraint_build_timeout: bool = True,
        solver_run_timeout: Optional[int] = None,
    ) -> None:
        super().__init__(
            timeout=timeout or 0,
            norm=norm,
            collect_constraints_with=collect_constraints_with,
            constraint_build_timeout=constraint_build_timeout,
            solver_run_timeout=solver_run_timeout,
        )
        self.pixel_source = pixel_source
        self.base_seed = base_seed
        self.model_type = model_type
        self.collect_constraints_with = collect_constraints_with

    def _run_single(self, payload: Dict[str, Any]) -> None:
        ton_plans = payload.pop("ton_plans", None)
        if not ton_plans:
            return self._run_random_assign_for_plan(payload)

        base_payload = dict(payload)
        last_result = None
        ton_sequence = [plan.get("ton") for plan in ton_plans if "ton" in plan]

        for plan in ton_plans:
            plan_payload = dict(base_payload)
            plan_payload["con_dict"] = plan["con_dict"]
            plan_payload["save_exp"] = plan["save_exp"]

            result = self._run_random_assign_for_plan(plan_payload)
            last_result = result
            if result.success:
                break

        if last_result is None:
            raise RuntimeError("Random assign baseline failed to produce any attempt result.")
        if hasattr(last_result, "__setattr__"):
            last_result.ton_sequence = ton_sequence or None
        return last_result

    def _run_random_assign_for_plan(self, payload: Dict[str, Any]) -> Any:
        start_time = time.monotonic()
        attempt = 0
        final_result = None

        while True:
            result = run_random_assign_step(
                payload,
                pixel_source=self.pixel_source,
                base_seed=self.base_seed,
                attempt=attempt,
            )
            result.attack_wall_time = time.monotonic() - start_time
            write_combined_log(result)
            final_result = result
            if result.success:
                break

            attempt += 1
            if time.monotonic() - start_time >= self.timeout:
                break

        if final_result is None:
            raise RuntimeError("Random assign baseline failed to produce any attempt result.")
        final_result.attack_wall_time = time.monotonic() - start_time
        write_experiment_artifacts(final_result)
        return final_result


def run_attack_with_shap(
    args: Sequence[Dict[str, Any]],
    timeout: int,
    norm: bool,
    constraint_build_timeout: bool = True,
    solver_run_timeout: Optional[int] = None,
    model_type: str = "transformer",
    collect_constraints_with: Literal["priority_queue", "queue"] = "priority_queue",
) -> None:
    ShapRunner(
        timeout=timeout,
        norm=norm,
        model_type=model_type,
        collect_constraints_with=collect_constraints_with,
        constraint_build_timeout=constraint_build_timeout,
        solver_run_timeout=solver_run_timeout,
    ).run_tasks(args)


def run_attack_with_queue(
    args: Sequence[Dict[str, Any]],
    timeout: int,
    norm: bool,
    constraint_build_timeout: bool = True,
    solver_run_timeout: Optional[int] = None,
    collect_constraints_with: Literal["priority_queue", "queue"] = "queue",
) -> None:
    QueueRunner(
        timeout=timeout,
        norm=norm,
        constraint_build_timeout=constraint_build_timeout,
        solver_run_timeout=solver_run_timeout,
        collect_constraints_with=collect_constraints_with,
    ).run_tasks(args)


def run_attack_with_random_assign(
    args: Sequence[Dict[str, Any]],
    timeout: int,
    norm: bool,
    constraint_build_timeout: bool = True,
    solver_run_timeout: Optional[int] = None,
    *,
    pixel_source: str,
    base_seed: int,
    model_type: str = "transformer",
    collect_constraints_with: Literal["priority_queue", "queue"] = "priority_queue",
) -> None:
    RandomAssignRunner(
        timeout=timeout,
        constraint_build_timeout=constraint_build_timeout,
        solver_run_timeout=solver_run_timeout,
        norm=norm,
        pixel_source=pixel_source,
        base_seed=base_seed,
        model_type=model_type,
        collect_constraints_with=collect_constraints_with,
    ).run_tasks(args)


# def shap_prefetch(
#     args: Sequence[Dict[str, Any]],
# ) -> None:
#     _ShapPrefetchRunner().run_tasks(args)
