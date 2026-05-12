from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

from libct.random_assign_attack import (
    run_random_assign_step,
    write_combined_log,
    write_experiment_artifacts,
)
from orchestration.progress import derive_ton_outcome, update_ton_progress_stats

log = logging.getLogger("ct.runner")

__all__ = [
    "QueueRunner",
    "ShapRunner",
    "RandomAssignRunner",
    "run_attack_with_shap",
    "run_attack_with_queue",
    "run_attack_with_random_assign",
    "update_ton_progress_stats",
]


@dataclass
class BaseRunner:
    timeout: int
    norm: bool
    collect_constraints_with: Literal["priority_queue", "queue"]
    constraint_build_timeout: bool = True
    constraint_build_timeout_seconds: int = 30
    solver_run_timeout: Optional[int] = None
    sat_batch_size: int = 1
    error_retry_limit: int = 2

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

    def _run_single(self, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    @staticmethod
    def _get_result_recorder(result: Any) -> Any:
        if isinstance(result, tuple) and len(result) >= 2:
            return result[1]
        return None

    @staticmethod
    def _get_recorder_error_meta(recorder: Any) -> Tuple[Optional[str], Optional[str]]:
        if recorder is None:
            return None, None
        meta = getattr(recorder, "extra_meta", {}) or {}
        return meta.get("status"), meta.get("error_type")

    @staticmethod
    def _get_result_context(payload: Dict[str, Any], recorder: Any) -> Tuple[Optional[str], Optional[str]]:
        save_exp = payload.get("save_exp") or {}
        input_name = getattr(recorder, "input_name", None) if recorder is not None else None
        save_dir = getattr(recorder, "save_dir", None) if recorder is not None else None
        if input_name is None:
            input_name = save_exp.get("input_name")
        return input_name, save_dir

    def _is_retryable_transfer_error(self, recorder: Any) -> bool:
        # "incomplete" means the run was valid but unfinished. "error" means
        # the run itself was semantically invalid and may need targeted retry.
        status, error_type = self._get_recorder_error_meta(recorder)
        return status == "error" and error_type == "constraint_transfer_failure"

    def _is_terminal_error(self, recorder: Any) -> bool:
        status, _error_type = self._get_recorder_error_meta(recorder)
        return status == "error"

    def _execute_plan_with_retries(
        self,
        plan_payload: Dict[str, Any],
        *,
        payload_idx: Any,
        ton_value: Any,
    ) -> Any:
        retry_count = 0
        while True:
            result = self._execute_attack(plan_payload)
            recorder = self._get_result_recorder(result)
            if not self._is_retryable_transfer_error(recorder) or retry_count >= self.error_retry_limit:
                return result
            retry_count += 1
            input_name, save_dir = self._get_result_context(plan_payload, recorder)
            log.warning(
                "[PAYLOAD-RETRY] idx=%s ton=%s retry=%s/%s reason=constraint_transfer_failure input_name=%s save_dir=%s",
                payload_idx,
                ton_value,
                retry_count,
                self.error_retry_limit,
                input_name,
                save_dir,
            )

    def _log_payload_end(self, payload: Dict[str, Any], result: Any) -> None:
        recorder = self._get_result_recorder(result)
        input_name, save_dir = self._get_result_context(payload, recorder)
        if recorder is not None and self._is_terminal_error(recorder):
            _status, error_type = self._get_recorder_error_meta(recorder)
            log.error(
                "[PAYLOAD-ERROR] idx=%s attack=%s error_type=%s input_name=%s save_dir=%s",
                payload.get("idx"),
                payload.get("popped_log_attack_mode"),
                error_type or "unknown",
                input_name,
                save_dir,
            )
        elif recorder is not None and getattr(recorder, "is_timeout", False):
            log.warning(
                "[PAYLOAD-TIMEOUT] idx=%s attack=%s total_iter=%s input_name=%s save_dir=%s",
                payload.get("idx"),
                payload.get("popped_log_attack_mode"),
                getattr(recorder, "total_iter", "?"),
                input_name,
                save_dir,
            )
        else:
            log.info(
                "[PAYLOAD-END] idx=%s attack=%s input_name=%s save_dir=%s",
                payload.get("idx"),
                payload.get("popped_log_attack_mode"),
                input_name,
                save_dir,
            )

    def _cleanup(self, payload: Dict[str, Any]) -> None:
        payload.clear()
        gc.collect()

    def _execute_attack(self, payload: Dict[str, Any]) -> Any:
        from engine.executor import run

        return run(
            **payload,
            norm=self.norm,
            max_iter=0,
            total_timeout=self.timeout,
            single_timeout=self.timeout,
            timeout=self.timeout,
            constraint_build_timeout=self.constraint_build_timeout,
            constraint_build_timeout_seconds=self.constraint_build_timeout_seconds,
            collect_constraints_with=self.collect_constraints_with,
            solver_run_timeout=self.solver_run_timeout,
            sat_batch_size=self.sat_batch_size,
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
        should_continue, reason = derive_ton_outcome(recorder)
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
        constraint_build_timeout_seconds: int = 30,
        solver_run_timeout: Optional[int] = None,
        sat_batch_size: int = 1,
        collect_constraints_with: Literal["priority_queue", "queue"] = "queue",
        error_retry_limit: int = 2,
    ) -> None:
        super().__init__(
            timeout=timeout,
            norm=norm,
            collect_constraints_with=collect_constraints_with,
            constraint_build_timeout=constraint_build_timeout,
            constraint_build_timeout_seconds=constraint_build_timeout_seconds,
            solver_run_timeout=solver_run_timeout,
            sat_batch_size=sat_batch_size,
            error_retry_limit=error_retry_limit,
        )
        self.collect_constraints_with = collect_constraints_with

    def _run_single(self, payload: Dict[str, Any]) -> None:
        ton_plans = payload.pop("ton_plans", None)
        if not ton_plans:
            return self._execute_plan_with_retries(
                payload,
                payload_idx=payload.get("idx"),
                ton_value=(payload.get("save_exp") or {}).get("ton"),
            )

        base_payload = dict(payload)
        last_result = None
        ton_sequence = [plan.get("ton") for plan in ton_plans if "ton" in plan]

        for plan in ton_plans:
            plan_payload = dict(base_payload)
            plan_payload["con_dict"] = plan["con_dict"]
            plan_payload["save_exp"] = plan["save_exp"]

            # Retry stays at the runner layer because only the runner owns
            # re-executing a single TON payload; progress only interprets results.
            result = self._execute_plan_with_retries(
                plan_payload,
                payload_idx=base_payload.get("idx"),
                ton_value=plan.get("ton"),
            )
            last_result = result

            recorder = self._get_result_recorder(result)
            if recorder is not None:
                self._write_ton_sequence(recorder, ton_sequence, plan.get("ton"))
                should_continue, reason = derive_ton_outcome(recorder)
                if reason == "adv_found":
                    break
                if should_continue:
                    continue
                break

        return last_result


class ShapRunner(BaseRunner):
    def __init__(
        self,
        timeout: int,
        norm: bool,
        *,
        model_type: str = "transformer",
        collect_constraints_with: Literal["priority_queue", "queue"] = "priority_queue",
        constraint_build_timeout: bool = True,
        constraint_build_timeout_seconds: int = 30,
        solver_run_timeout: Optional[int] = None,
        sat_batch_size: int = 1,
        error_retry_limit: int = 2,
    ) -> None:
        super().__init__(
            timeout=timeout or 0,
            norm=norm,
            collect_constraints_with=collect_constraints_with,
            constraint_build_timeout=constraint_build_timeout,
            constraint_build_timeout_seconds=constraint_build_timeout_seconds,
            solver_run_timeout=solver_run_timeout,
            sat_batch_size=sat_batch_size,
            error_retry_limit=error_retry_limit,
        )
        self.collect_constraints_with = collect_constraints_with
        self.model_type = model_type

    def _run_single(self, payload: Dict[str, Any]) -> None:
        ton_plans = payload.pop("ton_plans", None)
        if not ton_plans:
            return self._execute_plan_with_retries(
                payload,
                payload_idx=payload.get("idx"),
                ton_value=(payload.get("save_exp") or {}).get("ton"),
            )

        base_payload = dict(payload)
        last_result = None
        ton_sequence = [plan.get("ton") for plan in ton_plans if "ton" in plan]

        for plan in ton_plans:
            plan_payload = dict(base_payload)
            plan_payload["con_dict"] = plan["con_dict"]
            plan_payload["save_exp"] = plan["save_exp"]

            # Retry stays at the runner layer because only the runner owns
            # re-executing a single TON payload; progress only interprets results.
            result = self._execute_plan_with_retries(
                plan_payload,
                payload_idx=base_payload.get("idx"),
                ton_value=plan.get("ton"),
            )
            last_result = result

            recorder = self._get_result_recorder(result)
            if recorder is not None:
                self._write_ton_sequence(recorder, ton_sequence, plan.get("ton"))
                should_continue, reason = derive_ton_outcome(recorder)
                if reason == "adv_found":
                    break
                if should_continue:
                    continue
                break

        return last_result


class RandomAssignRunner(BaseRunner):
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
        constraint_build_timeout_seconds: int = 30,
        solver_run_timeout: Optional[int] = None,
        sat_batch_size: int = 1,
    ) -> None:
        super().__init__(
            timeout=timeout or 0,
            norm=norm,
            collect_constraints_with=collect_constraints_with,
            constraint_build_timeout=constraint_build_timeout,
            constraint_build_timeout_seconds=constraint_build_timeout_seconds,
            solver_run_timeout=solver_run_timeout,
            sat_batch_size=sat_batch_size,
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
    constraint_build_timeout_seconds: int = 30,
    solver_run_timeout: Optional[int] = None,
    sat_batch_size: int = 1,
    model_type: str = "transformer",
    collect_constraints_with: Literal["priority_queue", "queue"] = "priority_queue",
    error_retry_limit: int = 2,
) -> None:
    ShapRunner(
        timeout=timeout,
        norm=norm,
        model_type=model_type,
        collect_constraints_with=collect_constraints_with,
        constraint_build_timeout=constraint_build_timeout,
        constraint_build_timeout_seconds=constraint_build_timeout_seconds,
        solver_run_timeout=solver_run_timeout,
        sat_batch_size=sat_batch_size,
        error_retry_limit=error_retry_limit,
    ).run_tasks(args)


def run_attack_with_queue(
    args: Sequence[Dict[str, Any]],
    timeout: int,
    norm: bool,
    constraint_build_timeout: bool = True,
    constraint_build_timeout_seconds: int = 30,
    solver_run_timeout: Optional[int] = None,
    sat_batch_size: int = 1,
    collect_constraints_with: Literal["priority_queue", "queue"] = "queue",
    error_retry_limit: int = 2,
) -> None:
    QueueRunner(
        timeout=timeout,
        norm=norm,
        constraint_build_timeout=constraint_build_timeout,
        constraint_build_timeout_seconds=constraint_build_timeout_seconds,
        solver_run_timeout=solver_run_timeout,
        sat_batch_size=sat_batch_size,
        collect_constraints_with=collect_constraints_with,
        error_retry_limit=error_retry_limit,
    ).run_tasks(args)


def run_attack_with_random_assign(
    args: Sequence[Dict[str, Any]],
    timeout: int,
    norm: bool,
    constraint_build_timeout: bool = True,
    constraint_build_timeout_seconds: int = 30,
    solver_run_timeout: Optional[int] = None,
    sat_batch_size: int = 1,
    *,
    pixel_source: str,
    base_seed: int,
    model_type: str = "transformer",
    collect_constraints_with: Literal["priority_queue", "queue"] = "priority_queue",
) -> None:
    RandomAssignRunner(
        timeout=timeout,
        constraint_build_timeout=constraint_build_timeout,
        constraint_build_timeout_seconds=constraint_build_timeout_seconds,
        solver_run_timeout=solver_run_timeout,
        sat_batch_size=sat_batch_size,
        norm=norm,
        pixel_source=pixel_source,
        base_seed=base_seed,
        model_type=model_type,
        collect_constraints_with=collect_constraints_with,
    ).run_tasks(args)
