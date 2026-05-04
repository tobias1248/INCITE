from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tasks.paths import get_save_dir_from_save_exp


def _derive_error_outcome(status: Any, error_type: Any) -> Optional[Tuple[bool, str]]:
    if status != "error":
        return None
    suffix = str(error_type or "unknown").strip().replace(" ", "_")
    return False, f"error_{suffix}"


def derive_ton_outcome(recorder: Any) -> Tuple[bool, str]:
    extra_meta = getattr(recorder, "extra_meta", {}) or {}
    if (error_outcome := _derive_error_outcome(extra_meta.get("status"), extra_meta.get("error_type"))) is not None:
        return error_outcome
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


def stats_indicate_completion(payload: Dict[str, Any]) -> bool:
    meta = payload.get("meta") or {}
    status = meta.get("status")
    attack_label = payload.get("attack_label", meta.get("attack_label"))
    is_finished = bool(meta.get("is_finish"))
    is_timeout = bool(meta.get("is_timeout"))
    return bool(attack_label is not None or is_finished or is_timeout or status == "error")


def load_stats_payload(stats_path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    if not stats_path.is_file():
        return None, "missing_stats"
    try:
        with stats_path.open("r", encoding="utf-8") as handle:
            return json.load(handle), "ok"
    except (OSError, json.JSONDecodeError):
        return None, "invalid_stats"


def coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def extract_last_ton(stats: Dict[str, Any]) -> Optional[int]:
    meta = stats.get("meta") or {}
    ton = coerce_int(meta.get("ton"))
    if ton is not None:
        return ton
    progress = meta.get("progress") or stats.get("progress") or {}
    ton = coerce_int(progress.get("ton_current"))
    if ton is not None:
        return ton
    ton_progress = meta.get("ton_progress") or stats.get("ton_progress") or {}
    ton = coerce_int(ton_progress.get("current"))
    if ton is not None:
        return ton
    return None


def derive_stage_outcome_payload(stats: Dict[str, Any]) -> Tuple[bool, str]:
    meta = stats.get("meta") or {}
    if (error_outcome := _derive_error_outcome(meta.get("status"), meta.get("error_type"))) is not None:
        return error_outcome
    attack_label = meta.get("attack_label")
    solved_all = bool(meta.get("solve_all_ctr"))
    is_timeout = bool(meta.get("is_timeout"))
    if attack_label is not None:
        return False, "adv_found"
    if solved_all:
        return True, "solve_all_ctr"
    if is_timeout:
        return True, "timeout"
    return False, "incomplete"


def should_run_ton(
    case: Dict[str, Any],
    ton_value: int,
    ton_sequence: Sequence[int],
    *,
    force_refresh: bool,
) -> bool:
    if force_refresh:
        return True
    stats_path = Path(case["save_dir"]) / "stats.json"
    stats, _ = load_stats_payload(stats_path)
    if not stats:
        return ton_value == ton_sequence[0]
    meta = stats.get("meta") or {}
    if meta.get("attack_label") is not None or meta.get("status") == "error":
        return False
    last_ton = extract_last_ton(stats)
    if last_ton is None:
        return ton_value == ton_sequence[0]
    if last_ton > ton_value:
        return False
    should_continue, reason = derive_stage_outcome_payload(stats)
    if last_ton == ton_value:
        return reason == "incomplete"
    try:
        idx = list(ton_sequence).index(last_ton)
    except ValueError:
        return ton_value == ton_sequence[0]
    if idx + 1 >= len(ton_sequence) or ton_sequence[idx + 1] != ton_value:
        return False
    return should_continue


def should_run_payload(payload: Dict[str, Any], *, force_refresh: bool) -> bool:
    if force_refresh:
        return True
    save_exp = payload.get("save_exp") or {}
    attack_mode = save_exp.get("attack_mode", payload.get("popped_log_attack_mode", "unknown"))
    save_dir = get_save_dir_from_save_exp(
        save_exp,
        payload.get("model_name", "unknown"),
        attack_mode,
        only_first_forward=bool(save_exp.get("only_first_forward", False)),
    )
    stats_path = Path(save_dir) / "stats.json"
    stats, _ = load_stats_payload(stats_path)
    if not stats:
        return True
    return not stats_indicate_completion(stats)


def collect_stage_cases(inputs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for payload in inputs:
        ton_plans = payload.get("ton_plans") or []
        if not ton_plans:
            continue
        base_payload = dict(payload)
        base_payload.pop("ton_plans", None)
        plan_by_ton = {
            plan.get("ton"): plan for plan in ton_plans if plan.get("ton") is not None
        }
        first_plan = ton_plans[0]
        save_exp = first_plan.get("save_exp", {})
        attack_mode = save_exp.get("attack_mode", base_payload.get("popped_log_attack_mode", "unknown"))
        save_dir = get_save_dir_from_save_exp(
            save_exp,
            base_payload["model_name"],
            attack_mode,
            only_first_forward=bool(save_exp.get("only_first_forward", False)),
        )
        cases.append(
            {
                "idx": base_payload.get("idx"),
                "input_name": save_exp.get("input_name"),
                "base_payload": base_payload,
                "plans": plan_by_ton,
                "save_dir": save_dir,
            }
        )
    return cases


__all__ = [
    "collect_stage_cases",
    "derive_stage_outcome_payload",
    "derive_ton_outcome",
    "extract_last_ton",
    "load_stats_payload",
    "should_run_payload",
    "should_run_ton",
    "stats_indicate_completion",
    "update_ton_progress_stats",
]
