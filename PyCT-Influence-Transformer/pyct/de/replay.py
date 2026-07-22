from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from datasets.common import enable_attack_pixels, tensor_to_in_dict_and_con_dict
from libct.branch_trace import BranchTraceEvent


@dataclass(frozen=True)
class BranchReplay:
    events: Tuple[BranchTraceEvent, ...]
    complete: bool
    event_type: str | None = None
    duration_seconds: float = 0.0
    timeout_seconds: int | None = None


def replay_one_pixel_path(
    *,
    model_name: str,
    case_index: int,
    clean_image: np.ndarray,
    coordinate: Sequence[int],
    value: float,
    model_sha256: str,
    timeout: int = 120,
) -> BranchReplay:
    sample = np.asarray(clean_image, dtype=np.float32).copy()
    coord = tuple(int(part) for part in coordinate)
    if len(coord) != sample.ndim:
        raise ValueError(f"Coordinate rank {len(coord)} does not match image rank {sample.ndim}")
    if any(part < 0 or part >= sample.shape[axis] for axis, part in enumerate(coord)):
        raise ValueError(f"Coordinate {coord} is outside image shape {sample.shape}")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value < 0.0 or numeric_value > 1.0:
        raise ValueError("Replay value must be finite and inside [0, 1]")
    sample[coord] = numeric_value
    in_dict, con_dict = tensor_to_in_dict_and_con_dict(sample)
    enable_attack_pixels(con_dict, [coord])

    from engine.executor import run

    started = time.perf_counter()
    try:
        _iterations, recorder = run(
            model_name=model_name,
            in_dict=in_dict,
            con_dict=con_dict,
            norm=True,
            solve_order_stack=False,
            idx=int(case_index),
            max_iter=0,
            single_timeout=int(timeout),
            timeout=int(timeout),
            total_timeout=int(timeout),
            only_first_forward=True,
            trace_only=True,
            branch_trace_enabled=True,
            branch_model_sha256=model_sha256,
            collect_constraints_with="queue",
            popped_log_attack_mode="de-guidance-replay",
        )
    finally:
        duration_seconds = time.perf_counter() - started
    trace = getattr(recorder, "branch_trace", None)
    if trace is None:
        raise RuntimeError("Concolic replay completed without a branch trace")
    extra_meta = getattr(recorder, "extra_meta", {})
    if extra_meta.get("status") == "error":
        raise RuntimeError(
            "Concolic replay failed: "
            f"{extra_meta.get('error_type', 'unknown')}: {extra_meta.get('error_reason', '')}"
        )
    event_type = extra_meta.get("child_event_type")
    if event_type not in {None, "soft_timeout"}:
        raise RuntimeError(f"Concolic replay ended with unsupported event: {event_type}")
    return BranchReplay(
        events=tuple(trace),
        complete=bool(getattr(recorder, "branch_trace_complete", False)),
        event_type=str(event_type) if event_type else None,
        duration_seconds=float(duration_seconds),
        timeout_seconds=int(timeout),
    )


__all__ = ["BranchReplay", "replay_one_pixel_path"]
