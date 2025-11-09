from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import dnn_predict_common
from utils.experiment_task_specs import get_save_dir_from_save_exp

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


PixelCoordinates = Tuple[int, ...]


@dataclass
class PixelAssignment:
    coordinate: PixelCoordinates
    original_value: float
    assigned_value: float


@dataclass
class RandomAssignResult:
    model_name: str
    pixel_source: str
    idx: int
    ton: int
    assignments: List[PixelAssignment]
    logits_before: np.ndarray
    logits_after: np.ndarray
    pred_before: int
    pred_after: int
    success: bool
    timestamp: datetime
    save_exp: Dict[str, Any]
    original_input: np.ndarray
    modified_input: np.ndarray
    base_seed: int
    attempt: int
    attack_wall_time: float = 0.0


def extract_target_pixels(con_dict: Dict[str, Any]) -> List[PixelCoordinates]:
    """Return coordinates whose constraint flag is enabled."""
    coords: List[PixelCoordinates] = []
    for key, value in con_dict.items():
        if not value:
            continue
        parts = key.split("_")[1:]
        coords.append(tuple(int(part) for part in parts))
    coords.sort()
    return coords


def _forward(array: np.ndarray) -> np.ndarray:
    """Forward the NNModel and return logits as a 1D numpy array."""
    outputs = dnn_predict_common.myModel.forward(array.tolist())
    logits = np.asarray(outputs, dtype=np.float32)
    if logits.ndim > 1:
        logits = logits.reshape(-1)
    return logits


def _ensure_model_loaded(model_name: str) -> None:
    model_path = Path("model") / f"{model_name}.h5"
    dnn_predict_common.init_model(str(model_path))


def run_random_assign_step(
    payload: Dict[str, Any],
    *,
    pixel_source: str,
    base_seed: int,
    attempt: int,
) -> RandomAssignResult:
    """Apply random assignments to the selected pixels and collect results."""
    model_name = payload["model_name"]
    _ensure_model_loaded(model_name)

    idx = int(payload["idx"])
    con_dict: Dict[str, Any] = payload["con_dict"]
    coords = extract_target_pixels(con_dict)

    if not coords:
        raise ValueError("No attack pixels identified in con_dict; unable to perform random assign.")

    rng = np.random.default_rng(base_seed + idx + attempt)
    original = np.array(payload["input_for_shap"], dtype=np.float32, copy=True)
    modified = original.copy()

    assignments: List[PixelAssignment] = []
    for coordinate in coords:
        original_value = float(modified[coordinate])
        assigned_value = float(rng.uniform(0.0, 1.0))
        modified[coordinate] = assigned_value
        assignments.append(
            PixelAssignment(
                coordinate=coordinate,
                original_value=original_value,
                assigned_value=assigned_value,
            )
        )

    logits_before = _forward(original)
    logits_after = _forward(modified)
    pred_before = int(np.argmax(logits_before))
    pred_after = int(np.argmax(logits_after))

    return RandomAssignResult(
        model_name=model_name,
        pixel_source=pixel_source,
        idx=idx,
        ton=len(coords),
        assignments=assignments,
        logits_before=logits_before,
        logits_after=logits_after,
        pred_before=pred_before,
        pred_after=pred_after,
        success=pred_before != pred_after,
        timestamp=datetime.now(timezone.utc),
        save_exp=payload.get("save_exp", {}),
        original_input=original,
        modified_input=modified,
        base_seed=base_seed,
        attempt=attempt,
    )


def format_assignments(assignments: Sequence[PixelAssignment]) -> str:
    entries: List[str] = []
    for assignment in assignments:
        coord = ",".join(str(axis) for axis in assignment.coordinate)
        entries.append(
            f"{coord}:{assignment.original_value:.6f}->{assignment.assigned_value:.6f}"
        )
    return "[" + "; ".join(entries) + "]"


def write_combined_log(result: RandomAssignResult) -> None:
    base_dir = (
        Path("popped_constraint_position")
        / result.model_name
        / f"random_assign_{result.pixel_source}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / f"random-assign-{result.idx}.txt"

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"success: {result.success}\n")
        handle.write(f"attack_wall_time: {result.attack_wall_time:.6f}\n")
        for assignment in result.assignments:
            position_repr = f"(0, {assignment.coordinate})"
            handle.write("\n")
            handle.write(f"attack position: {position_repr}\n")
            handle.write(f"assign value: {assignment.assigned_value:.6f}\n")


def _save_adversarial_image(result: RandomAssignResult, save_dir: Path) -> None:
    if cv2 is None:
        return
    array = np.clip(result.modified_input * 255.0, 0.0, 255.0).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array.reshape(array.shape[0], array.shape[1])
    img_name = f"adv_{result.pred_before}_to_{result.pred_after}_attempt{result.attempt}.jpg"
    cv2.imwrite(str(save_dir / img_name), array)


def write_experiment_artifacts(result: RandomAssignResult) -> None:
    save_dir = Path(
        get_save_dir_from_save_exp(
            result.save_exp,
            result.model_name,
            "priority_queue",
        )
    )
    os.makedirs(save_dir, exist_ok=True)

    stats = {
        "meta": {
            "mode": "random_assign",
            "pixel_source": result.pixel_source,
            "ton": result.ton,
            "idx": result.idx,
            "original_label": result.pred_before,
            "attack_label": result.pred_after,
            "success": result.success,
            "timestamp": result.timestamp.isoformat(),
            "base_seed": result.base_seed,
            "attempts": result.attempt + 1,
            "attack_wall_time": result.attack_wall_time,
            "is_finish": True,
        },
        "assignments": [
            {
                "coordinate": list(assignment.coordinate),
                "original": assignment.original_value,
                "assigned": assignment.assigned_value,
            }
            for assignment in result.assignments
        ],
        "logits_before": result.logits_before.tolist(),
        "logits_after": result.logits_after.tolist(),
    }

    stats_path = save_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle)

    sat_inputs = result.modified_input[np.newaxis, ...].astype(np.float32)
    np.save(save_dir / "sat_inputs.npy", sat_inputs)

    if result.success:
        _save_adversarial_image(result, save_dir)
