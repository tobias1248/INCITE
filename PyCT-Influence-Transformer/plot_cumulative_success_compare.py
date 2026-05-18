#!/usr/bin/env python3
"""Plot cumulative success rate versus cumulative time for two attack tools.

This script compares:
- PyCT experiment outputs stored as case_*/stats.json
- one-pixel-attack outputs stored as payload/results/checkpoint pickle files

The output figure is a pure SVG so it can be generated without matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pickle
import re
import sys
from collections import Counter
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape


OPA_RESULT_COLUMNS = [
    "model",
    "pixels",
    "image",
    "true",
    "predicted",
    "success",
    "cdiff",
    "prior_probs",
    "predicted_probs",
    "perturbation",
    "original_pred",
    "prediction_flipped",
    "original_correct",
    "duration",
    "stop_reason",
    "timed_out",
    "de_nit",
    "de_nfev",
    "rss_mb",
    "rss_hwm_mb",
    "mem_available_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyCT and one-pixel-attack cumulative success rate versus cumulative time.",
    )
    parser.add_argument("--pyct-root", required=True, help="Root directory containing case_*/stats.json.")
    parser.add_argument(
        "--opa-path",
        required=True,
        help="Path to one-pixel-attack payload pickle, checkpoint pickle, run directory, or results root.",
    )
    parser.add_argument("--output-svg", default=None, help="Optional output SVG path.")
    parser.add_argument("--output-csv", default=None, help="Optional CSV path for plot-ready time-series data.")
    parser.add_argument(
        "--raw-output-dir",
        default="tmp",
        help="Directory for raw case-level CSV exports. Default: ./tmp",
    )
    parser.add_argument("--model-name", required=True, help="Model name shown in the figure title.")
    parser.add_argument("--pyct-label", default="PyCT", help="Legend label for PyCT.")
    parser.add_argument("--opa-label", default="one-pixel-attack", help="Legend label for one-pixel-attack.")
    parser.add_argument(
        "--time-horizon",
        type=float,
        default=900.0,
        help="Elapsed-time horizon shown on the x-axis. All cases are treated as if they started at t=0.",
    )
    parser.add_argument(
        "--total-cases",
        type=int,
        default=None,
        help="Override denominator used for cumulative success rate. Defaults to requiring equal case counts.",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_case_index(case_dir: str) -> Optional[int]:
    match = re.search(r"case_(\d+)$", case_dir)
    if not match:
        return None
    return int(match.group(1))


def load_pyct_records(pyct_root: str) -> Tuple[List[Dict[str, object]], Counter]:
    pattern = os.path.join(pyct_root, "case_*", "stats.json")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No stats.json files found under {pyct_root}")

    records: List[Dict[str, object]] = []
    statuses: Counter = Counter()

    for path in paths:
        data = load_json(path)
        meta = data.get("meta", {})
        summary = data.get("summary", {})
        case_dir = os.path.basename(os.path.dirname(path))
        idx = meta.get("idx")
        if idx is None:
            idx = parse_case_index(case_dir)
        if idx is None:
            raise ValueError(f"Cannot determine case index for {path}")

        status = str(meta.get("status", "unknown"))
        duration = float(summary.get("total_wall_time", 0.0))
        records.append(
            {
                "index": int(idx),
                "case_name": case_dir,
                "success": status == "success",
                "duration": duration,
                "status": status,
                "tool": "PyCT",
                "source_root": pyct_root,
                "source_path": path,
                "original_label": meta.get("original_label"),
                "attack_label": meta.get("attack_label"),
                "is_finish": meta.get("is_finish"),
                "is_timeout": meta.get("is_timeout"),
                "model_name": meta.get("model_name"),
                "attack_mode": meta.get("attack_mode"),
                "score_alpha": meta.get("score_alpha"),
                "symbolic_path_threshold": meta.get("symbolic_path_threshold"),
                "ton": meta.get("ton"),
                "constraint_build_timeout": meta.get("constraint_build_timeout"),
                "constraint_build_timeout_seconds": meta.get("constraint_build_timeout_seconds"),
                "total_cpu_time": summary.get("total_cpu_time"),
                "total_iter": summary.get("total_iter"),
                "execute_wall_time_total": summary.get("execute_wall_time_total"),
                "solve_constraint_wall_time_total": summary.get("solve_constraint_wall_time_total"),
            }
        )
        statuses[status] += 1

    records.sort(key=lambda item: item["index"])
    return records, statuses


def _read_pointer(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = f.read().strip()
    except OSError:
        return None
    if not value:
        return None
    return value if os.path.exists(value) else None


def _newest_run_dir(runs_root: str) -> Optional[str]:
    latest_run = None
    latest_mtime = None
    for root, dirs, _ in os.walk(runs_root):
        for name in dirs:
            if not name.startswith("run_"):
                continue
            path = os.path.join(root, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
                latest_run = path
    return latest_run


def resolve_opa_pickle(opa_path: str) -> str:
    if os.path.isfile(opa_path):
        return opa_path

    if not os.path.isdir(opa_path):
        raise FileNotFoundError(f"one-pixel-attack path not found: {opa_path}")

    direct_candidates = []
    for name in sorted(os.listdir(opa_path)):
        full = os.path.join(opa_path, name)
        if not os.path.isfile(full):
            continue
        if name.endswith("_results.pkl"):
            direct_candidates.append(full)
    for preferred in direct_candidates:
        return preferred

    results_pkl = os.path.join(opa_path, "results.pkl")
    if os.path.isfile(results_pkl):
        return results_pkl

    checkpoint_pkl = os.path.join(opa_path, "checkpoint.pkl")
    if os.path.isfile(checkpoint_pkl):
        return checkpoint_pkl

    for pointer_name in ("latest_run.txt", "latest_untargeted_run.txt", "latest_targeted_run.txt"):
        pointed = _read_pointer(os.path.join(opa_path, pointer_name))
        if pointed:
            return resolve_opa_pickle(pointed)

    runs_root = os.path.join(opa_path, "runs")
    if os.path.isdir(runs_root):
        latest_run = _newest_run_dir(runs_root)
        if latest_run:
            return resolve_opa_pickle(latest_run)

    recursive_results = []
    recursive_checkpoints = []
    for root, _, files in os.walk(opa_path):
        for name in files:
            full = os.path.join(root, name)
            if name.endswith("_results.pkl"):
                recursive_results.append(full)
            elif name == "results.pkl":
                recursive_results.append(full)
            elif name == "checkpoint.pkl":
                recursive_checkpoints.append(full)

    if recursive_results:
        recursive_results.sort(key=os.path.getmtime, reverse=True)
        return recursive_results[0]
    if recursive_checkpoints:
        recursive_checkpoints.sort(key=os.path.getmtime, reverse=True)
        return recursive_checkpoints[0]

    raise FileNotFoundError(f"Could not resolve one-pixel-attack pickle from {opa_path}")


def load_pickle(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Loading {path} requires optional Python modules during pickle deserialization: {exc}. "
            "Run this script in an environment that has the one-pixel-attack dependencies installed."
        ) from exc


def _stringify_jsonish(value: object) -> str:
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(value)


def record_from_mapping(entry: Dict[str, object]) -> Dict[str, object]:
    index = entry.get("image")
    if index is None:
        index = entry.get("idx")
    if index is None:
        raise ValueError(f"one-pixel-attack record is missing image/index: {entry!r}")

    duration = entry.get("duration")
    success = entry.get("success")
    if duration is None or success is None:
        raise ValueError(f"one-pixel-attack record is missing duration/success: {entry!r}")

    return {
        "index": int(index),
        "success": bool(success),
        "duration": float(duration),
        "status": str(entry.get("stop_reason", "unknown")),
        "tool": "one-pixel-attack",
        "case_name": f"image_{int(index)}",
        "image": int(index),
        "pixels": entry.get("pixels"),
        "true": entry.get("true"),
        "predicted": entry.get("predicted"),
        "original_pred": entry.get("original_pred"),
        "prediction_flipped": entry.get("prediction_flipped"),
        "original_correct": entry.get("original_correct"),
        "stop_reason": entry.get("stop_reason"),
        "timed_out": entry.get("timed_out"),
        "de_nit": entry.get("de_nit"),
        "de_nfev": entry.get("de_nfev"),
        "cdiff": entry.get("cdiff"),
        "prior_probs_json": _stringify_jsonish(entry.get("prior_probs")),
        "predicted_probs_json": _stringify_jsonish(entry.get("predicted_probs")),
        "perturbation_json": _stringify_jsonish(entry.get("perturbation")),
    }


def record_from_sequence(entry: Sequence[object]) -> Dict[str, object]:
    if len(entry) < len(OPA_RESULT_COLUMNS):
        raise ValueError(
            "Unsupported one-pixel-attack list result length "
            f"{len(entry)}; expected at least {len(OPA_RESULT_COLUMNS)}"
        )

    mapped = dict(zip(OPA_RESULT_COLUMNS, entry))
    return {
        "index": int(mapped["image"]),
        "success": bool(mapped["success"]),
        "duration": float(mapped["duration"]),
        "status": str(mapped.get("stop_reason", "unknown")),
        "tool": "one-pixel-attack",
        "case_name": f"image_{int(mapped['image'])}",
        "image": int(mapped["image"]),
        "pixels": mapped.get("pixels"),
        "true": mapped.get("true"),
        "predicted": mapped.get("predicted"),
        "original_pred": mapped.get("original_pred"),
        "prediction_flipped": mapped.get("prediction_flipped"),
        "original_correct": mapped.get("original_correct"),
        "stop_reason": mapped.get("stop_reason"),
        "timed_out": mapped.get("timed_out"),
        "de_nit": mapped.get("de_nit"),
        "de_nfev": mapped.get("de_nfev"),
        "cdiff": mapped.get("cdiff"),
        "prior_probs_json": _stringify_jsonish(mapped.get("prior_probs")),
        "predicted_probs_json": _stringify_jsonish(mapped.get("predicted_probs")),
        "perturbation_json": _stringify_jsonish(mapped.get("perturbation")),
    }


def load_opa_records(opa_path: str) -> Tuple[str, List[Dict[str, object]]]:
    resolved_path = resolve_opa_pickle(opa_path)
    payload = load_pickle(resolved_path)

    if isinstance(payload, dict) and "results" in payload:
        entries = payload["results"]
    elif isinstance(payload, list):
        entries = payload
    else:
        raise TypeError(
            f"Unsupported one-pixel-attack payload type {type(payload).__name__} from {resolved_path}"
        )

    records: List[Dict[str, object]] = []
    for entry in entries:
        if isinstance(entry, dict):
            records.append(record_from_mapping(entry))
        elif isinstance(entry, (list, tuple)):
            records.append(record_from_sequence(entry))
        else:
            raise TypeError(f"Unsupported one-pixel-attack record type: {type(entry).__name__}")

    for record in records:
        record["source_path"] = resolved_path

    records.sort(key=lambda item: item["index"])
    return resolved_path, records


def make_cumulative(records: Sequence[Dict[str, object]], total_cases: int) -> Dict[str, object]:
    cumulative = []
    total_time = 0.0
    success_count = 0
    status_counter = Counter()

    for record in records:
        total_time += float(record["duration"])
        success_count += int(bool(record["success"]))
        status_counter[str(record.get("status", "unknown"))] += 1
        cumulative.append(
            {
                "index": int(record["index"]),
                "duration": float(record["duration"]),
                "cum_time": total_time,
                "cum_success": success_count,
                "rate": success_count / total_cases,
                "success": bool(record["success"]),
                "status": str(record.get("status", "unknown")),
            }
        )

    return {
        "records": list(records),
        "cumulative": cumulative,
        "successes": success_count,
        "total_time": total_time,
        "status_counter": status_counter,
        "final_rate": (success_count / total_cases) if total_cases else 0.0,
    }


def make_parallel_summary(
    records: Sequence[Dict[str, object]],
    total_cases: int,
    time_horizon: float,
) -> Dict[str, object]:
    success_times = sorted(
        min(float(record["duration"]), time_horizon)
        for record in records
        if bool(record["success"])
    )

    curve = []
    success_count = 0
    for time_value in success_times:
        success_count += 1
        curve.append(
            {
                "time": time_value,
                "cum_success": success_count,
                "rate": success_count / total_cases,
            }
        )

    status_counter = Counter(str(record.get("status", "unknown")) for record in records)
    median_success_time = median(success_times) if success_times else None
    first_success_time = success_times[0] if success_times else None
    success_over_horizon = sum(
        1 for record in records if bool(record["success"]) and float(record["duration"]) > time_horizon
    )

    return {
        "records": list(records),
        "curve": curve,
        "successes": len(success_times),
        "status_counter": status_counter,
        "final_rate": (len(success_times) / total_cases) if total_cases else 0.0,
        "median_success_time": median_success_time,
        "first_success_time": first_success_time,
        "time_horizon": time_horizon,
        "success_over_horizon": success_over_horizon,
    }


def resolve_total_cases(pyct_records: Sequence[Dict[str, object]], opa_records: Sequence[Dict[str, object]], override: Optional[int]) -> int:
    pyct_count = len(pyct_records)
    opa_count = len(opa_records)
    if override is not None:
        return override
    if pyct_count != opa_count:
        raise ValueError(
            "Case-count mismatch with no --total-cases override: "
            f"PyCT has {pyct_count}, one-pixel-attack has {opa_count}"
        )
    return pyct_count


def find_first_cross(pyct_points: Sequence[Dict[str, object]], opa_points: Sequence[Dict[str, object]]) -> Optional[Tuple[float, float, float]]:
    times = sorted({point["time"] for point in pyct_points} | {point["time"] for point in opa_points})
    py_idx = 0
    opa_idx = 0
    py_rate = 0.0
    opa_rate = 0.0

    for time_value in times:
        while py_idx < len(pyct_points) and pyct_points[py_idx]["time"] <= time_value + 1e-12:
            py_rate = float(pyct_points[py_idx]["rate"])
            py_idx += 1
        while opa_idx < len(opa_points) and opa_points[opa_idx]["time"] <= time_value + 1e-12:
            opa_rate = float(opa_points[opa_idx]["rate"])
            opa_idx += 1
        if py_rate > opa_rate:
            return time_value, py_rate, opa_rate
    return None


def build_plot_rows(
    pyct_points: Sequence[Dict[str, object]],
    opa_points: Sequence[Dict[str, object]],
    total_cases: int,
    time_horizon: float,
    first_cross: Optional[Tuple[float, float, float]],
) -> List[Dict[str, object]]:
    times = sorted({0.0, float(time_horizon)} | {float(point["time"]) for point in pyct_points} | {float(point["time"]) for point in opa_points})
    py_idx = 0
    opa_idx = 0
    py_rate = 0.0
    opa_rate = 0.0
    py_success = 0
    opa_success = 0
    cross_time = None if first_cross is None else float(first_cross[0])
    rows: List[Dict[str, object]] = []

    for time_value in times:
        while py_idx < len(pyct_points) and float(pyct_points[py_idx]["time"]) <= time_value + 1e-12:
            py_rate = float(pyct_points[py_idx]["rate"])
            py_success = int(pyct_points[py_idx]["cum_success"])
            py_idx += 1
        while opa_idx < len(opa_points) and float(opa_points[opa_idx]["time"]) <= time_value + 1e-12:
            opa_rate = float(opa_points[opa_idx]["rate"])
            opa_success = int(opa_points[opa_idx]["cum_success"])
            opa_idx += 1

        rows.append(
            {
                "time_s": round(float(time_value), 10),
                "pyct_success_rate": py_rate,
                "one_pixel_success_rate": opa_rate,
                "pyct_cumulative_success": py_success,
                "one_pixel_cumulative_success": opa_success,
                "total_cases": total_cases,
                "is_crossover_time": bool(cross_time is not None and abs(time_value - cross_time) <= 1e-12),
            }
        )

    return rows


def write_plot_csv(
    path: str,
    pyct_summary: Dict[str, object],
    opa_summary: Dict[str, object],
    total_cases: int,
    time_horizon: float,
    first_cross: Optional[Tuple[float, float, float]],
) -> None:
    rows = build_plot_rows(pyct_summary["curve"], opa_summary["curve"], total_cases, time_horizon, first_cross)
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_s",
                "pyct_success_rate",
                "one_pixel_success_rate",
                "pyct_cumulative_success",
                "one_pixel_cumulative_success",
                "total_cases",
                "is_crossover_time",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_rows(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_raw_csvs(
    output_dir: str,
    model_name: str,
    pyct_records: Sequence[Dict[str, object]],
    opa_records: Sequence[Dict[str, object]],
) -> List[str]:
    safe_model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_") or "model"
    export_dir = os.path.join(output_dir, f"{safe_model_name}_raw")
    os.makedirs(export_dir, exist_ok=True)

    pyct_rows = [
        {
            "tool": row.get("tool"),
            "index": row.get("index"),
            "case_name": row.get("case_name"),
            "success": row.get("success"),
            "status": row.get("status"),
            "duration_s": row.get("duration"),
            "is_finish": row.get("is_finish"),
            "is_timeout": row.get("is_timeout"),
            "original_label": row.get("original_label"),
            "attack_label": row.get("attack_label"),
            "model_name": row.get("model_name"),
            "attack_mode": row.get("attack_mode"),
            "score_alpha": row.get("score_alpha"),
            "symbolic_path_threshold": row.get("symbolic_path_threshold"),
            "ton": row.get("ton"),
            "constraint_build_timeout": row.get("constraint_build_timeout"),
            "constraint_build_timeout_seconds": row.get("constraint_build_timeout_seconds"),
            "total_cpu_time": row.get("total_cpu_time"),
            "total_iter": row.get("total_iter"),
            "execute_wall_time_total": row.get("execute_wall_time_total"),
            "solve_constraint_wall_time_total": row.get("solve_constraint_wall_time_total"),
            "source_path": row.get("source_path"),
        }
        for row in pyct_records
    ]
    pyct_fields = list(pyct_rows[0].keys()) if pyct_rows else [
        "tool", "index", "case_name", "success", "status", "duration_s", "source_path"
    ]

    opa_rows = [
        {
            "tool": row.get("tool"),
            "index": row.get("index"),
            "case_name": row.get("case_name"),
            "success": row.get("success"),
            "status": row.get("status"),
            "duration_s": row.get("duration"),
            "image": row.get("image"),
            "pixels": row.get("pixels"),
            "true": row.get("true"),
            "predicted": row.get("predicted"),
            "original_pred": row.get("original_pred"),
            "prediction_flipped": row.get("prediction_flipped"),
            "original_correct": row.get("original_correct"),
            "stop_reason": row.get("stop_reason"),
            "timed_out": row.get("timed_out"),
            "de_nit": row.get("de_nit"),
            "de_nfev": row.get("de_nfev"),
            "cdiff": row.get("cdiff"),
            "prior_probs_json": row.get("prior_probs_json"),
            "predicted_probs_json": row.get("predicted_probs_json"),
            "perturbation_json": row.get("perturbation_json"),
            "source_path": row.get("source_path"),
        }
        for row in opa_records
    ]
    opa_fields = list(opa_rows[0].keys()) if opa_rows else [
        "tool", "index", "case_name", "success", "status", "duration_s", "source_path"
    ]

    combined_fieldnames = [
        "tool",
        "index",
        "case_name",
        "success",
        "status",
        "duration_s",
        "source_path",
        "image",
        "pixels",
        "true",
        "predicted",
        "original_pred",
        "prediction_flipped",
        "original_correct",
        "stop_reason",
        "timed_out",
        "de_nit",
        "de_nfev",
        "cdiff",
        "prior_probs_json",
        "predicted_probs_json",
        "perturbation_json",
        "is_finish",
        "is_timeout",
        "original_label",
        "attack_label",
        "model_name",
        "attack_mode",
        "score_alpha",
        "symbolic_path_threshold",
        "ton",
        "constraint_build_timeout",
        "constraint_build_timeout_seconds",
        "total_cpu_time",
        "total_iter",
        "execute_wall_time_total",
        "solve_constraint_wall_time_total",
    ]
    combined_rows = pyct_rows + opa_rows

    pyct_path = os.path.join(export_dir, f"{safe_model_name}_pyct_raw.csv")
    opa_path = os.path.join(export_dir, f"{safe_model_name}_one_pixel_raw.csv")
    combined_path = os.path.join(export_dir, f"{safe_model_name}_combined_raw.csv")

    _write_csv_rows(pyct_path, pyct_rows, pyct_fields)
    _write_csv_rows(opa_path, opa_rows, opa_fields)
    _write_csv_rows(combined_path, combined_rows, combined_fieldnames)
    return [pyct_path, opa_path, combined_path]


def nice_step(max_value: float, tick_count: int = 6) -> float:
    if max_value <= 0:
        return 1.0

    rough = max_value / max(tick_count, 1)
    exponent = math.floor(math.log10(rough))
    fraction = rough / (10 ** exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return nice_fraction * (10 ** exponent)


def build_x_ticks(max_value: float, tick_count: int = 6) -> List[float]:
    step = nice_step(max_value, tick_count)
    ticks = [0.0]
    current = step
    while current <= max_value + 1e-9:
        ticks.append(float(current))
        current += step
    if abs(ticks[-1] - max_value) > 1e-9:
        ticks.append(float(max_value))
    return ticks


def build_step_points(
    cumulative_points: Sequence[Dict[str, object]],
    end_x: Optional[float] = None,
) -> List[Tuple[float, float]]:
    points = [(0.0, 0.0)]
    prev_rate = 0.0
    for point in cumulative_points:
        x = float(point["time"])
        rate = float(point["rate"])
        points.append((x, prev_rate))
        points.append((x, rate))
        prev_rate = rate
    if end_x is not None and points[-1][0] < end_x:
        points.append((float(end_x), prev_rate))
    return points


def build_clipped_step_points(
    cumulative_points: Sequence[Dict[str, object]],
    max_x: float,
) -> List[Tuple[float, float]]:
    points = [(0.0, 0.0)]
    prev_rate = 0.0

    for point in cumulative_points:
        x = float(point["time"])
        rate = float(point["rate"])
        if x > max_x:
            points.append((max_x, prev_rate))
            return points
        points.append((x, prev_rate))
        points.append((x, rate))
        prev_rate = rate

    if points[-1][0] < max_x:
        points.append((max_x, prev_rate))
    return points


def format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def format_seconds_for_tick(value: float) -> str:
    if value == 0:
        return "0s"
    # if value >= 3600:
    #     return f"{value:.0f}s / {value / 3600:.2f}h"
    # if value >= 60:
    #     return f"{value:.0f}s / {value / 60:.1f}m"
    return f"{value:.0f}s"


def svg_text(x: float, y: float, text: str, **attrs: object) -> str:
    joined = " ".join(f'{key}="{escape(str(value))}"' for key, value in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {joined}>{escape(text)}</text>'


def build_svg(
    output_path: str,
    model_name: str,
    pyct_label: str,
    opa_label: str,
    pyct_summary: Dict[str, object],
    opa_summary: Dict[str, object],
    total_cases: int,
    first_cross: Optional[Tuple[float, float, float]],
    time_horizon: float,
) -> None:
    width = 1280
    height = 840
    margin_left = 110
    margin_right = 40
    margin_top = 95
    margin_bottom = 115
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    plot_right = margin_left + plot_width
    plot_bottom = margin_top + plot_height

    max_x = float(time_horizon)
    x_ticks = build_x_ticks(max_x, tick_count=6)
    y_ticks = [i / 10 for i in range(11)]

    def map_x(value: float) -> float:
        if max_x <= 0:
            return float(margin_left)
        return margin_left + (value / max_x) * plot_width

    def map_y(value: float) -> float:
        return plot_bottom - value * plot_height

    pyct_points = build_step_points(pyct_summary["curve"], end_x=max_x)
    opa_points = build_step_points(opa_summary["curve"], end_x=max_x)

    pyct_poly = " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in pyct_points)
    opa_poly = " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in opa_points)

    bg = "#fbfcfe"
    border = "#d7dfeb"
    grid = "#e6ecf3"
    text = "#102033"
    muted = "#5b6b81"
    pyct_color = "#0f766e"
    opa_color = "#c2410c"
    cross_color = "#475569"
    box_fill = "#ffffff"

    elements: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="{box_fill}" stroke="{border}"/>',
    ]

    title = f"{model_name}: success rate vs elapsed time"
    subtitle = f"Parallel-run assumption, denominator={total_cases}, all cases start at t=0, horizon={time_horizon:.0f}s"
    elements.append(svg_text(margin_left, 42, title, fill=text, **{"font-size": 28, "font-family": "DejaVu Sans, Arial, sans-serif", "font-weight": 700}))
    elements.append(svg_text(margin_left, 70, subtitle, fill=muted, **{"font-size": 16, "font-family": "DejaVu Sans, Arial, sans-serif"}))

    for tick in x_ticks:
        x = map_x(tick)
        elements.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="{grid}" stroke-width="1"/>')
        elements.append(svg_text(x, plot_bottom + 28, format_seconds_for_tick(tick), fill=muted, **{"font-size": 12, "font-family": "DejaVu Sans, Arial, sans-serif", "text-anchor": "middle"}))

    for tick in y_ticks:
        y = map_y(tick)
        elements.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="{grid}" stroke-width="1"/>')
        elements.append(svg_text(margin_left - 14, y + 4, f"{tick:.1f}", fill=muted, **{"font-size": 12, "font-family": "DejaVu Sans, Arial, sans-serif", "text-anchor": "end"}))

    elements.append(f'<polyline fill="none" stroke="{pyct_color}" stroke-width="3.5" points="{pyct_poly}"/>')
    elements.append(f'<polyline fill="none" stroke="{opa_color}" stroke-width="3.5" points="{opa_poly}"/>')

    legend_x = plot_right - 300
    legend_y = margin_top + 22
    elements.append(f'<rect x="{legend_x}" y="{legend_y}" width="220" height="76" rx="10" fill="{box_fill}" stroke="{border}"/>')
    elements.append(f'<line x1="{legend_x + 18}" y1="{legend_y + 24}" x2="{legend_x + 62}" y2="{legend_y + 24}" stroke="{pyct_color}" stroke-width="4"/>')
    elements.append(svg_text(legend_x + 74, legend_y + 29, pyct_label, fill=text, **{"font-size": 15, "font-family": "DejaVu Sans, Arial, sans-serif"}))
    elements.append(f'<line x1="{legend_x + 18}" y1="{legend_y + 52}" x2="{legend_x + 62}" y2="{legend_y + 52}" stroke="{opa_color}" stroke-width="4"/>')
    elements.append(svg_text(legend_x + 74, legend_y + 57, opa_label, fill=text, **{"font-size": 15, "font-family": "DejaVu Sans, Arial, sans-serif"}))

    summary_box_x = margin_left + 18
    summary_box_y = margin_top + 18
    summary_box_w = 260
    summary_box_h = 110
    elements.append(f'<rect x="{summary_box_x}" y="{summary_box_y}" width="{summary_box_w}" height="{summary_box_h}" rx="10" fill="{box_fill}" stroke="{border}"/>')
    elements.append(svg_text(summary_box_x + 16, summary_box_y + 28, f"{pyct_label}: {pyct_summary['successes']}/{total_cases} ({pyct_summary['final_rate']:.3f})", fill=pyct_color, **{"font-size": 15, "font-family": "DejaVu Sans, Arial, sans-serif", "font-weight": 700}))
    pyct_median = pyct_summary["median_success_time"]
    pyct_median_text = "Median success time: n/a" if pyct_median is None else f"Median success time: {pyct_median:.2f}s"
    elements.append(svg_text(summary_box_x + 16, summary_box_y + 50, pyct_median_text, fill=text, **{"font-size": 13, "font-family": "DejaVu Sans, Arial, sans-serif"}))
    elements.append(svg_text(summary_box_x + 16, summary_box_y + 74, f"{opa_label}: {opa_summary['successes']}/{total_cases} ({opa_summary['final_rate']:.3f})", fill=opa_color, **{"font-size": 15, "font-family": "DejaVu Sans, Arial, sans-serif", "font-weight": 700}))
    opa_median = opa_summary["median_success_time"]
    opa_median_text = "Median success time: n/a" if opa_median is None else f"Median success time: {opa_median:.2f}s"
    elements.append(svg_text(summary_box_x + 16, summary_box_y + 96, opa_median_text, fill=text, **{"font-size": 13, "font-family": "DejaVu Sans, Arial, sans-serif"}))

    if first_cross is not None:
        cross_x, py_rate, opa_rate = first_cross
        x = map_x(cross_x)
        y = map_y(py_rate)
        elements.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x:.2f}" y2="{plot_bottom}" stroke="{cross_color}" stroke-width="2" stroke-dasharray="8 6"/>')
        elements.append(f'<line x1="{x:.2f}" y1="{plot_bottom:.2f}" x2="{x:.2f}" y2="{plot_bottom + 10:.2f}" stroke="{cross_color}" stroke-width="2"/>')
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{pyct_color}" stroke="{box_fill}" stroke-width="2"/>')
        # label = f"{pyct_label} first exceeds {opa_label} at ~{cross_x:.2f}s"
        # label_x = summary_box_x + 16
        anchor = "start"
        # elements.append(svg_text(label_x, margin_top + 138, label, fill=cross_color, **{"font-size": 14, "font-family": "DejaVu Sans, Arial, sans-serif", "text-anchor": anchor, "font-weight": 700}))
        delta = py_rate - opa_rate
        # elements.append(svg_text(label_x, margin_top + 160, f"At crossover: {pyct_label}={py_rate:.3f}, {opa_label}={opa_rate:.3f}, gap={delta:.3f}", fill=muted, **{"font-size": 12, "font-family": "DejaVu Sans, Arial, sans-serif", "text-anchor": anchor}))
        elements.append(svg_text(x+5, plot_bottom + 28, f"{cross_x:.2f}s", fill=cross_color, **{"font-size": 12, "font-family": "DejaVu Sans, Arial, sans-serif", "text-anchor": "middle", "font-weight": 700}))

    elements.append(svg_text((margin_left + plot_right) / 2, height - 28, "Elapsed wall time under parallel-run assumption", fill=text, **{"font-size": 16, "font-family": "DejaVu Sans, Arial, sans-serif", "text-anchor": "middle", "font-weight": 700}))
    elements.append(f'<g transform="translate(28 {(margin_top + plot_bottom) / 2:.2f}) rotate(-90)">')
    elements.append(svg_text(0, 0, "Success rate by time t", fill=text, **{"font-size": 16, "font-family": "DejaVu Sans, Arial, sans-serif", "text-anchor": "middle", "font-weight": 700}))
    elements.append("</g>")

    elements.append("</svg>")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(elements) + "\n")


def print_summary(
    pyct_summary: Dict[str, object],
    opa_summary: Dict[str, object],
    total_cases: int,
    resolved_opa_path: str,
    first_cross: Optional[Tuple[float, float, float]],
    time_horizon: float,
) -> None:
    print(f"Resolved one-pixel-attack pickle: {resolved_opa_path}")
    print(f"X-axis mode: parallel-run assumption, horizon={time_horizon:.2f}s")
    pyct_median = pyct_summary["median_success_time"]
    opa_median = opa_summary["median_success_time"]
    pyct_median_text = "n/a" if pyct_median is None else f"{float(pyct_median):.2f}s"
    opa_median_text = "n/a" if opa_median is None else f"{float(opa_median):.2f}s"
    print(
        "PyCT: "
        f"cases={len(pyct_summary['records'])} "
        f"success={pyct_summary['successes']}/{total_cases} "
        f"success_rate={pyct_summary['final_rate']:.4f} "
        f"median_success_time={pyct_median_text}"
    )
    print(
        "one-pixel-attack: "
        f"cases={len(opa_summary['records'])} "
        f"success={opa_summary['successes']}/{total_cases} "
        f"success_rate={opa_summary['final_rate']:.4f} "
        f"median_success_time={opa_median_text}"
    )
    if first_cross is None:
        print("Crossover: PyCT never exceeds one-pixel-attack within the elapsed-time horizon.")
    else:
        cross_x, py_rate, opa_rate = first_cross
        print(
            "Crossover: "
            f"PyCT first exceeds one-pixel-attack at {cross_x:.2f}s "
            f"(PyCT={py_rate:.4f}, one-pixel-attack={opa_rate:.4f})"
        )


def main() -> int:
    args = parse_args()

    pyct_records, pyct_statuses = load_pyct_records(args.pyct_root)
    resolved_opa_path, opa_records = load_opa_records(args.opa_path)
    total_cases = resolve_total_cases(pyct_records, opa_records, args.total_cases)
    time_horizon = float(args.time_horizon)

    if args.total_cases is None and len(pyct_records) != len(opa_records):
        warn(
            f"Case counts differ without explicit --total-cases: PyCT={len(pyct_records)} "
            f"one-pixel-attack={len(opa_records)}"
        )
    if args.total_cases is not None:
        if len(pyct_records) != total_cases:
            warn(f"PyCT record count ({len(pyct_records)}) does not equal total_cases ({total_cases}).")
        if len(opa_records) != total_cases:
            warn(
                f"one-pixel-attack record count ({len(opa_records)}) "
                f"does not equal total_cases ({total_cases})."
            )

    extra_pyct_statuses = {
        status: count
        for status, count in pyct_statuses.items()
        if status not in {"success", "timeout", "exhausted"}
    }
    if extra_pyct_statuses:
        warn(
            "PyCT has non-terminal-or-unexpected statuses counted as non-success: "
            + ", ".join(f"{status}={count}" for status, count in sorted(extra_pyct_statuses.items()))
        )

    pyct_summary = make_parallel_summary(pyct_records, total_cases, time_horizon)
    opa_summary = make_parallel_summary(opa_records, total_cases, time_horizon)
    if pyct_summary["success_over_horizon"] > 0:
        warn(
            f"PyCT has {pyct_summary['success_over_horizon']} success case(s) after the horizon; "
            f"they are clipped to {time_horizon:.2f}s."
        )
    if opa_summary["success_over_horizon"] > 0:
        warn(
            f"one-pixel-attack has {opa_summary['success_over_horizon']} success case(s) after the horizon; "
            f"they are clipped to {time_horizon:.2f}s."
        )

    first_cross = find_first_cross(pyct_summary["curve"], opa_summary["curve"])

    print_summary(pyct_summary, opa_summary, total_cases, resolved_opa_path, first_cross, time_horizon)
    raw_paths = write_raw_csvs(args.raw_output_dir, args.model_name, pyct_records, opa_records)
    for raw_path in raw_paths:
        print(f"Saved raw CSV: {raw_path}")

    if args.output_svg:
        build_svg(
            output_path=args.output_svg,
            model_name=args.model_name,
            pyct_label=args.pyct_label,
            opa_label=args.opa_label,
            pyct_summary=pyct_summary,
            opa_summary=opa_summary,
            total_cases=total_cases,
            first_cross=first_cross,
            time_horizon=time_horizon,
        )
        print(f"Saved SVG: {args.output_svg}")
    if args.output_csv:
        write_plot_csv(args.output_csv, pyct_summary, opa_summary, total_cases, time_horizon, first_cross)
        print(f"Saved CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
