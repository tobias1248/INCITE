#!/usr/bin/env python3
import argparse
import fnmatch
import json
import os
import statistics
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _get_nested(data: Dict[str, Any], dotted_key: str) -> Optional[Any]:
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _collect_files(root: str, pattern: str) -> List[str]:
    if os.path.isfile(root):
        return [root]
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if fnmatch.fnmatch(name, pattern):
                matches.append(os.path.join(dirpath, name))
    return sorted(matches)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    values_sorted = sorted(values)
    return {
        "count": float(len(values_sorted)),
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "mean": statistics.mean(values_sorted),
        "median": statistics.median(values_sorted),
        "sum": sum(values_sorted),
    }


def _ratio(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
    if numer is None or denom is None or denom == 0:
        return None
    return numer / denom


def _format_summary(name: str, summary: Dict[str, float]) -> str:
    if not summary:
        return f"{name}: (no data)"
    return (
        f"{name}: count={int(summary['count'])} "
        f"min={summary['min']:.4g} max={summary['max']:.4g} "
        f"mean={summary['mean']:.4g} median={summary['median']:.4g} "
        f"sum={summary['sum']:.4g}"
    )


def _increment(counter: Dict[Any, int], key: Any) -> None:
    counter[key] = counter.get(key, 0) + 1


def _collect_metrics(data: Dict[str, Any], metric_keys: Iterable[str]) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}
    for key in metric_keys:
        metrics[key] = _safe_float(_get_nested(data, key))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize stats.json metrics under a directory."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Root path containing stats.json files (or a single stats.json path).",
    )
    parser.add_argument(
        "--pattern",
        default="stats.json",
        help="Filename pattern to match (default: stats.json).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON instead of text.",
    )
    args = parser.parse_args()

    files = _collect_files(args.path, args.pattern)
    if not files:
        print("No files matched.")
        return 1

    total = 0
    parse_errors = 0
    counts: Dict[str, int] = {
        "success": 0,
        "timeout": 0,
        "exhausted": 0,
        "incomplete": 0,
        "finished": 0,
    }
    status_counts: Dict[Any, int] = {}
    attack_mode_counts: Dict[Any, int] = {}
    ton_counts: Dict[Any, int] = {}
    mixed_success_timeout = 0

    metric_keys = [
        "summary.total_wall_time",
        "summary.total_cpu_time",
        "summary.total_iter",
        "summary.attack_wall_time",
        "solver.sat",
        "solver.unsat",
        "solver.unknown",
        "solver.solver_time_total",
        "constraints.generated_total",
        "constraints.solved_total",
        "constraints.queue_max",
    ]
    metric_values: Dict[str, List[float]] = {k: [] for k in metric_keys}
    derived_values: Dict[str, List[float]] = {
        "solver_calls": [],
        "sat_rate": [],
        "solve_rate": [],
    }

    for path in files:
        total += 1
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            parse_errors += 1
            continue

        meta = data.get("meta") or {}
        attack_label = meta.get("attack_label", data.get("attack_label"))
        is_timeout = bool(meta.get("is_timeout", False))
        solve_all_ctr = bool(meta.get("solve_all_ctr", False))
        is_finish = bool(meta.get("is_finish", False))
        status = meta.get("status", data.get("status"))
        if status is not None:
            _increment(status_counts, status)

        attack_mode = meta.get("attack_mode")
        if attack_mode is not None:
            _increment(attack_mode_counts, attack_mode)

        ton = meta.get("ton")
        if ton is not None:
            _increment(ton_counts, ton)

        success = attack_label is not None
        exhausted = solve_all_ctr
        timeout = is_timeout
        incomplete = not success and not exhausted and not timeout
        if success:
            counts["success"] += 1
        if exhausted:
            counts["exhausted"] += 1
        if timeout:
            counts["timeout"] += 1
        if incomplete:
            counts["incomplete"] += 1
        if is_finish:
            counts["finished"] += 1
        if success and timeout:
            mixed_success_timeout += 1

        metrics = _collect_metrics(data, metric_keys)
        for key, value in metrics.items():
            if value is not None:
                metric_values[key].append(value)

        sat = metrics.get("solver.sat")
        unsat = metrics.get("solver.unsat")
        unknown = metrics.get("solver.unknown")
        generated = metrics.get("constraints.generated_total")
        solved = metrics.get("constraints.solved_total")

        solver_calls = None
        if sat is not None or unsat is not None or unknown is not None:
            solver_calls = (sat or 0.0) + (unsat or 0.0) + (unknown or 0.0)
            derived_values["solver_calls"].append(solver_calls)

        sat_rate = _ratio(sat, solver_calls)
        if sat_rate is not None:
            derived_values["sat_rate"].append(sat_rate)

        solve_rate = _ratio(solved, generated)
        if solve_rate is not None:
            derived_values["solve_rate"].append(solve_rate)

    summary_payload = {
        "total_files": total,
        "parse_errors": parse_errors,
        "counts": counts,
        "status_counts": status_counts,
        "attack_mode_counts": attack_mode_counts,
        "ton_counts": ton_counts,
        "mixed_success_timeout": mixed_success_timeout,
        "metrics": {k: _summarize(v) for k, v in metric_values.items()},
        "derived": {k: _summarize(v) for k, v in derived_values.items()},
        "definitions": {
            "success": "attack_label is not None",
            "exhausted": "solve_all_ctr is True",
            "timeout": "is_timeout is True",
            "incomplete": "not success and not exhausted and not timeout",
        },
    }

    if args.json:
        print(json.dumps(summary_payload, indent=2))
        return 0

    print(f"Path: {args.path}")
    print(f"Matched files: {total} (parse errors: {parse_errors})")
    print("Definitions:")
    for k, v in summary_payload["definitions"].items():
        print(f"  {k}: {v}")

    print("Counts:")
    print(
        "  total={total} success={success} timeout={timeout} "
        "exhausted={exhausted} incomplete={incomplete} finished={finished}".format(
            total=total,
            success=counts["success"],
            timeout=counts["timeout"],
            exhausted=counts["exhausted"],
            incomplete=counts["incomplete"],
            finished=counts["finished"],
        )
    )
    if mixed_success_timeout:
        print(f"  success+timeout cases: {mixed_success_timeout}")

    if status_counts:
        print("Status distribution:")
        for key in sorted(status_counts, key=lambda k: str(k)):
            print(f"  {key}: {status_counts[key]}")

    if attack_mode_counts:
        print("Attack mode distribution:")
        for key in sorted(attack_mode_counts, key=lambda k: str(k)):
            print(f"  {key}: {attack_mode_counts[key]}")

    if ton_counts:
        print("Ton distribution:")
        for key in sorted(ton_counts, key=lambda k: str(k)):
            print(f"  {key}: {ton_counts[key]}")

    print("Metrics:")
    for key in metric_keys:
        print(f"  {_format_summary(key, summary_payload['metrics'][key])}")

    print("Derived metrics:")
    for key in derived_values:
        print(f"  {_format_summary(key, summary_payload['derived'][key])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
