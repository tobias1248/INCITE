#!/usr/bin/env python3
import argparse
import fnmatch
import json
import math
import os
import statistics
from typing import Any, Dict, Iterable, List, Optional


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


def _percentile(sorted_vals: List[float], pct: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pct = min(max(pct, 0.0), 1.0)
    k = (len(sorted_vals) - 1) * pct
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    values_sorted = sorted(values)
    return {
        "count": float(len(values_sorted)),
        "min": float(values_sorted[0]),
        "median": float(statistics.median(values_sorted)),
        "mean": float(statistics.mean(values_sorted)),
        "max": float(values_sorted[-1]),
        "p90": float(_percentile(values_sorted, 0.9) or values_sorted[-1]),
    }


def _format_stat(label: str, summary: Dict[str, float]) -> str:
    if not summary:
        return f"{label}: (no data)"
    return (
        f"{label}: n={int(summary['count'])} "
        f"median={summary['median']:.4g} p90={summary['p90']:.4g} "
        f"max={summary['max']:.4g}"
    )


def _increment(counter: Dict[Any, int], key: Any) -> None:
    counter[key] = counter.get(key, 0) + 1


def _collect_metrics(data: Dict[str, Any], metric_keys: Iterable[str]) -> Dict[str, Optional[float]]:
    def get_nested(d: Dict[str, Any], dotted_key: str) -> Optional[Any]:
        cur: Any = d
        for part in dotted_key.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    metrics: Dict[str, Optional[float]] = {}
    for key in metric_keys:
        metrics[key] = _safe_float(get_nested(data, key))
    return metrics


def _count_le(values: List[float], threshold: float) -> int:
    return sum(1 for v in values if v <= threshold)


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
    missing_entries = 0

    counts: Dict[str, int] = {
        "success": 0,
        "timeout": 0,
        "exhausted": 0,
        "incomplete": 0,
        "finished": 0,
    }

    status_counts: Dict[Any, int] = {}

    metric_keys = [
        "summary.total_wall_time",
        "solver.solver_time_total",
    ]
    metric_values: Dict[str, List[float]] = {k: [] for k in metric_keys}

    # constraint complexity
    status_counter: Dict[Any, int] = {}
    all_assert: List[float] = []
    all_byte: List[float] = []
    all_path: List[float] = []
    all_total_time: List[float] = []

    sat_assert: List[float] = []
    sat_byte: List[float] = []
    sat_path: List[float] = []
    sat_total_time: List[float] = []

    unsat_assert: List[float] = []
    unsat_byte: List[float] = []
    unsat_path: List[float] = []
    unsat_total_time: List[float] = []

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

        metrics = _collect_metrics(data, metric_keys)
        for key, value in metrics.items():
            if value is not None:
                metric_values[key].append(value)

        entries = (data.get("constraint_complexity") or {}).get("entries")
        if not isinstance(entries, list):
            missing_entries += 1
            continue

        for entry in entries:
            status = entry.get("status")
            _increment(status_counter, status)

            assert_num = _safe_float(entry.get("assert_num"))
            byte = _safe_float(entry.get("byte"))
            path_len = _safe_float(entry.get("path_len"))
            total_time = _safe_float(entry.get("total_time"))

            if assert_num is not None:
                all_assert.append(assert_num)
            if byte is not None:
                all_byte.append(byte)
            if path_len is not None:
                all_path.append(path_len)
            if total_time is not None:
                all_total_time.append(total_time)

            if status == "sat":
                if assert_num is not None:
                    sat_assert.append(assert_num)
                if byte is not None:
                    sat_byte.append(byte)
                if path_len is not None:
                    sat_path.append(path_len)
                if total_time is not None:
                    sat_total_time.append(total_time)
            elif status == "unsat":
                if assert_num is not None:
                    unsat_assert.append(assert_num)
                if byte is not None:
                    unsat_byte.append(byte)
                if path_len is not None:
                    unsat_path.append(path_len)
                if total_time is not None:
                    unsat_total_time.append(total_time)

    summary_payload = {
        "path": args.path,
        "cases": {
            "total": total,
            "parse_errors": parse_errors,
            "missing_entries": missing_entries,
        },
        "outcome": {
            **counts,
            "success_rate": (counts["success"] / total) if total else 0.0,
        },
        "status_counts": status_counts,
        "time": {
            "total_wall_time": _summarize(metric_values["summary.total_wall_time"]),
            "solver_time_total": _summarize(metric_values["solver.solver_time_total"]),
        },
        "constraint_complexity": {
            "status_counts": status_counter,
            "all": {
                "assert_num": _summarize(all_assert),
                "byte": _summarize(all_byte),
                "path_len": _summarize(all_path),
                "total_time": _summarize(all_total_time),
            },
            "sat": {
                "assert_num": _summarize(sat_assert),
                "byte": _summarize(sat_byte),
                "path_len": _summarize(sat_path),
                "total_time": _summarize(sat_total_time),
            },
            "unsat": {
                "assert_num": _summarize(unsat_assert),
                "byte": _summarize(unsat_byte),
                "path_len": _summarize(unsat_path),
                "total_time": _summarize(unsat_total_time),
            },
        },
    }

    thresholds = [300, 500, 1000, 5000]
    sat_rate_by_threshold: Dict[int, Dict[str, float]] = {}
    all_assert_vals = [v for v in all_assert if v is not None]
    sat_assert_vals = [v for v in sat_assert if v is not None]
    for th in thresholds:
        total_le = _count_le(all_assert_vals, th)
        sat_le = _count_le(sat_assert_vals, th)
        sat_rate = (sat_le / total_le) if total_le else 0.0
        sat_rate_by_threshold[th] = {
            "sat": float(sat_le),
            "total": float(total_le),
            "sat_rate": float(sat_rate),
        }

    sat_small = _count_le(sat_assert_vals, 1000)
    sat_total = len(sat_assert_vals)
    summary_payload["constraint_complexity"]["sat_preference"] = {
        "sat_le_1000": sat_small,
        "sat_total": sat_total,
        "sat_le_1000_rate": (sat_small / sat_total) if sat_total else 0.0,
        "sat_rate_by_assert_num_threshold": sat_rate_by_threshold,
    }

    if args.json:
        print(json.dumps(summary_payload, indent=2))
        return 0

    print(f"Path: {args.path}")
    print(
        "Cases: total={total} success={success} timeout={timeout} incomplete={incomplete} "
        "finished={finished} (success_rate={rate:.1%})".format(
            total=total,
            success=counts["success"],
            timeout=counts["timeout"],
            incomplete=counts["incomplete"],
            finished=counts["finished"],
            rate=(counts["success"] / total) if total else 0.0,
        )
    )
    if parse_errors or missing_entries:
        print(f"Note: parse_errors={parse_errors} missing_entries={missing_entries}")

    print("Time:")
    print(f"  {_format_stat('total_wall_time', summary_payload['time']['total_wall_time'])}")
    print(f"  {_format_stat('solver_time_total', summary_payload['time']['solver_time_total'])}")

    print("Constraint complexity (all constraints):")
    print(f"  {_format_stat('assert_num', summary_payload['constraint_complexity']['all']['assert_num'])}")
    print(f"  {_format_stat('byte', summary_payload['constraint_complexity']['all']['byte'])}")
    print(f"  {_format_stat('path_len', summary_payload['constraint_complexity']['all']['path_len'])}")
    print(f"  {_format_stat('total_time', summary_payload['constraint_complexity']['all']['total_time'])}")
    if status_counter:
        status_line = " ".join(f"{k}={v}" for k, v in sorted(status_counter.items(), key=lambda x: str(x[0])))
        print(f"  status_counts: {status_line}")

    print("SAT preference (assert_num):")
    sat_pref = summary_payload["constraint_complexity"]["sat_preference"]
    if sat_total:
        print(
            "  sat<=1000: {sat}/{total} ({rate:.1%})".format(
                sat=sat_pref["sat_le_1000"],
                total=sat_pref["sat_total"],
                rate=sat_pref["sat_le_1000_rate"],
            )
        )
    else:
        print("  sat<=1000: (no sat data)")

    if sat_rate_by_threshold:
        line_parts = []
        for th in thresholds:
            info = sat_rate_by_threshold[th]
            line_parts.append(f"<= {th}: {info['sat']:.0f}/{info['total']:.0f} ({info['sat_rate']:.2%})")
        print("  sat_rate_by_threshold: " + "; ".join(line_parts))

    sat_summary = summary_payload["constraint_complexity"]["sat"]["assert_num"]
    unsat_summary = summary_payload["constraint_complexity"]["unsat"]["assert_num"]
    if sat_summary and unsat_summary:
        print(
            "  median_assert_num: sat={sat:.4g} unsat={unsat:.4g}".format(
                sat=sat_summary["median"],
                unsat=unsat_summary["median"],
            )
        )

    sat_time_summary = summary_payload["constraint_complexity"]["sat"]["total_time"]
    unsat_time_summary = summary_payload["constraint_complexity"]["unsat"]["total_time"]
    if sat_time_summary and unsat_time_summary:
        print(
            "  median_total_time: sat={sat:.4g}s unsat={unsat:.4g}s".format(
                sat=sat_time_summary["median"],
                unsat=unsat_time_summary["median"],
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
