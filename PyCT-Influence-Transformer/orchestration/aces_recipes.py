"""Bounded sequential photo-control recipes within one case deadline.

Each attempt owns a process session so a deadline also stops its forward and
solver descendants. Only small progress envelopes cross the pipe; images and
ordinary PyCT statistics remain in the attempt's artifact directory.
"""
from __future__ import annotations

import copy
import itertools
import json
import logging
import multiprocessing
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

from libct.aces_like_multi import normalize_bounds, normalize_controls, parameter_name
from tasks.paths import get_save_dir_from_save_exp

log = logging.getLogger("ct.recipes")
SCHEMA_VERSION = "aces-recipes-v1"


def recipe_sequence(controls):
    selected = normalize_controls(controls)
    pairs = list(itertools.combinations(selected, 2))
    return pairs + ([selected] if len(selected) == 3 else [])


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _pwl_metadata(config):
    return {key: config[key] for key in (
        "pwl_region_budgets", "pwl_max_regions", "pwl_region_count", "pwl_max_abs_error",
        "pwl_error_tolerance", "pwl_build_history", "pwl_build_wall_time",
    ) if key in config}


def _send_event(connection, message):
    connection.send(dict(message, sent_at=time.monotonic()))


def _attempt_child(connection, payload, options, deadline, build_seconds):
    # Do not inherit the case worker's shutdown handler/Event in this session.
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        build_deadline = min(deadline, time.monotonic() + build_seconds)
        _send_event(connection, {"event": "building", "deadline": build_deadline})
        from libct.aces_like_pwl_nd import PWLBuildTimeout, PWLRegionBudgetError
        from libct.global_real import _input_mapping_to_rgb, build_aces_like_global_real_config

        config = dict(payload["global_real_config"], pwl_build_deadline=build_deadline)
        try:
            rgb, _ = _input_mapping_to_rgb(payload["in_dict"])
            config = build_aces_like_global_real_config(rgb, config)
        except (PWLBuildTimeout, PWLRegionBudgetError) as exc:
            _send_event(connection, {"event": "finished", "status": (
                "pwl_build_timeout" if isinstance(exc, PWLBuildTimeout) else "pwl_budget_exhausted"
            ), "reason": str(exc), "pwl": getattr(exc, "build_stats", {})})
            return
        payload["global_real_config"] = config
        remaining = deadline - time.monotonic()
        search_seconds = int(remaining)
        if search_seconds < 1:
            _send_event(connection, {"event": "finished", "status": "timeout",
                                    "reason": "Less than one whole second remains after PWL build",
                                    "timeout_phase": "integer_search_budget",
                                    "search_seconds_available": max(0.0, remaining),
                                    "pwl": _pwl_metadata(config)})
            return
        _send_event(connection, {"event": "searching", "pwl": _pwl_metadata(config)})
        from engine.executor import run

        # Initialization, forward and solver execution are all additionally
        # bounded by the supervising parent, including uninterruptible work.
        # The solver/runtime contract uses integer seconds. Round down so
        # discretization never extends the stage's monotonic deadline.
        run(**payload, **options, timeout=search_seconds, total_timeout=search_seconds,
            single_timeout=search_seconds)
        stats_path = Path(payload["save_exp"]["input_name"]) / "stats.json"
        data = json.loads(stats_path.read_text())
        meta = data["meta"]
        status = meta.get("status")
        if status not in ("success", "timeout", "exhausted", "incomplete", "error"):
            raise ValueError("Unexpected recipe search status: {!r}".format(status))
        _send_event(connection, {"event": "finished", "status": status,
                         "reason": meta.get("error_reason", status),
                         "original_label": meta.get("original_label"),
                         "attack_label": meta.get("attack_label"),
                         "pwl": _pwl_metadata(config)})
    except Exception as exc:
        log.exception("[RECIPE-ERROR] case=%s", payload.get("idx"))
        _send_event(connection, {"event": "finished", "status": "error",
                         "reason": "{}: {}".format(type(exc).__name__, exc)})
    finally:
        connection.close()


def _stop_attempt(process, session_ready):
    """Clean descendants even when the session leader has already exited."""
    # The child may have called setsid before its first pipe event is read.
    # Its pid cannot name the parent's group, so always try the session group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process.is_alive():
        process.terminate()
    process.join(timeout=0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.is_alive():
        process.kill()
    process.join(timeout=0.2)


def _run_isolated(payload, options, deadline, build_seconds, update):
    # Linux experiments use fork, as does the existing launcher/runtime.
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_attempt_child,
                              args=(send, payload, options, deadline, build_seconds))
    ready, phase, wait_deadline = False, "starting", deadline
    process.start()
    send.close()
    try:
        while True:
            remaining = wait_deadline - time.monotonic()
            # A timely phase/result may already be queued while the parent
            # was persisting a previous event. Inspect it before expiring the
            # old phase, and use the child's monotonic timestamp for lateness.
            if remaining <= 0 and not receive.poll():
                return {"status": "pwl_build_timeout" if phase == "building" else "timeout",
                        "reason": "{} deadline exceeded".format(phase)}
            if receive.poll(max(0.0, min(0.1, remaining))):
                try:
                    message = receive.recv()
                except EOFError:
                    return {"status": "error", "reason": "Recipe process closed without a result"}
                event = message.get("event")
                sent_at = message.get("sent_at")
                if sent_at is not None and (
                    type(sent_at) not in (int, float) or not 0 <= sent_at < float("inf")
                ):
                    return {"status": "error", "reason": "Invalid recipe event timestamp"}
                late = sent_at is not None and sent_at > wait_deadline
                if late and (event == "searching" or (
                    event == "finished" and message.get("status") in ("success", "exhausted", "incomplete")
                )):
                    return {"status": "pwl_build_timeout" if phase == "building" else "timeout",
                            "reason": "{} completed after deadline".format(phase)}
                if event == "building":
                    ready, phase = True, "building"
                    wait_deadline = min(deadline, message["deadline"])
                elif event == "searching":
                    phase, wait_deadline = "searching", deadline
                elif event != "finished":
                    return {"status": "error", "reason": "Unknown recipe process event"}
                update({key: value for key, value in message.items() if key not in ("event", "deadline", "sent_at")})
                if event == "finished":
                    return {key: value for key, value in message.items() if key not in ("event", "sent_at")}
            elif not process.is_alive():
                # Drain a result written just before exit before classifying a crash.
                if receive.poll():
                    continue
                return {"status": "error", "reason": "Recipe process exited without a result ({})".format(process.exitcode)}
    finally:
        _stop_attempt(process, ready)
        receive.close()


def _terminal_status(attempts, expected, deadline):
    statuses = [attempt["status"] for attempt in attempts]
    if "error" in statuses:
        return "error"
    if "success" in statuses:
        return "success"
    searched = [s for s in statuses if s not in ("pwl_budget_exhausted", "pwl_build_timeout")]
    if statuses and not searched:
        return "invalid"
    if time.monotonic() >= deadline or "timeout" in statuses:
        return "timeout"
    if len(attempts) < expected or any(s != "exhausted" for s in statuses):
        return "incomplete"
    return "exhausted"


def run_recipes(payload, *, timeout, **options):
    """Run selected pairs, then the triple; never grant a fresh case budget."""
    started = time.monotonic()
    if not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
        raise ValueError("Recipe timeout must be finite and positive")
    deadline = started + timeout
    payload = copy.deepcopy(payload)
    plan = payload.pop("aces_recipe_plan")
    controls = normalize_controls(plan["controls"])
    bounds = normalize_bounds(plan["controls"], plan["control_bounds"])
    budgets = list(plan["region_budgets"])
    build_seconds = plan["pwl_build_timeout_seconds"]
    if (plan.get("policy") != "staged" or not budgets
            or any(type(b) is not int or b < 1 for b in budgets)
            or any(b <= a for a, b in zip(budgets, budgets[1:]))
            or not isinstance(build_seconds, (int, float)) or not 0 < build_seconds < float("inf")):
        raise ValueError("Invalid staged recipe budgets")
    config = payload["global_real_config"]
    if (config.get("global_shift_kind") != "aces-multi" or config.get("hybrid_de_enabled")
            or config.get("shap_priority_enabled") is not True or config.get("probe_enabled") is not False
            or config.get("shap_sensitivity_enabled") is not False or payload.get("score_alpha") != 0):
        raise ValueError("Recipes require standalone SHAP-only, alpha=0 and No Probe")
    sequence = recipe_sequence(controls)
    bound_map = dict(zip(controls, bounds))
    parent = Path(get_save_dir_from_save_exp(
        payload["save_exp"], payload["model_name"], payload["popped_log_attack_mode"],
        timeout=timeout, score_alpha=payload.get("score_alpha"),
        symbolic_path_threshold=payload.get("symbolic_path_threshold"),
        constraint_build_timeout=options.get("constraint_build_timeout", True),
        constraint_build_timeout_seconds=options.get("constraint_build_timeout_seconds", 30),
        ternary_simplification=payload.get("ternary_simplification", False),
        ternary_threshold_scale=payload.get("ternary_threshold_scale", 0.75),
    )).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": SCHEMA_VERSION, "model": payload["model_name"],
               "case_id": payload["idx"], "policy": "staged", "controls": list(controls),
               "control_bounds": [list(b) for b in bounds], "region_budgets": budgets,
               "pwl_build_timeout_seconds": build_seconds, "total_budget_seconds": timeout,
               "total_wall_time": 0.0, "status": "running", "original_label": None,
               "attack_label": None, "winning_recipe": None, "attempts": []}

    def persist():
        summary["total_wall_time"] = time.monotonic() - started
        _write_json(parent / "recipe_summary.json", summary)

    persist()
    try:
        for index, selected in enumerate(sequence):
            now = time.monotonic()
            if now >= deadline:
                break
            allocated = (deadline - now) / (len(sequence) - index)
            attempt_deadline = now + allocated
            tag = "{:02d}-{}".format(index + 1, "-".join(selected))
            directory = parent / "recipes" / tag
            stage = copy.deepcopy(payload)
            # Existing path composition accepts an absolute input_name. The
            # path is generated here, never taken from CLI or a candidate.
            stage["save_exp"]["input_name"] = str(directory)
            stage["in_dict"] = {key: value for key, value in stage["in_dict"].items()
                                if key not in [parameter_name(c) for c in controls]}
            stage["in_dict"].update({parameter_name(c): 0.0 for c in selected})
            stage["con_dict"] = {parameter_name(c): 1 for c in selected}
            selected_bounds = [list(bound_map[c]) for c in selected]
            stage["global_real_config"].update(controls=list(selected), control_bounds=selected_bounds,
                                                pwl_max_regions=budgets[-1], pwl_region_budgets=budgets)
            attempt = {"index": index, "controls": list(selected), "control_bounds": selected_bounds,
                       "allocated_seconds": allocated, "started_after_seconds": now - started,
                       "wall_time": 0.0, "status": "running", "reason": "starting",
                       "stats_path": str(directory / "stats.json"), "pwl": {}}
            summary["attempts"].append(attempt)

            def update(progress):
                attempt.update(progress)
                attempt["wall_time"] = time.monotonic() - now
                persist()

            persist()
            log.info("[RECIPE-START] case=%s recipe=%s budget=%.3fs", payload["idx"], tag, allocated)
            update(_run_isolated(stage, options, attempt_deadline, build_seconds, update))
            # A killed process may have persisted a legitimate original label.
            stats_path = directory / "stats.json"
            if stats_path.is_file():
                try:
                    meta = json.loads(stats_path.read_text()).get("meta", {})
                    attempt.setdefault("original_label", meta.get("original_label"))
                except (OSError, ValueError) as exc:
                    if attempt["status"] not in ("timeout", "pwl_build_timeout"):
                        raise
                    # Hard termination can interrupt the recorder's JSON write.
                    attempt["stats_path"] = None
                    attempt["stats_unavailable_reason"] = "Interrupted stats write: {}".format(exc)
            else:
                attempt["stats_path"] = None
            original = attempt.get("original_label")
            if original is not None:
                if summary["original_label"] not in (None, original):
                    attempt.update(status="error", reason="Original label differs between recipes")
                summary["original_label"] = original
            if attempt["status"] == "success":
                summary["attack_label"] = attempt.get("attack_label")
                summary["winning_recipe"] = index
            persist()
            log.info("[RECIPE-END] case=%s recipe=%s status=%s wall=%.3fs", payload["idx"], tag,
                     attempt["status"], attempt["wall_time"])
            if attempt["status"] in ("success", "error"):
                break
        summary["status"] = _terminal_status(summary["attempts"], len(sequence), deadline)
    except BaseException as exc:
        summary["status"] = "error"
        summary["error_reason"] = "{}: {}".format(type(exc).__name__, exc)
        raise
    finally:
        persist()
        meta = {"model_name": payload["model_name"], "idx": payload["idx"],
                "semantic_attack_mode": "aces-recipes", "status": summary["status"],
                "original_label": summary["original_label"], "attack_label": summary["attack_label"],
                "is_finish": summary["status"] != "running", "is_timeout": summary["status"] == "timeout"}
        if summary["status"] == "error":
            meta.update(error_type="aces_recipe_failure", error_reason=summary.get("error_reason", "Recipe error"))
        _write_json(parent / "stats.json", {"meta": meta, "summary": {"total_wall_time": summary["total_wall_time"]}})
    return 0, SimpleNamespace(extra_meta=meta, save_dir=str(parent),
                              input_name=payload["save_exp"].get("input_name"),
                              attack_label=summary["attack_label"], is_timeout=meta["is_timeout"],
                              solve_all_ctr=summary["status"] == "exhausted")
