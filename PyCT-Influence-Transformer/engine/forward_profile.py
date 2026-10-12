"""Isolated, diagnostic-only profiling of one cached CIFAR10 image.

No solver, DE or probe is invoked. The no-path mode cannot generate attacks.
Each mode runs in a fresh spawned process and retains partial progress on timeout.
"""
from __future__ import annotations

import cProfile
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import pstats
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import numpy as np


MODES = ("keras_concrete", "nnmodel_concrete", "symbolic_path", "symbolic_no_path")
PROFILE_SHIFT_KINDS = ("aces-brightness", "aces-contrast")


def normalize_profile_options(options: dict) -> dict:
    """Resolve selections before creating artifacts or starting workers."""
    normalized = dict(options)
    normalized.setdefault("mode", "all")
    normalized.setdefault("global_shift_kind", "aces-contrast")
    if normalized["mode"] not in ("all",) + MODES:
        raise ValueError("Unknown profile mode")
    if normalized["global_shift_kind"] not in PROFILE_SHIFT_KINDS:
        raise ValueError("Profile control must be aces-brightness or aces-contrast")
    return normalized


class ProfileTimeout(BaseException):
    """Deadline event, kept distinct from a model failure."""


class _CountingPath:
    def __init__(self):
        self.current_constraint = SimpleNamespace(height=0)

    def add_branch(self, conbool):
        # ConcolicBool applies the same guard before invoking either path mode.
        self.current_constraint.height += 1


class ProfileBranchEngine:
    """Forward-only adapter using production path and priority scheduling code."""
    SHAP_SCORE_EPS = 1e-12

    def __init__(self, threshold: int, record_path: bool, global_real_config=None):
        if type(threshold) is not int or threshold < 1:
            raise ValueError("threshold must be a positive integer")
        from libct.constraint import Constraint
        from libct.path import PathToConstraint
        from libct.searcher import ConstraintScheduler, create_constraint_searcher
        Constraint.global_constraints = []
        self.symbolic_path_threshold = threshold
        self.symbolic_enabled = True
        self.symbolic_disabled_at_path_len = None
        self.path = PathToConstraint() if record_path else _CountingPath()
        self.constraints_to_solve = create_constraint_searcher("priority_queue")
        self.constraints_collection_type = "priority_queue"
        self.shap_score_alpha = 0.8
        self.comparator = None
        self.constraint_log_enabled = False
        self.global_real_config = global_real_config
        self.concolic_name_list = []
        self.concolic_flag_dict = {}
        self.idx = 0
        self._recorder = SimpleNamespace(queue_last=0, queue_max=0)
        self._scheduler = ConstraintScheduler(self)

    @property
    def branch_count(self):
        return self.path.current_constraint.height

    def _get_recorder(self):
        return self._recorder

    def push_constraint(self, constraint, position):
        self._scheduler.push_constraint(constraint, position)

    def _maybe_disable_symbolic(self, current_height):
        if self.symbolic_enabled and current_height >= self.symbolic_path_threshold:
            self.symbolic_enabled = False
            self.symbolic_disabled_at_path_len = current_height


def summarize_profile(profiler) -> Dict[str, Any]:
    """Disjoint self-time categories; cumulative times are never added together."""
    groups = {name: {"self_time": 0.0, "calls": 0} for name in
              ("path_management", "symbolic_arithmetic_and_objects", "model_python", "other")}
    functions = []
    for (filename, line, name), (primitive, total, own, cumulative, callers) in pstats.Stats(profiler).stats.items():
        normalized = filename.replace("\\", "/")
        if any(part in normalized for part in ("/libct/path.py", "/libct/constraint.py", "/libct/predicate.py", "/libct/searcher/")):
            category = "path_management"
        elif any(part in normalized for part in ("/libct/concolic/", "/libct/utils.py", "/libct/solver.py")):
            category = "symbolic_arithmetic_and_objects"
        elif "/dnnct/" in normalized:
            category = "model_python"
        else:
            category = "other"
        groups[category]["self_time"] += own
        groups[category]["calls"] += total
        functions.append({"file": filename, "line": line, "name": name, "calls": total,
                          "self_time": own, "cumulative_time": cumulative, "category": category})
    return {"groups": groups, "top_functions": sorted(functions, key=lambda f: f["self_time"], reverse=True)[:30]}


@contextmanager
def _layer_timers(model, layers, emit):
    originals = []
    try:
        for index, layer in enumerate(getattr(model, "layers", [])):
            forward = layer.forward
            existed = "forward" in vars(layer)
            old = vars(layer).get("forward")
            originals.append((layer, existed, old))

            def timed(*args, _forward=forward, _index=index, _type=type(layer).__name__, **kwargs):
                start, cpu = time.perf_counter(), time.process_time()
                row = {"index": _index, "type": _type, "status": "running"}
                layers.append(row)
                try:
                    emit({"event": "layer_start", **row})
                    value = _forward(*args, **kwargs)
                    row["status"] = "completed"
                    return value
                except BaseException as exc:
                    row["status"] = "timeout" if isinstance(exc, ProfileTimeout) else "error"
                    raise
                finally:
                    row.update(wall_time=time.perf_counter() - start, cpu_time=time.process_time() - cpu)
                    emit({"event": "layer_end", **row})

            layer.forward = timed
        yield
    finally:
        for layer, existed, old in reversed(originals):
            if existed:
                layer.forward = old
            else:
                del layer.forward


def profile_callable(func: Callable, model=None, timeout: float = 1800,
                     emit=None, profile_path: Optional[Path] = None) -> Dict[str, Any]:
    """Measure an already prepared forward. Durations include profiler overhead."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    emit = emit or (lambda event: None)
    layers, profiler = [], cProfile.Profile()
    previous = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError("Profiling cannot overlap an active real-time timer")
    expired = False

    def deadline(signum, frame):
        nonlocal expired
        expired = True
        raise ProfileTimeout()

    result = {"status": "completed", "label": None, "layers": layers}
    with _layer_timers(model, layers, emit):
        signal.signal(signal.SIGALRM, deadline)
        start, cpu = time.perf_counter(), time.process_time()
        profiler.enable()
        signal.setitimer(signal.ITIMER_REAL, timeout, 0.5)
        try:
            result["label"] = int(func())
        except ProfileTimeout:
            result["status"] = "timeout"
        except Exception as exc:
            result.update(status="error", error_type=type(exc).__name__, error=str(exc))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            profiler.disable()
            result.update(wall_time=time.perf_counter() - start, cpu_time=time.process_time() - cpu)
            signal.signal(signal.SIGALRM, previous)
    # Some legacy arithmetic catches BaseException. Never call an expired run
    # completed just because it swallowed the deadline exception and returned.
    if expired or result["wall_time"] >= timeout:
        result.update(status="timeout", label=None)
    result["profile_groups"] = summarize_profile(profiler)
    if profile_path is not None:
        profiler.dump_stats(str(profile_path))
    return result


def _timed_phase(name, func, emit, phases):
    emit({"event": "phase_start", "phase": name})
    start, cpu = time.perf_counter(), time.process_time()
    value = func()
    row = {"phase": name, "wall_time": time.perf_counter() - start, "cpu_time": time.process_time() - cpu}
    phases.append(row)
    emit({"event": "phase_end", **row})
    return value


def _load_case(cache: Path, index: int):
    with cache.open("rb") as handle:
        batch = pickle.load(handle, encoding="bytes")
    if not 0 <= index < len(batch[b"labels"]):
        raise ValueError("case index outside cached dataset")
    image = np.asarray(batch[b"data"][index]).reshape(3, 32, 32).transpose(1, 2, 0).astype(np.float32) / 255
    return image, int(batch[b"labels"][index])


def _profile_worker(connection, mode, options, directory):
    """All heavy model imports and mutable instrumentation live in this process."""
    emit = connection.send
    phases = []
    try:
        emit({"event": "phase_start", "phase": "imports"})
        start, cpu = time.perf_counter(), time.process_time()
        from engine import predictor_runtime as predictor
        from libct.global_real import (GLOBAL_X_INPUT_NAME, build_aces_like_global_real_config,
                                       build_concolic_global_real_kwargs)
        from libct.solver import Solver
        from libct.constraint import Constraint
        phases.append({"phase": "imports", "wall_time": time.perf_counter() - start,
                       "cpu_time": time.process_time() - cpu})
        emit({"event": "phase_end", **phases[-1]})
        # Safety validation would call a solver on arithmetic: explicitly disable it.
        Solver.safety = 0
        image, truth = _timed_phase("dataset_load", lambda: _load_case(Path(options["dataset_cache"]), options["case_index"]), emit, phases)
        model_path = options["model_path"]
        _timed_phase("keras_model_load", lambda: predictor.init_reference_model(model_path), emit, phases)
        if mode != "keras_concrete":
            _timed_phase("nnmodel_conversion", lambda: predictor.init_model(model_path, role="search"), emit, phases)
        config = _timed_phase("pwl_build", lambda: build_aces_like_global_real_config(image, {
            "global_shift_kind": options["global_shift_kind"], "bounds_mode": "clip",
            "effective_min": -0.2, "effective_max": 0.2,
            "requested_min": -0.2, "requested_max": 0.2,
            "pwl_max_segments": 32, "pwl_error_tolerance": 1.0 / 255,
            "probe_enabled": False,
        }), emit, phases)
        inputs = {"v_" + "_".join(map(str, coordinates)): float(image[coordinates]) for coordinates in np.ndindex(image.shape)}
        engine = None
        if mode.startswith("symbolic"):
            engine = ProfileBranchEngine(options["path_threshold"], mode == "symbolic_path", config)
            inputs[GLOBAL_X_INPUT_NAME] = 0.0
            inputs = _timed_phase("symbolic_input_build", lambda: build_concolic_global_real_kwargs(engine, inputs), emit, phases)
        # Warm only the native Keras baseline, outside measured forward.
        if mode == "keras_concrete":
            _timed_phase("keras_warmup", lambda: predictor.predict_reference(**inputs), emit, phases)
            forward = lambda: predictor.predict_reference(**inputs)
            model = None
        else:
            forward = lambda: predictor.predict_search(**inputs)
            model = predictor.searchModel
        emit({"event": "phase_start", "phase": "forward"})
        result = profile_callable(forward, model=model, timeout=options["timeout"], emit=emit,
                                  profile_path=Path(directory) / "forward.prof")
        result.update(mode=mode, phases=phases, ground_truth=truth,
                      global_shift_kind=config["global_shift_kind"],
                      source_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                      pwl_segment_count=config["pwl_segment_count"],
                      pwl_max_abs_error=config["pwl_max_abs_error"],
                      branch_count=engine.branch_count if engine else 0,
                      symbolic_disabled_at_path_len=engine.symbolic_disabled_at_path_len if engine else None,
                      constraint_nodes=len(Constraint.global_constraints),
                      queued_candidates=len(engine.constraints_to_solve) if engine else 0,
                      includes_profiler_overhead=True)
        emit({"event": "result", "result": result})
    except Exception as exc:
        emit({"event": "result", "result": {"mode": mode, "status": "error", "phases": phases,
              "error_type": type(exc).__name__, "error": str(exc)}})
    finally:
        connection.close()


def run_mode(mode: str, options: dict, directory: Path, worker=None) -> dict:
    """Persist progress independently of child shutdown, with a hard watchdog."""
    if mode not in MODES:
        raise ValueError("Unknown profile mode")
    options = normalize_profile_options(options)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=worker or _profile_worker, args=(send, mode, options, str(directory)))
    process.start()
    send.close()
    started = time.monotonic()
    deadline = started + options.get("setup_timeout", 300)
    phase = "setup"
    layers, phases, result = {}, [], None
    try:
        with (directory / "events.jsonl").open("w") as handle:
            while result is None:
                if receive.poll(0.1):
                    try:
                        event = receive.recv()
                    except EOFError:
                        break
                    handle.write(json.dumps(event) + "\n")
                    handle.flush()
                    if event["event"] == "phase_start":
                        phase = event["phase"]
                        if phase == "forward":
                            deadline = time.monotonic() + options["timeout"] + 5
                    elif event["event"] == "phase_end":
                        phases.append(event)
                    elif event["event"] in ("layer_start", "layer_end"):
                        layers[event["index"]] = event
                    elif event["event"] == "result":
                        result = event["result"]
                if time.monotonic() >= deadline:
                    break
                if not process.is_alive() and not receive.poll():
                    break
        if result is None:
            timed_out = time.monotonic() >= deadline
            result = {"mode": mode, "status": "timeout" if timed_out else "error", "phase": phase,
                      "hard_timeout": timed_out, "error": "Worker did not return a final profile",
                      "label": None, "phases": phases, "layers": list(layers.values()),
                      "wall_time": None, "cpu_time": None, "profile_groups": None}
            for layer in result["layers"]:
                if layer["status"] == "running":
                    layer["status"] = "timeout" if timed_out else "interrupted"
        result.setdefault("global_shift_kind", options["global_shift_kind"])
        result["process_wall_time"] = time.monotonic() - started
    finally:
        receive.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def run_profile_case(options: dict, output_dir: Path) -> dict:
    """Run the selected mode(s). Refuse overwriting previous artifacts."""
    options = normalize_profile_options(options)
    modes = MODES if options["mode"] == "all" else (options["mode"],)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    model = Path(options["model_path"])
    manifest = {"schema_version": "forward-profile-v1", "options": options,
                "fixed_settings": {"shift_kind": options["global_shift_kind"], "X": 0, "range": [-0.2, 0.2],
                                   "bounds_mode": "clip", "pwl_max_segments": 32,
                                   "pwl_error_tolerance": 1.0 / 255, "score_alpha": 0.8,
                                   "solver": "not invoked", "probe": False, "de": False,
                                   "ternary_simplification": False},
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "environment": {key: os.environ.get(key) for key in
                    ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
                "notes": ["Diagnostic only: no solver, probe, DE, or attack success measurement.",
                          "Forward times include cProfile and layer instrumentation overhead.",
                          "Profile group self-times are disjoint; layer times overlap those groups.",
                          "Timeout durations are partial; no speedup is inferred unless both symbolic modes finish."]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for mode in modes:
        print("Starting {}".format(mode), flush=True)
        result = run_mode(mode, options, output_dir / mode)
        results.append(result)
        print("{}: {} wall={} branches={}".format(mode, result["status"], result.get("wall_time"), result.get("branch_count")), flush=True)
        summary = {"manifest": manifest, "results": results}
        full, no_path = [next((r for r in results if r["mode"] == m), None) for m in MODES[2:]]
        comparable = bool(full and no_path and full["status"] == no_path["status"] == "completed"
                          and full["label"] == no_path["label"] and full["branch_count"] == no_path["branch_count"]
                          and full["symbolic_disabled_at_path_len"] == no_path["symbolic_disabled_at_path_len"])
        summary["symbolic_comparable"] = comparable
        summary["symbolic_path_speedup"] = full["wall_time"] / no_path["wall_time"] if comparable and no_path["wall_time"] > 0 else None
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
