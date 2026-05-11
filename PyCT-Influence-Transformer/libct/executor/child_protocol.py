from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Any, Dict, Optional

from libct.constraint import Constraint


log = logging.getLogger("ct.explore")


class ConstraintTransferError(RuntimeError):
    """Raised when child process constraints cannot be transferred safely."""


class ChildProtocol:
    """Child-process envelope and transfer protocol adapter for ExplorationEngine."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def _recorder(self) -> Any:
        get_recorder = getattr(self._engine, "_get_recorder", None)
        if callable(get_recorder):
            return get_recorder()
        return None

    def mark_constraint_transfer_failure(self, reason: str) -> None:
        self.mark_runtime_error(
            "constraint_transfer_failure",
            reason,
            phase="transfer",
        )

    def mark_runtime_error(
        self,
        error_type: str,
        reason: str,
        *,
        phase: Optional[str] = None,
        child_pid: Optional[int] = None,
        event_type: Optional[str] = None,
    ) -> None:
        recorder = self._recorder()
        if recorder is None:
            return
        mark_error = getattr(recorder, "mark_error", None)
        if callable(mark_error):
            mark_error(
                error_type,
                reason,
                phase=phase,
                child_pid=child_pid,
                event_type=event_type,
            )
            return
        extra_meta = getattr(recorder, "extra_meta", None)
        if extra_meta is None:
            extra_meta = {}
            recorder.extra_meta = extra_meta
        extra_meta["status"] = "error"
        extra_meta["error_type"] = error_type
        extra_meta["error_reason"] = reason
        if phase is not None:
            extra_meta["error_phase"] = phase
        if child_pid is not None:
            extra_meta["child_pid"] = child_pid
        if event_type is not None:
            extra_meta["child_event_type"] = event_type

    def record_child_event(
        self,
        event_type: str,
        message: str,
        *,
        phase: str,
        child_pid: Optional[int],
    ) -> None:
        recorder = self._recorder()
        if recorder is not None:
            mark_event = getattr(recorder, "mark_child_event", None)
            if callable(mark_event):
                mark_event(
                    event_type,
                    message,
                    phase=phase,
                    child_pid=child_pid,
                )
            else:
                extra_meta = getattr(recorder, "extra_meta", None)
                if extra_meta is None:
                    extra_meta = {}
                    recorder.extra_meta = extra_meta
                extra_meta["child_event_type"] = event_type
                extra_meta["child_event_message"] = message
                extra_meta["child_event_phase"] = phase
                if child_pid is not None:
                    extra_meta["child_pid"] = child_pid
        log.warning(
            "[CHILD-EVENT] idx=%s pid=%s phase=%s event_type=%s input_name=%s save_dir=%s message=%s",
            self._engine.idx,
            child_pid,
            phase,
            event_type,
            self._engine.input_name,
            self._engine.save_dir,
            message,
        )

    def write_diagnostic_file(self, filename: str, contents: Optional[str]) -> None:
        save_dir = getattr(self._engine, "save_dir", None)
        if not save_dir or not contents:
            return
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as handle:
            handle.write(contents)

    def build_child_shared_state(self, updated_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "updated_args": updated_args,
            "var_to_types": self._engine.var_to_types,
            "concolic_name_list": self._engine.concolic_name_list,
            "concolic_flag_dict": self._engine.concolic_flag_dict,
        }

    def build_child_ok_envelope(
        self,
        *,
        pid: int,
        updated_args: Dict[str, Any],
        result: Any,
        constraint_payload: Any,
    ) -> Dict[str, Any]:
        envelope = {
            "kind": "ok",
            "pid": pid,
            "phase": "execute",
            "result": result,
            "constraint_payload": constraint_payload,
            "message": "child execution completed successfully",
        }
        envelope.update(self.build_child_shared_state(updated_args))
        return envelope

    def build_child_event_envelope(
        self,
        *,
        pid: int,
        updated_args: Optional[Dict[str, Any]],
        result: Any,
        event_type: str,
        message: str,
        error_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        envelope = {
            "kind": "child_event",
            "pid": pid,
            "phase": "execute",
            "result": result,
            "event_type": event_type,
            "message": message,
        }
        if error_class is not None:
            envelope["error_class"] = error_class
        envelope.update(self.build_child_shared_state(updated_args))
        return envelope

    def build_child_error_envelope(
        self,
        *,
        pid: int,
        updated_args: Optional[Dict[str, Any]],
        error_type: str,
        phase: str,
        message: str,
        error_class: Optional[str] = None,
        traceback_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        envelope = {
            "kind": "child_error",
            "pid": pid,
            "phase": phase,
            "result": self._engine.Exception,
            "error_type": error_type,
            "message": message,
        }
        if error_class is not None:
            envelope["error_class"] = error_class
        if traceback_text is not None:
            envelope["traceback"] = traceback_text
        envelope.update(self.build_child_shared_state(updated_args))
        return envelope

    def validate_child_envelope(self, envelope: Any) -> Dict[str, Any]:
        if not isinstance(envelope, dict):
            raise ValueError(f"child returned non-dict envelope: {type(envelope).__name__}")
        kind = envelope.get("kind")
        if kind not in {"ok", "child_event", "child_error"}:
            raise ValueError(f"child returned unknown envelope kind: {kind!r}")
        if "pid" not in envelope:
            raise ValueError("child envelope missing pid")
        if "phase" not in envelope:
            raise ValueError("child envelope missing phase")
        if "result" not in envelope:
            raise ValueError("child envelope missing result")
        if kind == "ok" and "constraint_payload" not in envelope:
            raise ValueError("ok envelope missing constraint_payload")
        if kind == "child_event" and "event_type" not in envelope:
            raise ValueError("child_event envelope missing event_type")
        if kind == "child_error" and "error_type" not in envelope:
            raise ValueError("child_error envelope missing error_type")
        return envelope

    def apply_child_shared_state(self, all_args: Dict[str, Any], envelope: Dict[str, Any]) -> None:
        updated_args = envelope.get("updated_args")
        if isinstance(updated_args, dict):
            all_args.clear()
            all_args.update(updated_args)
        self._engine.var_to_types = envelope.get("var_to_types", self._engine.var_to_types)
        self._engine.concolic_name_list = envelope.get(
            "concolic_name_list",
            self._engine.concolic_name_list,
        )
        self._engine.concolic_flag_dict = envelope.get(
            "concolic_flag_dict",
            self._engine.concolic_flag_dict,
        )

    def raise_transport_failure(
        self,
        reason: str,
        *,
        phase: str,
        child_pid: Optional[int] = None,
        details: Optional[str] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        log.error(
            "[PARENT-RECV-ERROR] idx=%s pid=%s phase=%s error_type=constraint_transfer_failure input_name=%s save_dir=%s message=%s",
            self._engine.idx,
            child_pid,
            phase,
            self._engine.input_name,
            self._engine.save_dir,
            reason,
        )
        self.mark_runtime_error(
            "constraint_transfer_failure",
            reason,
            phase=phase,
            child_pid=child_pid,
        )
        if details:
            self.write_diagnostic_file("transfer_error_traceback.txt", details)
        if exc is None:
            raise ConstraintTransferError(reason)
        raise ConstraintTransferError(reason) from exc

    def receive_child_envelope(
        self,
        conn: Any,
        process: Any,
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            if conn.poll(0.05):
                break
            if not process.is_alive():
                reason = f"child exited before sending a valid envelope (exitcode={process.exitcode})"
                self.raise_transport_failure(
                    reason,
                    phase="transport",
                    child_pid=process.pid,
                    details=reason,
                )
            if time.monotonic() >= deadline:
                reason = f"timed out waiting for child envelope after {timeout_seconds}s"
                self.raise_transport_failure(
                    reason,
                    phase="transport",
                    child_pid=process.pid,
                    details=reason,
                )

        try:
            envelope = conn.recv()
        except (EOFError, OSError, ValueError) as exc:
            reason = f"failed to receive child envelope: {exc.__class__.__name__}: {exc}"
            self.raise_transport_failure(
                reason,
                phase="transport",
                child_pid=process.pid,
                details=traceback.format_exc(),
                exc=exc,
            )

        try:
            return self.validate_child_envelope(envelope)
        except ValueError as exc:
            reason = str(exc)
            self.raise_transport_failure(
                reason,
                phase="protocol",
                child_pid=envelope.get("pid") if isinstance(envelope, dict) else process.pid,
                details=repr(envelope),
                exc=exc,
            )

    def handle_child_envelope(self, all_args: Dict[str, Any], envelope: Dict[str, Any]) -> Any:
        self.apply_child_shared_state(all_args, envelope)
        kind = envelope["kind"]
        child_pid = envelope.get("pid")
        phase = str(envelope.get("phase", "execute"))
        message = str(envelope.get("message", ""))

        if kind == "ok":
            self.apply_constraint_transfer_payload(envelope["constraint_payload"])
            return envelope["result"]

        if kind == "child_event":
            self.record_child_event(
                str(envelope["event_type"]),
                message,
                phase=phase,
                child_pid=child_pid,
            )
            return envelope["result"]

        error_type = str(envelope["error_type"])
        log.error(
            "[CHILD-ERROR] idx=%s pid=%s phase=%s error_type=%s input_name=%s save_dir=%s message=%s",
            self._engine.idx,
            child_pid,
            phase,
            error_type,
            self._engine.input_name,
            self._engine.save_dir,
            message,
        )
        traceback_text = envelope.get("traceback")
        if traceback_text:
            filename = (
                "transfer_error_traceback.txt"
                if error_type == "constraint_transfer_failure"
                else "child_error_traceback.txt"
            )
            self.write_diagnostic_file(filename, traceback_text)
        self.mark_runtime_error(
            error_type,
            message,
            phase=phase,
            child_pid=child_pid,
        )
        if error_type == "constraint_transfer_failure":
            raise ConstraintTransferError(message)
        raise RuntimeError(message)

    def apply_constraint_transfer_payload(self, payload: Any) -> None:
        if payload is self._engine.Unpicklable:
            reason = "child process returned an unpicklable constraint/path payload"
            log.error("Constraint transfer failed: %s", reason)
            self.mark_constraint_transfer_failure(reason)
            raise ConstraintTransferError(reason)

        symbolic_disabled_at_path_len = None
        if isinstance(payload, tuple) and len(payload) == 4:
            (
                Constraint.global_constraints,
                self._engine.constraints_to_solve,
                self._engine.path,
                symbolic_disabled_at_path_len,
            ) = payload
        else:
            Constraint.global_constraints, self._engine.constraints_to_solve, self._engine.path = payload
        if symbolic_disabled_at_path_len is not None:
            self._engine.symbolic_disabled_at_path_len = symbolic_disabled_at_path_len
