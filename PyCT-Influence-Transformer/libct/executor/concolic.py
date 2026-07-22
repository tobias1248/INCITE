from __future__ import annotations

import builtins
import func_timeout
import logging
import multiprocessing
import os
import sys
import traceback
from typing import Any, Dict, Optional

from libct.constraint import Constraint
from libct.utils import unwrap


log = logging.getLogger("ct.explore")


def prepare_child_environment() -> None:
    #################################################################
    # Since the source code in https://github.com/python/cpython/blob/e822e37946f27c09953bb5733acf3b07c2db690f/Modules/socketmodule.c#L6485
    # only accepts "unwrapped" input arguments, we simply do it here.
    #################################################################
    import socket

    _socket_getaddrinfo = socket.getaddrinfo

    def socket_getaddrinfo(*args, **kwargs):
        return _socket_getaddrinfo(*map(unwrap, args), **{k: unwrap(v) for (k, v) in kwargs.items()})

    socket.getaddrinfo = socket_getaddrinfo
    #####################################################################
    # The builtin len(...) function will automatically unwrap our result,
    # so we want to avoid this by doing the following line.
    #####################################################################
    builtins.len = lambda x: x.__len__()


class ConcolicExecutionRunner:
    """Subprocess runner for one concolic target execution."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def run(self, all_args: Dict[str, Any], concolic_dict: Dict[str, Any]) -> Any:
        envelope = self.run_deferred(all_args, concolic_dict)
        return self._engine._handle_child_envelope(all_args, envelope)

    def run_deferred(self, all_args: Dict[str, Any], concolic_dict: Dict[str, Any]) -> Dict[str, Any]:
        r2, s2 = multiprocessing.Pipe()

        process = multiprocessing.Process(
            target=self._child_process,
            args=(s2, all_args, concolic_dict),
        )
        process.start()

        try:
            envelope = self._engine._receive_child_envelope(
                r2,
                process,
                self._engine.single_timeout + 5,
            )
        finally:
            r2.close()
            s2.close()
            if process.is_alive():
                process.kill()
            process.join(timeout=0.1)
        return envelope

    def _child_process(self, send_conn: Any, all_args: Dict[str, Any], concolic_dict: Dict[str, Any]) -> None:
        # Very important to prevent the later primitive mode from using concolic objects imported here.
        sys.dont_write_bytecode = True
        child_pid = os.getpid()
        updated_args: Optional[Dict[str, Any]] = None
        envelope = None
        try:
            prepare_child_environment()
            self._engine.path.__init__()
            self._engine._reset_symbolic_guard()
            if self._engine.can_use_concolic_wrapper:
                import libct.wrapper  # noqa: F401
            else:
                import libct  # noqa: F401

            execute = self._engine._get_execute()
            ccc_args, ccc_kwargs = self._engine._get_concolic_arguments(
                execute,
                all_args,
                concolic_dict,
            )
            updated_args = dict(all_args)
            result = self._engine.Exception
            try:
                result = unwrap(
                    func_timeout.func_timeout(
                        self._engine.single_timeout,
                        execute,
                        args=ccc_args,
                        kwargs=ccc_kwargs,
                    )
                )
                log.info(f"Return: {result}")
                envelope = self._engine._build_child_ok_envelope(
                    pid=child_pid,
                    updated_args=updated_args,
                    result=result,
                    constraint_payload=(
                        Constraint.global_constraints,
                        self._engine.constraints_to_solve,
                        self._engine.path,
                        self._engine.symbolic_disabled_at_path_len,
                    ),
                )
            except func_timeout.FunctionTimedOut:
                result = self._engine.Timeout
                if getattr(self._engine, "branch_trace_enabled", False):
                    message = (
                        "Timeout (soft) for trace-only execution: "
                        f"idx={self._engine.idx}, variables={len(all_args)}"
                    )
                else:
                    message = (
                        f"Timeout (soft) for: {all_args} >> ./pyct.py -r '{self._engine.root}' "
                        f"'{self._engine.modpath}' -s {self._engine.funcname} {{}} --lib "
                        f"'{self._engine.lib}' --include_exception"
                    )
                log.error(message)
                if self._engine.statsdir:
                    with open(self._engine.statsdir + '/exception.txt', 'a') as f:
                        f.write(message + "\n")
                envelope = self._engine._build_child_event_envelope(
                    pid=child_pid,
                    updated_args=updated_args,
                    result=result,
                    event_type="soft_timeout",
                    message=message,
                    branch_trace=(
                        tuple(getattr(self._engine.path, "branch_trace", ()))
                        if getattr(self._engine, "branch_trace_enabled", False)
                        else None
                    ),
                )
            except Exception as e:
                if getattr(self._engine, "branch_trace_enabled", False):
                    message = (
                        "Exception during trace-only execution: "
                        f"idx={self._engine.idx}, variables={len(all_args)}"
                    )
                else:
                    message = (
                        f"Exception for: {all_args} >> ./pyct '{self._engine.root}' "
                        f"'{self._engine.modpath}' -s {self._engine.funcname} {{}} -m 20 --lib "
                        f"'{self._engine.lib}' --include_exception"
                    )
                log.exception(message)
                if self._engine.statsdir:
                    with open(self._engine.statsdir + '/exception.txt', 'a') as f:
                        f.write(message + "\n")
                        f.write(f"{e}\n")
                envelope = self._engine._build_child_event_envelope(
                    pid=child_pid,
                    updated_args=updated_args,
                    result=self._engine.Exception,
                    event_type="target_exception",
                    message=str(e) or message,
                    error_class=e.__class__.__name__,
                    branch_trace=(
                        tuple(getattr(self._engine.path, "branch_trace", ()))
                        if getattr(self._engine, "branch_trace_enabled", False)
                        else None
                    ),
                )
        except Exception as exc:
            traceback_text = traceback.format_exc()
            envelope = self._engine._build_child_error_envelope(
                pid=child_pid,
                updated_args=updated_args,
                error_type="child_unexpected_error",
                phase="execute",
                message=str(exc) or exc.__class__.__name__,
                error_class=exc.__class__.__name__,
                traceback_text=traceback_text,
            )

        try:
            send_conn.send(envelope)
        except Exception as exc:
            fallback = self._engine._build_child_error_envelope(
                pid=child_pid,
                updated_args=updated_args,
                error_type="constraint_transfer_failure",
                phase="transfer",
                message=(
                    "failed to send child envelope to parent: "
                    f"{exc.__class__.__name__}: {exc}"
                ),
                error_class=exc.__class__.__name__,
                traceback_text=traceback.format_exc(),
            )
            try:
                send_conn.send(fallback)
            except Exception:
                pass
