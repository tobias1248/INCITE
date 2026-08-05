from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Tuple

from libct.global_real import build_concolic_global_real_kwargs
from libct.utils import ConcolicObject, unwrap


log = logging.getLogger("ct.explore")


class ConcolicArgumentBuilder:
    """Compatibility builder for concolic target call arguments."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def build(
        self,
        func: Any,
        prim_args: Dict[str, Any],
        concolic_dict: Dict[str, Any],
    ) -> Tuple[list, dict]:
        ccc_args = []
        ccc_kwargs = {}
        self._engine.concolic_name_list = []

        for param in inspect.signature(func).parameters.values():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,):
                # do not support *args currently
                prim_args.pop(param.name, None)
                continue

            if param.kind in (inspect.Parameter.VAR_KEYWORD,):
                # only support 1 **kwargs and no other arguments.
                assert len(inspect.signature(func).parameters.values()) == 1
                if getattr(self._engine, "global_real_config", None) is not None:
                    ccc_kwargs = build_concolic_global_real_kwargs(
                        self._engine,
                        prim_args,
                    )
                else:
                    for name, value in prim_args.items():
                        ccc_kwargs[name] = self._wrap_kwarg(name, value, concolic_dict)
                break

            value = self._resolve_argument_value(param, prim_args)
            self._engine.concolic_flag_dict[param.name + "_VAR"] = 0
            if type(value) in (bool, float, int, str) and concolic_dict.get(param.name, 1):
                value = self._wrap_value(param.name, value)

            if param.kind is inspect.Parameter.KEYWORD_ONLY:
                ccc_kwargs[param.name] = value
            else:
                ccc_args.append(value)

        if getattr(self._engine, "global_real_config", None) is not None:
            if not self._engine.var_to_types:
                self._engine.var_to_types["__pyct_global_x_VAR"] = "Real"
        else:
            self._record_var_types(prim_args)
        log.info(
            "[WRAP] idx=%s concolic=%s primitive=%s queue_type=%s",
            self._engine.idx,
            len(self._engine.concolic_name_list),
            len(prim_args),
            self._engine.constraints_collection_type,
        )

        return ccc_args, ccc_kwargs

    def _wrap_kwarg(self, name: str, value: Any, concolic_dict: Dict[str, Any]) -> Any:
        ccc_obj_name = name + "_VAR"  # '_VAR' is used to avoid name collision
        self._engine.concolic_flag_dict[ccc_obj_name] = 0
        if type(value) in (bool, float, int, str) and concolic_dict.get(name, 0):
            value = ConcolicObject(value, ccc_obj_name, self._engine)
            self._engine.concolic_name_list.append(ccc_obj_name)
            self._engine.concolic_flag_dict[ccc_obj_name] = 1
        return value

    def _resolve_argument_value(self, param: inspect.Parameter, prim_args: Dict[str, Any]) -> Any:
        if param.name in prim_args:
            return prim_args[param.name]

        has_value = False
        if (annotation := param.annotation) is not inspect._empty:
            try:
                value = annotation()
                # may raise TypeError: Cannot instantiate ...
                has_value = True
            except Exception:
                pass
        if not has_value:
            if (default := param.default) is not inspect._empty:
                # default values may also be wrapped
                value = unwrap(default)
            else:
                value = ""
        prim_args[param.name] = value if type(value) in (bool, float, int, str) else self._engine.LazyLoading
        return value

    def _wrap_value(self, name: str, value: Any) -> Any:
        ccc_obj_name = name + "_VAR"  # '_VAR' is used to avoid name collision
        value = ConcolicObject(value, ccc_obj_name, self._engine)
        self._engine.concolic_name_list.append(ccc_obj_name)
        self._engine.concolic_flag_dict[ccc_obj_name] = 1
        return value

    def _record_var_types(self, prim_args: Dict[str, Any]) -> None:
        if self._engine.var_to_types:
            return
        for key, value in prim_args.items():
            var_name = key + "_VAR"  # '_VAR' is used to avoid name collision
            if type(value) is bool:
                self._engine.var_to_types[var_name] = "Bool"
            elif type(value) is float:
                self._engine.var_to_types[var_name] = "Real"
            elif type(value) is int:
                self._engine.var_to_types[var_name] = "Int"
            elif type(value) is str:
                self._engine.var_to_types[var_name] = "String"
