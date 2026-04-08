from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from tasks.types import SaveExpConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_repo_root() -> Path:
    return REPO_ROOT


def get_repo_output_dir(dirname: str) -> Path:
    path = REPO_ROOT / dirname
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_repo_output_subdir(dirname: str, *parts: str) -> Path:
    path = get_repo_output_dir(dirname).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_save_dir_from_save_exp(
    save_exp: SaveExpConfig,
    model_name: str,
    attack_mode: str,
    *,
    only_first_forward: bool = False,
    timeout: Optional[int] = None,
    constraint_build_timeout: Optional[bool] = None,
    constraint_build_timeout_seconds: Optional[int] = None,
    score_alpha: Optional[float] = None,
    symbolic_path_threshold: Optional[int] = None,
) -> str:
    def _format_alpha(value: Optional[Any]) -> str:
        if value is None:
            return "aNA"
        try:
            val = float(value)
        except (TypeError, ValueError):
            return f"a{value}"
        scaled = val * 10.0
        if abs(scaled - round(scaled)) < 1e-6:
            return f"a{int(round(scaled)):02d}"
        cleaned = f"{val:g}".replace(".", "p").replace("-", "m")
        return f"a{cleaned}"

    def _resolve_value(key: str, explicit: Optional[Any], env_key: str) -> Optional[Any]:
        if explicit is not None:
            return explicit
        if isinstance(save_exp, dict) and key in save_exp:
            return save_exp.get(key)
        return os.environ.get(env_key)

    def _resolve_bool(key: str, explicit: Optional[bool], env_key: str) -> Optional[bool]:
        if explicit is not None:
            return bool(explicit)
        if isinstance(save_exp, dict) and key in save_exp:
            return bool(save_exp.get(key))
        raw = os.environ.get(env_key)
        if raw is None:
            return None
        return raw.lower() not in {"0", "false", "no", "off"}

    def _format_component(value: Optional[Any]) -> str:
        if value is None:
            return "na"
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    def _format_build_timeout_component(enabled: Optional[bool], seconds: Optional[Any]) -> str:
        if enabled is False:
            return "0"
        if seconds is None:
            return "30"
        try:
            return str(int(seconds))
        except (TypeError, ValueError):
            return _format_component(seconds)

    base_model = f"{model_name}_only_first_forward" if only_first_forward else model_name
    timeout_val = _resolve_value("timeout", timeout, "PYCT_TIMEOUT")
    build_timeout_enabled = _resolve_bool(
        "constraint_build_timeout",
        constraint_build_timeout,
        "PYCT_CONSTRAINT_BUILD_TIMEOUT_ENABLED",
    )
    build_timeout_seconds_val = _resolve_value(
        "constraint_build_timeout_seconds",
        constraint_build_timeout_seconds,
        "PYCT_CONSTRAINT_BUILD_TIMEOUT_SECONDS",
    )
    alpha_val = _resolve_value("score_alpha", score_alpha, "PYCT_SCORE_ALPHA")
    threshold_val = _resolve_value(
        "symbolic_path_threshold",
        symbolic_path_threshold,
        "PYCT_SYMBOLIC_PATH_THRESHOLD",
    )
    alpha_component = _format_alpha(alpha_val)
    build_timeout_component = _format_build_timeout_component(
        build_timeout_enabled,
        build_timeout_seconds_val,
    )
    base_dir = "{}_{}_{}_{}_{}_{}".format(
        base_model,
        attack_mode,
        _format_component(timeout_val),
        build_timeout_component,
        alpha_component,
        _format_component(threshold_val),
    )
    idx = save_exp.get("idx")
    if idx is None:
        input_name = save_exp.get("input_name", "")
        if input_name.startswith("case_"):
            try:
                idx = int(input_name.split("_")[-1])
            except ValueError:
                idx = "unknown"
        else:
            idx = "unknown"
    case_name = save_exp.get("input_name", f"case_{idx}")
    return str(get_repo_output_dir("exp") / base_dir / case_name)


__all__ = [
    "get_repo_output_dir",
    "get_repo_output_subdir",
    "get_repo_root",
    "get_save_dir_from_save_exp",
]
