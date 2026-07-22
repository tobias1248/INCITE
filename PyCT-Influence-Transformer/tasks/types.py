from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypedDict

AttackMode = Literal["shap", "random", "random-assign", "queue"]
PixelSelector = Literal["pixel-shap", "patch-shap", "token-shap"]


class SaveExpConfig(TypedDict, total=False):
    input_name: str
    exp_name: str
    idx: int
    attack_mode: str
    save_smt: bool
    ton: int
    ton_next: Optional[int]
    only_first_forward: bool
    timeout: int
    constraint_build_timeout: bool
    constraint_build_timeout_seconds: int
    score_alpha: float
    symbolic_path_threshold: int
    ternary_simplification: bool
    ternary_threshold_scale: float
    fallback: bool
    fallback_type: str
    fallback_trigger: str
    fallback_source_attack_mode: str
    fallback_source_ton: Optional[int]
    fallback_source_ton_next: Optional[int]


class TonPlan(TypedDict):
    ton: int
    con_dict: Dict[str, int]
    save_exp: SaveExpConfig


class TaskPayload(TypedDict, total=False):
    model_name: str
    idx: int
    in_dict: Dict[str, Any]
    con_dict: Dict[str, int]
    solve_order_stack: Any
    input_for_shap: Any
    background_dataset_for_shap: Any
    shap_value_pre_calculated: bool
    popped_log_attack_mode: str
    ton_plans: List[TonPlan]
    save_exp: SaveExpConfig
    score_alpha: Optional[float]
    symbolic_path_threshold: Optional[int]
    ternary_simplification: bool
    ternary_threshold_scale: float
    ternary_fallback: bool
    smt_formula_sharing: Literal["raw", "let_cse"]


@dataclass(frozen=True)
class QueueMode:
    solve_order_stack: Any
    identifier: str


@dataclass
class TaskGenerationSpec:
    dataset_factory: Callable[[], Any]
    attack_pixel_fn: Callable[[int, int], List[Any]]
    queue_modes: Sequence[QueueMode]
    ton_values: Sequence[int]
    save_exp_builder: Callable[[int, int, QueueMode], SaveExpConfig]
    payload_builder: Callable[[Any, int, List[Any], int, QueueMode], Dict[str, Any]]
    skip_existing: bool = True
    save_dir_flag: Callable[[Dict[str, Any], QueueMode], bool] = field(
        default=lambda _save_exp, _mode: False
    )


@dataclass
class GenerationResult:
    inputs: List[TaskPayload]
    skipped: int = 0


__all__ = [
    "AttackMode",
    "PixelSelector",
    "SaveExpConfig",
    "TonPlan",
    "TaskPayload",
    "QueueMode",
    "TaskGenerationSpec",
    "GenerationResult",
]
