from __future__ import annotations

from libct.concolic import Concolic
from libct.executor import ConcolicArgumentBuilder


class _Engine:
    class LazyLoading:
        pass

    def __init__(self) -> None:
        self.idx = 5
        self.constraints_collection_type = "queue"
        self.concolic_name_list = []
        self.concolic_flag_dict = {}
        self.var_to_types = {}


def test_argument_builder_wraps_selected_positional_and_keyword_arguments() -> None:
    def target(x, y, *, z):
        return x, y, z

    engine = _Engine()
    prim_args = {"x": 1, "y": 2.5, "z": "off"}

    args, kwargs = ConcolicArgumentBuilder(engine).build(
        target,
        prim_args,
        {"x": 1, "y": 0, "z": 1},
    )

    assert isinstance(args[0], Concolic)
    assert args[1] == 2.5
    assert isinstance(kwargs["z"], Concolic)
    assert engine.concolic_name_list == ["x_VAR", "z_VAR"]
    assert engine.concolic_flag_dict == {"x_VAR": 1, "y_VAR": 0, "z_VAR": 1}
    assert engine.var_to_types == {"x_VAR": "Int", "y_VAR": "Real", "z_VAR": "String"}


def test_argument_builder_supports_kwargs_only_signature() -> None:
    def target(**kwargs):
        return kwargs

    engine = _Engine()
    prim_args = {"a": True, "b": 3}

    args, kwargs = ConcolicArgumentBuilder(engine).build(
        target,
        prim_args,
        {"a": 1, "b": 0},
    )

    assert args == []
    assert isinstance(kwargs["a"], Concolic)
    assert kwargs["b"] == 3
    assert engine.concolic_name_list == ["a_VAR"]
    assert engine.concolic_flag_dict == {"a_VAR": 1, "b_VAR": 0}
    assert engine.var_to_types == {"a_VAR": "Bool", "b_VAR": "Int"}


def test_argument_builder_fills_defaults_and_lazy_loading_marker() -> None:
    class NonPrimitive:
        pass

    def target(x: int, y=1.25, z=NonPrimitive()):
        return x, y, z

    engine = _Engine()
    prim_args = {}

    args, kwargs = ConcolicArgumentBuilder(engine).build(target, prim_args, {})

    assert len(args) == 3
    assert isinstance(args[0], Concolic)
    assert isinstance(args[1], Concolic)
    assert args[2].__class__ is NonPrimitive
    assert prim_args["x"] == ""
    assert prim_args["y"] == 1.25
    assert prim_args["z"] is engine.LazyLoading
    assert kwargs == {}
    assert engine.var_to_types == {"x_VAR": "String", "y_VAR": "Real"}


def test_argument_builder_preserves_existing_var_to_types() -> None:
    def target(value: int):
        return value

    engine = _Engine()
    engine.var_to_types = {"existing_VAR": "Real"}

    ConcolicArgumentBuilder(engine).build(target, {"value": 1}, {"value": 1})

    assert engine.var_to_types == {"existing_VAR": "Real"}
