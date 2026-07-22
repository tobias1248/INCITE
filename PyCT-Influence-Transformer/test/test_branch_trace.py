from __future__ import annotations

from types import SimpleNamespace

from libct.branch_trace import branch_site_digest
from libct.constraint import Constraint
from libct.executor.child_protocol import ChildProtocol
from libct.path import PathToConstraint
from libct.position import register_current_indices, register_current_layer_number
from libct.explore import ExplorationEngine


class _ConBool:
    def __init__(self, engine, outcome=True) -> None:
        self.engine = engine
        self.expr = [">", "x", "0"]
        self._outcome = outcome

    def __bool__(self):
        return self._outcome


def setup_function() -> None:
    Constraint.global_constraints.clear()
    PathToConstraint.root_constraint = None
    register_current_layer_number(3)
    register_current_indices((1, 2, 3))


def _engine(enabled: bool):
    engine = SimpleNamespace(
        symbolic_enabled=True,
        branch_trace_enabled=enabled,
        branch_model_sha256="abc",
        pushed=[],
    )
    engine.push_constraint = lambda constraint, position: engine.pushed.append((constraint, position))
    return engine


def test_branch_trace_is_opt_in_and_preserves_constraint_push() -> None:
    disabled_engine = _engine(False)
    disabled_path = PathToConstraint()
    disabled_path.add_branch(_ConBool(disabled_engine, True))

    assert disabled_path.branch_trace == []
    assert len(disabled_engine.pushed) == 1

    Constraint.global_constraints.clear()
    PathToConstraint.root_constraint = None
    enabled_engine = _engine(True)
    enabled_path = PathToConstraint()
    enabled_path.add_branch(_ConBool(enabled_engine, True))

    assert len(enabled_path.branch_trace) == 1
    assert enabled_path.branch_trace[0].observed_outcome is True
    assert enabled_path.branch_trace[0].depth == 1
    assert len(enabled_engine.pushed) == 1


def test_branch_site_digest_is_model_and_position_scoped() -> None:
    expr = [">", "x", "0"]

    first = branch_site_digest(expr, (3, (1, 2)), "model-a")
    repeated = branch_site_digest(expr, (3, (1, 2)), "model-a")
    other_model = branch_site_digest(expr, (3, (1, 2)), "model-b")
    other_position = branch_site_digest(expr, (4, (1, 2)), "model-a")

    assert first == repeated
    assert first != other_model
    assert first != other_position


def test_child_event_can_transfer_partial_branch_trace() -> None:
    engine = SimpleNamespace(
        path=SimpleNamespace(branch_trace=[]),
        var_to_types={},
        concolic_name_list=[],
        concolic_flag_dict={},
    )
    event = SimpleNamespace(site_digest="a", observed_outcome=True)

    ChildProtocol(engine).apply_child_shared_state(
        {},
        {
            "branch_trace": (event,),
            "var_to_types": {},
            "concolic_name_list": [],
            "concolic_flag_dict": {},
        },
    )

    assert engine.path.branch_trace == [event]


def test_exploration_event_facade_forwards_partial_branch_trace() -> None:
    class _Protocol:
        def build_child_event_envelope(self, **kwargs):
            return kwargs

    engine = ExplorationEngine.__new__(ExplorationEngine)
    engine._child_protocol = _Protocol()
    trace = (SimpleNamespace(site_digest="a"),)

    envelope = engine._build_child_event_envelope(
        pid=1,
        updated_args={},
        result="timeout",
        event_type="soft_timeout",
        message="timeout",
        branch_trace=trace,
    )

    assert envelope["branch_trace"] is trace
