from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import libct.explore as explore


class _RecorderStub:
    def __init__(self) -> None:
        self.original_label = None
        self.attack_label = None
        self.original_input = None
        self.gen_constraint = []
        self.extra_meta = {}
        self.total_iter = -1
        self.queue_max = 0
        self.queue_last = 0

    def start(self) -> None:
        return None

    def iter_start(self, _solver) -> None:
        return None

    def execution_start(self) -> None:
        return None

    def execution_end(self) -> None:
        return None

    def iter_end(self, _solver_stats, _solve_constr_num) -> None:
        self.total_iter += 1

    def solve_constr_start(self) -> None:
        return None

    def solve_constr_end(self) -> None:
        return None

    def first_execution_end(self) -> None:
        return None

    def save_original_input(self, inputs) -> None:
        self.original_input = dict(inputs)

    def save_stats_dict(self, constraint_complexity=None) -> None:
        return None

    def save_sat_input(self, _inputs) -> None:
        return None

    def find_adversarial_input(self, inputs, attack_label) -> None:
        self.attack_label = attack_label
        self.adversarial_input = dict(inputs)

    def total_timeout(self) -> None:
        return None

    def no_ctr_to_solve(self) -> None:
        return None


def _make_engine(validation_execute):
    engine = explore.ExplorationEngine.__new__(explore.ExplorationEngine)
    engine.validation_execute = validation_execute
    engine.normalize = None
    engine.limit_change_range = None
    engine.constraints_collection_type = "queue"
    engine.constraints_to_solve = deque([object()])
    engine.idx = 0
    engine.only_first_forward = False
    engine.symbolic_path_threshold = None
    engine.symbolic_enabled = True
    engine.symbolic_disabled_at_path_len = None
    engine.previous_result = None
    engine.original_args = {}
    return engine


def test_execution_loop_uses_validation_predictor_for_labels(monkeypatch) -> None:
    recorder = _RecorderStub()
    explore.recorder = recorder
    explore.Solver.stats = {
        "sat_number": 0,
        "sat_time": 0,
        "unsat_number": 0,
        "unsat_time": 0,
        "otherwise_number": 0,
        "otherwise_time": 0,
    }

    validation_calls = []
    search_calls = []

    def validation_execute(**data):
        validation_calls.append(dict(data))
        return 0 if data["v_0_0"] == 0.0 else 1

    engine = _make_engine(validation_execute)

    def fake_one_execution(all_args, concolic_dict):
        search_calls.append(dict(all_args))
        return True

    monkeypatch.setattr(engine, "_one_execution", fake_one_execution)
    monkeypatch.setattr(
        explore.Solver,
        "find_model_from_constraint",
        lambda *_args, **_kwargs: {"v_0_0": 1.0},
    )

    timed_out = engine._execution_loop(0, {"v_0_0": 0.0}, {})

    assert timed_out is False
    assert recorder.original_label == 0
    assert recorder.attack_label == 1
    assert validation_calls == [{"v_0_0": 0.0}, {"v_0_0": 1.0}]
    assert search_calls == [{"v_0_0": 0.0}]
