from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import orchestration.launcher as launcher


class _FakeEvent:
    def __init__(self) -> None:
        self._is_set = False

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        self._is_set = True


class _FakeQueue:
    def __init__(self) -> None:
        self.items = []

    def put(self, item) -> None:
        self.items.append(item)

    def join(self) -> None:
        return None


class _FakeProcess:
    created = []

    def __init__(self, target, args) -> None:
        self.target = target
        self.args = args
        self.started = False
        self.pid = 4321
        _FakeProcess.created.append(self)

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return False

    def join(self, timeout=None) -> None:
        return None

    def terminate(self) -> None:
        return None


def test_run_launcher_selects_queue_builder_and_sets_env(monkeypatch) -> None:
    builder_calls = []
    fake_inputs = [
        {
            "model_name": "demo",
            "idx": 0,
            "save_exp": {"input_name": "case_0", "attack_mode": "queue_solver1s"},
            "in_dict": {"v_0_0": 1.0},
            "con_dict": {"v_0_0": 1},
            "solve_order_stack": False,
        }
    ]

    def fake_mnist_shap(model_name, **kwargs):
        builder_calls.append((model_name, kwargs))
        return [dict(fake_inputs[0])]

    monkeypatch.setattr(launcher, "Event", _FakeEvent)
    monkeypatch.setattr(launcher, "JoinableQueue", _FakeQueue)
    monkeypatch.setattr(launcher, "Process", _FakeProcess)
    monkeypatch.setattr(launcher.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "collect_stage_cases", lambda inputs: [])
    monkeypatch.setattr(launcher, "should_run_payload", lambda payload, force_refresh: True)
    monkeypatch.setattr(launcher, "mnist_transformer_shap", fake_mnist_shap)
    monkeypatch.setattr(launcher, "mnist_transformer_random", lambda *args, **kwargs: [])

    args = SimpleNamespace(
        model_name="demo",
        num_process=1,
        timeout=3,
        constraint_build_timeout=True,
        constraint_build_timeout_seconds=15,
        solver_run_timeout=1,
        score_alpha=None,
        symbolic_path_threshold=2000,
        enable_constraint_log=False,
        pixel_search=(1,),
        attack_mode="queue",
        dataset="mnist",
        random_seed=2024,
        pixel_source="random",
        pixel_selector="pixel-shap",
        norm_01=False,
        first_n=1,
        spawn_delay=0.0,
        force_refresh=True,
    )

    launcher.run_launcher(args)

    assert builder_calls == [
        (
            "demo",
            {
                "first_n_img": range(0, 1),
                "force": True,
                "ton_values": (1,),
                "exp_prefix": "queue",
                "attack_mode": "queue_solver1s",
            },
        )
    ]
    assert launcher.os.environ["PYCT_TIMEOUT"] == "3"
    assert launcher.os.environ["PYCT_CONSTRAINT_BUILD_TIMEOUT_ENABLED"] == "1"
    assert launcher.os.environ["PYCT_CONSTRAINT_BUILD_TIMEOUT_SECONDS"] == "15"
    assert launcher.os.environ["PYCT_SYMBOLIC_PATH_THRESHOLD"] == "2000"
    assert _FakeProcess.created and _FakeProcess.created[0].started is True
