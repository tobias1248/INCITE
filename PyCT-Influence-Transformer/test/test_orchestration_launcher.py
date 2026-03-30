from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import logging
import sys

import pytest

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
    created = []

    def __init__(self) -> None:
        self.items = []
        type(self).created.append(self)

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


class _WorkerQueue:
    def __init__(self, items) -> None:
        self.items = list(items)
        self.task_done_count = 0

    def get(self, timeout=None):
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def task_done(self) -> None:
        self.task_done_count += 1


def _make_args(**overrides):
    base = dict(
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
    base.update(overrides)
    return SimpleNamespace(**base)


def _install_runtime_fakes(monkeypatch) -> None:
    _FakeProcess.created = []
    _FakeQueue.created = []
    monkeypatch.setattr(launcher, "Event", _FakeEvent)
    monkeypatch.setattr(launcher, "JoinableQueue", _FakeQueue)
    monkeypatch.setattr(launcher, "Process", _FakeProcess)
    monkeypatch.setattr(launcher.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher.signal, "signal", lambda *_args, **_kwargs: None)


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

    _install_runtime_fakes(monkeypatch)
    monkeypatch.setattr(launcher, "collect_stage_cases", lambda inputs: [])
    monkeypatch.setattr(launcher, "should_run_payload", lambda payload, force_refresh: True)
    monkeypatch.setattr(launcher, "mnist_transformer_shap", fake_mnist_shap)
    monkeypatch.setattr(launcher, "mnist_transformer_random", lambda *args, **kwargs: [])

    args = _make_args()

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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_process", 0, "--num-process must be >= 1"),
        ("first_n", 0, "--first-n must be >= 1"),
        ("timeout", 0, "--timeout must be >= 1 second"),
        ("spawn_delay", -0.1, "--spawn-delay must be non-negative"),
    ],
)
def test_run_launcher_validates_basic_numeric_args(monkeypatch, field, value, message) -> None:
    _install_runtime_fakes(monkeypatch)
    args = _make_args(**{field: value})

    with pytest.raises(ValueError, match=message):
        launcher.run_launcher(args)


def test_run_launcher_selects_random_builder_for_random_mode(monkeypatch) -> None:
    calls = []
    payload = {"idx": 0, "save_exp": {}, "con_dict": {}, "solve_order_stack": False}
    _install_runtime_fakes(monkeypatch)
    monkeypatch.setattr(launcher, "collect_stage_cases", lambda inputs: [])
    monkeypatch.setattr(launcher, "should_run_payload", lambda payload, force_refresh: True)
    monkeypatch.setattr(launcher, "mnist_transformer_shap", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        launcher,
        "mnist_transformer_random",
        lambda model_name, **kwargs: calls.append((model_name, kwargs)) or [dict(payload)],
    )

    launcher.run_launcher(_make_args(attack_mode="random"))

    assert calls == [
        (
            "demo",
            {
                "first_n_img": range(0, 1),
                "ton_values": (1,),
                "force": True,
                "base_seed": 2024,
                "attack_mode": "random_solver1s",
            },
        )
    ]


def test_run_launcher_selects_shap_builder_for_random_assign_with_shap_pixels(monkeypatch) -> None:
    calls = []
    payload = {"idx": 0, "save_exp": {}, "con_dict": {}, "solve_order_stack": False}
    _install_runtime_fakes(monkeypatch)
    monkeypatch.setattr(launcher, "collect_stage_cases", lambda inputs: [])
    monkeypatch.setattr(launcher, "should_run_payload", lambda payload, force_refresh: True)
    monkeypatch.setattr(
        launcher,
        "mnist_transformer_shap",
        lambda model_name, **kwargs: calls.append((model_name, kwargs)) or [dict(payload)],
    )
    monkeypatch.setattr(launcher, "mnist_transformer_random", lambda *args, **kwargs: [])

    launcher.run_launcher(_make_args(attack_mode="random-assign", pixel_source="shap"))

    assert calls == [
        (
            "demo",
            {
                "first_n_img": range(0, 1),
                "force": True,
                "ton_values": (1,),
                "exp_prefix": "random_assign_shap",
                "attack_mode": "random-assign_solver1s",
            },
        )
    ]


def test_run_launcher_selects_random_builder_for_random_assign_with_random_pixels(monkeypatch) -> None:
    calls = []
    payload = {"idx": 0, "save_exp": {}, "con_dict": {}, "solve_order_stack": False}
    _install_runtime_fakes(monkeypatch)
    monkeypatch.setattr(launcher, "collect_stage_cases", lambda inputs: [])
    monkeypatch.setattr(launcher, "should_run_payload", lambda payload, force_refresh: True)
    monkeypatch.setattr(launcher, "mnist_transformer_shap", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        launcher,
        "mnist_transformer_random",
        lambda model_name, **kwargs: calls.append((model_name, kwargs)) or [dict(payload)],
    )

    launcher.run_launcher(_make_args(attack_mode="random-assign", pixel_source="random"))

    assert calls == [
        (
            "demo",
            {
                "first_n_img": range(0, 1),
                "ton_values": (1,),
                "force": True,
                "base_seed": 2024,
                "exp_prefix": "random_assign_random",
                "attack_mode": "random-assign_solver1s",
            },
        )
    ]


def test_run_launcher_adds_patchshap_suffix_for_cifar10(monkeypatch) -> None:
    calls = []
    payload = {"idx": 0, "save_exp": {}, "con_dict": {}, "solve_order_stack": False}
    _install_runtime_fakes(monkeypatch)
    monkeypatch.setattr(launcher, "collect_stage_cases", lambda inputs: [])
    monkeypatch.setattr(launcher, "should_run_payload", lambda payload, force_refresh: True)
    monkeypatch.setattr(
        launcher,
        "cifar10_transformer_shap",
        lambda model_name, **kwargs: calls.append((model_name, kwargs)) or [dict(payload)],
    )
    monkeypatch.setattr(launcher, "cifar10_transformer_random", lambda *args, **kwargs: [])

    launcher.run_launcher(
        _make_args(dataset="cifar10", attack_mode="shap", pixel_selector="patch-shap")
    )

    assert calls == [
        (
            "demo",
            {
                "first_n_img": range(0, 1),
                "force": True,
                "ton_values": (1,),
                "attack_mode": "shap_patchshap_solver1s",
                "pixel_selector": "patch-shap",
            },
        )
    ]


def test_run_launcher_skips_payloads_when_progress_says_not_to_run(monkeypatch) -> None:
    payload = {"idx": 0, "save_exp": {}, "con_dict": {}, "solve_order_stack": False}
    _install_runtime_fakes(monkeypatch)
    monkeypatch.setattr(launcher, "collect_stage_cases", lambda inputs: [])
    monkeypatch.setattr(launcher, "should_run_payload", lambda payload, force_refresh: False)
    monkeypatch.setattr(launcher, "mnist_transformer_shap", lambda *args, **kwargs: [dict(payload)])
    monkeypatch.setattr(launcher, "mnist_transformer_random", lambda *args, **kwargs: [])

    launcher.run_launcher(_make_args(force_refresh=False))

    assert _FakeProcess.created == []
    assert _FakeQueue.created and _FakeQueue.created[0].items == []


def test_run_launcher_enqueues_only_payloads_selected_by_stage_progress(monkeypatch, tmp_path: Path) -> None:
    _install_runtime_fakes(monkeypatch)
    monkeypatch.setattr(launcher, "mnist_transformer_shap", lambda *args, **kwargs: [])
    monkeypatch.setattr(launcher, "mnist_transformer_random", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        launcher,
        "collect_stage_cases",
        lambda inputs: [
            {
                "base_payload": {"idx": 0, "save_exp": {"input_name": "case_0"}},
                "plans": {
                    1: {"con_dict": {"v_0_0_0": 1}, "save_exp": {"input_name": "case_0"}},
                    2: {"con_dict": {"v_0_0_1": 1}, "save_exp": {"input_name": "case_0"}},
                },
                "save_dir": str(tmp_path / "case_0"),
            }
        ],
    )
    monkeypatch.setattr(
        launcher,
        "should_run_ton",
        lambda case, ton, ton_sequence, force_refresh: ton == 1,
    )
    monkeypatch.setattr(launcher, "load_stats_payload", lambda path: ({"meta": {}}, None))
    monkeypatch.setattr(launcher, "extract_last_ton", lambda stats: 1)
    monkeypatch.setattr(launcher, "derive_stage_outcome_payload", lambda stats: (False, "stop"))
    updates = []
    monkeypatch.setattr(
        launcher,
        "update_ton_progress_stats",
        lambda stats_path, **kwargs: updates.append((Path(stats_path), kwargs)),
    )

    launcher.run_launcher(_make_args(pixel_search=(1, 2)))

    queue_items = _FakeQueue.created[0].items
    task = next(item for item in queue_items if isinstance(item, dict))
    assert task["con_dict"] == {"v_0_0_0": 1}
    assert task["save_exp"]["ton"] == 1
    assert task["save_exp"]["ton_next"] == 2
    assert updates[0][1]["current_ton"] == 1


def test_run_launcher_clears_score_alpha_env_when_not_set(monkeypatch) -> None:
    payload = {"idx": 0, "save_exp": {}, "con_dict": {}, "solve_order_stack": False}
    _install_runtime_fakes(monkeypatch)
    monkeypatch.setattr(launcher, "collect_stage_cases", lambda inputs: [])
    monkeypatch.setattr(launcher, "should_run_payload", lambda payload, force_refresh: True)
    monkeypatch.setattr(launcher, "mnist_transformer_shap", lambda *args, **kwargs: [dict(payload)])
    monkeypatch.setattr(launcher, "mnist_transformer_random", lambda *args, **kwargs: [])
    monkeypatch.setenv("PYCT_SCORE_ALPHA", "0.8")

    launcher.run_launcher(_make_args(score_alpha=None))

    assert "PYCT_SCORE_ALPHA" not in launcher.os.environ


def test_worker_exits_immediately_when_shutdown_requested(monkeypatch, caplog) -> None:
    event = _FakeEvent()
    event.set()
    queue = _WorkerQueue([])
    monkeypatch.setattr(launcher.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher.os, "getpid", lambda: 1234)
    monkeypatch.setattr(launcher, "QueueRunner", lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(launcher, "RandomAssignRunner", lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(launcher, "ShapRunner", lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")))

    with caplog.at_level(logging.INFO, logger="ct.cli"):
        launcher._worker(queue, 3, True, 15, 1, False, "queue", "random", 2024, event)

    assert "[WORKER-SHUTDOWN]" in caplog.text
    assert "[WORKER-EXIT]" in caplog.text


def test_worker_uses_queue_runner_and_handles_empty_queue(monkeypatch) -> None:
    created = []
    processed = []
    queue = _WorkerQueue([launcher.py_queue.Empty(), {"idx": 7}, None])

    class _FakeRunner:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)

        def run_tasks(self, tasks) -> None:
            processed.extend(tasks)

    monkeypatch.setattr(launcher.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher.os, "getpid", lambda: 2222)
    monkeypatch.setattr(launcher, "QueueRunner", _FakeRunner)

    launcher._worker(queue, 5, True, 15, 2, False, "queue", "random", 2024, _FakeEvent())

    assert created == [
        {
            "timeout": 5,
            "constraint_build_timeout": True,
            "constraint_build_timeout_seconds": 15,
            "solver_run_timeout": 2,
            "norm": False,
            "collect_constraints_with": "queue",
        }
    ]
    assert processed == [{"idx": 7}]
    assert queue.task_done_count == 2


def test_worker_logs_task_errors_and_continues(monkeypatch, caplog) -> None:
    calls = []
    queue = _WorkerQueue([{"idx": 1}, None])

    class _FakeRunner:
        def __init__(self, **kwargs) -> None:
            return None

        def run_tasks(self, tasks) -> None:
            calls.extend(tasks)
            raise RuntimeError("runner boom")

    monkeypatch.setattr(launcher.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher.os, "getpid", lambda: 5555)
    monkeypatch.setattr(launcher, "ShapRunner", _FakeRunner)

    with caplog.at_level(logging.ERROR, logger="ct.cli"):
        launcher._worker(queue, 5, True, 15, 2, False, "shap", "random", 2024, _FakeEvent())

    assert calls == [{"idx": 1}]
    assert "[WORKER-TASK-ERROR]" in caplog.text
    assert queue.task_done_count == 2


def test_worker_uses_random_assign_runner_and_handles_interrupt(monkeypatch, caplog) -> None:
    created = []
    queue = _WorkerQueue([KeyboardInterrupt()])

    class _FakeRunner:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)

    monkeypatch.setattr(launcher.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher.os, "getpid", lambda: 7777)
    monkeypatch.setattr(launcher, "RandomAssignRunner", _FakeRunner)

    with caplog.at_level(logging.INFO, logger="ct.cli"):
        launcher._worker(queue, 6, False, 12, None, True, "random-assign", "shap", 99, _FakeEvent())

    assert created == [
        {
            "timeout": 6,
            "constraint_build_timeout": False,
            "constraint_build_timeout_seconds": 12,
            "solver_run_timeout": None,
            "norm": True,
            "pixel_source": "shap",
            "base_seed": 99,
        }
    ]
    assert "[WORKER-INTERRUPT]" in caplog.text
