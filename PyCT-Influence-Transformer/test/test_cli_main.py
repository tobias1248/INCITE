from __future__ import annotations

from pathlib import Path
import importlib
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cli.main as main_mod


def test_importing_cli_main_does_not_import_launcher(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, 'orchestration.launcher', raising=False)

    reloaded = importlib.reload(main_mod)

    assert 'orchestration.launcher' not in sys.modules
    assert hasattr(reloaded, 'main')


def test_main_wires_args_logging_and_launcher(monkeypatch) -> None:
    events = []
    fake_args = object()
    fake_launcher = types.ModuleType('orchestration.launcher')

    def fake_parse(argv):
        events.append(("parse", argv))
        return fake_args

    def fake_configure(args):
        events.append(("configure", args))

    def fake_run(args):
        events.append(("run", args))

    fake_launcher.run_launcher = fake_run
    monkeypatch.setattr(main_mod, 'parse_args', fake_parse)
    monkeypatch.setattr(main_mod, 'configure_logging', fake_configure)
    monkeypatch.setitem(sys.modules, 'orchestration.launcher', fake_launcher)

    main_mod.main(["--dataset", "mnist", "--attack-mode", "queue"])

    assert events == [
        ("parse", ["--dataset", "mnist", "--attack-mode", "queue"]),
        ("configure", fake_args),
        ("run", fake_args),
    ]
