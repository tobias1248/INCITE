from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyct.main as main_mod


def test_python_m_pyct_invokes_main(monkeypatch) -> None:
    events = []

    monkeypatch.setattr(main_mod, "main", lambda: events.append("main"))

    runpy.run_module("pyct", run_name="__main__")

    assert events == ["main"]
