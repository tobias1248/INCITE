from __future__ import annotations

from pathlib import Path
import importlib
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyct.stats as stats_mod


def test_importing_pyct_stats_is_lightweight() -> None:
    reloaded = importlib.reload(stats_mod)

    assert hasattr(reloaded, "main")
