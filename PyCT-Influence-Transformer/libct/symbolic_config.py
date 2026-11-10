"""Settings for semantics-preserving symbolic serialization optimizations."""

import os


def resolve_smt_serialization() -> str:
    mode = os.environ.get("PYCT_SMT_SERIALIZATION", "raw").strip().lower()
    if mode not in {"raw", "shared"}:
        raise ValueError("PYCT_SMT_SERIALIZATION must be 'raw' or 'shared', got {!r}".format(mode))
    return mode
