import pytest

from libct.symbolic_config import resolve_smt_serialization


def test_smt_serialization_defaults_to_raw(monkeypatch):
    monkeypatch.delenv("PYCT_SMT_SERIALIZATION", raising=False)
    assert resolve_smt_serialization() == "raw"


@pytest.mark.parametrize("value,expected", [("raw", "raw"), (" SHARED ", "shared")])
def test_smt_serialization_resolves_explicit_mode(monkeypatch, value, expected):
    monkeypatch.setenv("PYCT_SMT_SERIALIZATION", value)
    assert resolve_smt_serialization() == expected


@pytest.mark.parametrize("value", ["", "dag", "exact_affine"])
def test_smt_serialization_rejects_invalid_mode(monkeypatch, value):
    monkeypatch.setenv("PYCT_SMT_SERIALIZATION", value)
    with pytest.raises(ValueError, match="PYCT_SMT_SERIALIZATION"):
        resolve_smt_serialization()
