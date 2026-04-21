---
name: pyct-test
description: Use when validating changes in PyCT-Influence-Transformer by writing or updating tests, running focused pytest targets, performing bounded smoke runs, or producing reproducible failure reports without taking ownership of feature design.
---

# PyCT Test

Use this skill when the request is about verification.

## Scope

- Own tests under `test/`.
- Run focused validation commands and smoke checks for `python -m pyct`, `python -m pyct.shap`, and `python -m pyct.stats`.
- Prefer local, deterministic validation over long experiment runs.

## Workflow

1. Start with the smallest test or command that can fail for the changed behavior.
2. Prefer unit tests with `monkeypatch`, `tmp_path`, or small payload fixtures.
3. For CLI or pipeline changes, add a bounded smoke run with small `--first-n`, `--timeout`, and `--solver-run-timeout` when the environment supports it.
4. Report exact failure commands and observed errors.

## Preferred Commands

- `.venv/bin/python -m pytest -q test`
- `uv run python -m pyct --help`
- `uv run python -m pyct.shap --help`
- `uv run python -m pyct.stats --help`

## Avoid

- Large production-code rewrites.
- Driving feature design.
- Network-dependent dataset downloads or GPU-only assumptions.
