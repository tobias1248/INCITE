---
name: pyct-code
description: Use when implementing a feature, fixing a bug, refactoring a small code path, or repairing compile, import, typing, or lint-style issues inside PyCT-Influence-Transformer without taking ownership of broad test execution or long-form docs.
---

# PyCT Code

Use this skill when the request is mainly about changing application code.

## Scope

- Own source changes in `pyct/`, `orchestration/`, `tasks/`, `datasets/`, `engine/`, `modeling/`, `explainability/`, `reporting/`, `libct/`, and `dnnct/`.
- Keep new entry logic inside the existing package boundaries from `AGENTS.md`.
- Prefer the smallest change that preserves the current architecture.

## Workflow

1. Read the affected module and nearby tests first.
2. Change the narrowest coherent slice.
3. Add or update only the minimum regression coverage that is directly required by the change.
4. Hand broader verification to `test` and risk scanning to `review`.

## Repo Constraints

- The canonical entrypoint is `python -m pyct`.
- Python target is 3.9-compatible.
- Reusable logic should stay out of root-level scripts.
- Core paths that should stay healthy are `pyct/`, `orchestration/`, `engine/`, `datasets/`, `explainability/`, and `reporting/`.

## Avoid

- Large speculative test additions.
- Full experiment orchestration as a primary task.
- Long README or migration-note work as a default outcome.
