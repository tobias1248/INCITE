---
name: pyct-doc
description: Use when writing or updating README content, usage guides, developer notes, migration notes, changelog-style markdown, or examples for PyCT-Influence-Transformer without taking ownership of core runtime logic or correctness validation.
---

# PyCT Doc

Use this skill when the main task is to explain the repo clearly.

## Scope

- `README.md`
- `docs/`
- `refactoring-docs/`
- Markdown examples attached to new features or workflows

## Workflow

1. Confirm the current behavior from code, tests, or existing repo docs.
2. Prefer short runnable commands and concrete paths.
3. Keep terminology aligned with the repo: `python -m pyct`, `python -m pyct.shap`, `python -m pyct.stats`, `exp/`, and local Keras cache paths.
4. When documenting a new workflow, include prerequisites and the smallest useful example.

## Repo Notes

- `cvc5` is the default solver.
- Datasets are expected under `~/.keras/datasets` unless `PYCT_KERAS_HOME` overrides the cache location.
- `exp/`, `shap_value/`, and `shap_value_all_layer/` are artifact-heavy paths and should not be documented as tracked source outputs.

## Avoid

- Changing logic just to match the docs.
- Treating docs work as a substitute for verification.
