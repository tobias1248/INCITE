# PyCT Architecture Status

This branch starts the engineering track for turning PyCT into a reusable research tool, a PyPI package, and an ICSE 2027 tool-paper artifact. The documents in `refactoring_docs/` describe the target KLEE-inspired runtime architecture; this file records what is true in the repository today.

## Current Baseline

- Canonical source entrypoint remains `python -m pyct`.
- Installable console scripts are now declared for `pyct`, `pyct-doctor`, `pyct-shap`, and `pyct-stats`.
- The repository still exposes multiple top-level packages (`pyct`, `orchestration`, `tasks`, `datasets`, `engine`, `modeling`, `explainability`, `reporting`, `libct`, `dnnct`) to preserve existing imports.
- `libct/explore.py` remains the large compatibility coordinator for the concolic engine, but searcher, executor, and state compatibility modules now exist under `libct/searcher/`, `libct/executor/`, and `libct/state/`.
- Artifact-heavy paths remain local runtime outputs: `exp/`, `shap_value/`, and `shap_value_all_layer/`.
- Solver attempt artifacts are now written outside the primary `stats.json` payload as bounded side outputs such as `solver_iter1_top3.jsonl` and `solver_iter1_top3_smt/`.

## First-Batch Engineering Scope

- Package metadata now supports editable/installable package workflows instead of `tool.uv.package = false`.
- Package metadata now declares console scripts, classifiers, project URLs, keywords, and explicit proprietary license metadata.
- README usage now documents module entrypoints, console scripts, and `pyct-doctor`.
- `pyct-doctor` provides a lightweight local readiness check for Python version, runtime dependency discoverability, solver availability, model files, dataset cache, SHAP artifacts, and output directory writability.
- The doctor command intentionally avoids importing TensorFlow, Keras, or SHAP. It uses import discovery only, so `--help` and CI tests stay lightweight.
- Tests cover doctor success/failure paths without network access, dataset downloads, GPU assumptions, or real solver execution.

## Known Gaps Before PyPI

- The package still publishes generic top-level names such as `datasets`, which is risky for downstream environments. A namespace migration should be planned after runtime refactoring stabilizes.
- No source distribution or wheel build job has been added yet.
- No artifact manifest format exists yet for paper reproduction bundles.
- `pyct-doctor` checks prerequisite presence only; it does not run an end-to-end attack smoke test.

## Runtime Refactoring Gap

The KLEE-inspired plan in `refactoring_docs/` is now in the compatibility modularization stage:

- `libct/state/` defines structured `ExecutionState`, `StateManager`, and `ConstraintWorkItem` primitives.
- `libct/searcher/` owns stack, queue, priority, and random constraint worklist strategies.
- `libct/executor/` introduces a thin executor boundary around the legacy concolic and primitive execution methods.
- `libct/explore.py` still owns the main exploration loop, recorder interactions, solver integration, and subprocess execution internals, but it no longer directly implements raw stack/queue/heap search policy for normal runs.
- `libct/record.py` and `libct/solver.py` now preserve bounded solver-attempt observability without embedding full SMT formulas directly in `stats.json`.

Deferred runtime work:

- Shrink `libct/explore.py` into a smaller coordinator after solver, recorder, and subprocess responsibilities are isolated.
- Move from constraint-level `ConstraintWorkItem` scheduling toward full path-level `ExecutionState` scheduling once compatibility tests cover the current output contracts.
- Add real CLI attack smoke validation when local cached datasets, model files, SHAP artifacts, and `cvc5` are available.

## Latest Validation Snapshot

- Current branch head: `e575769 feat(record): persist solver attempt artifacts`.
- Pre-sync working tree status: clean; this sync introduces documentation-only updates.
- Full local test command passed on 2026-05-07: `.venv/bin/python -m pytest -q test` (`272 passed`).
- CLI help smoke checks passed on 2026-05-07 for `python -m pyct --help`, `python -m pyct.shap --help`, and `python -m pyct.stats --help`.
- Environment-dependent real attack smoke tests remain deferred until local datasets, model artifacts, SHAP artifacts, and solver availability are confirmed.
