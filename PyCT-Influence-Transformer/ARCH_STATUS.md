# PyCT Architecture Status

This branch starts the engineering track for turning PyCT into a reusable research tool, a PyPI package, and an ICSE 2027 tool-paper artifact. The documents in `refactoring-docs/` describe the target KLEE-inspired runtime architecture; this file records what is true in the repository today.

## Current Baseline

- Canonical source entrypoint remains `python -m pyct`.
- Installable console scripts are now declared for `pyct`, `pyct-doctor`, `pyct-shap`, and `pyct-stats`.
- The repository still exposes multiple top-level packages (`pyct`, `orchestration`, `tasks`, `datasets`, `engine`, `modeling`, `explainability`, `reporting`, `libct`, `dnnct`) to preserve existing imports.
- `libct/explore.py` remains the monolithic concolic engine and has not yet been split into Searcher, Executor, State, and Engine modules.
- Artifact-heavy paths remain local runtime outputs: `exp/`, `shap_value/`, and `shap_value_all_layer/`.

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

The KLEE-inspired plan in `refactoring-docs/` is not implemented yet. The next runtime-focused batch should start with a compatibility-preserving extraction of state/searcher primitives from `libct/explore.py`, backed by unit tests from `refactoring-docs/07-INTEGRATION-CHECKLIST.md`.
