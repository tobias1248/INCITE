# PyCT Engineering Roadmap for PyPI and ICSE 2027

## Summary

This roadmap defines the long-term engineering track for turning PyCT from a research repository into a reusable Python package, a reproducible experimental artifact, and a credible ICSE 2027 tool-paper submission.

The work has three linked goals:

1. Make PyCT installable and usable through stable package entrypoints.
2. Refactor the attack runtime into testable modules without changing CLI behavior.
3. Produce a reproducible artifact story that reviewers can install, inspect, run, and compare.

`ARCH_STATUS.md` records the current repository state. This document records the target direction and the order in which the work should land.

## Milestones

### Milestone 1: Package Readiness

Goal: make PyCT installable in editable and wheel-based workflows while preserving the canonical `python -m pyct` entrypoint.

Expected outcomes:

- `pyproject.toml` contains complete package metadata, console scripts, classifiers, license metadata, and project URLs.
- The command set is documented around `pyct`, `pyct-doctor`, `pyct-shap`, and `pyct-stats`.
- `pyct-doctor` reports local readiness for solver, datasets, models, SHAP artifacts, and output paths without importing heavyweight ML libraries.
- Packaging tests verify that `python -m pyct --help` and console scripts resolve in a clean environment.
- Artifact-heavy runtime outputs stay untracked: `exp/`, `shap_value/`, and `shap_value_all_layer/`.

### Milestone 2: Runtime Modularization

Goal: split the monolithic concolic runtime into explicit Searcher, Executor, State, and Engine components while preserving current attack semantics.

The first implementation batch should be compatibility modularization: extract the current constraint-level worklist, state primitives, and executor boundary before rewriting the engine around full path-level `ExecutionState` scheduling.

Expected outcomes:

- `libct/explore.py` becomes a compatibility shell or a smaller coordinator instead of the primary location for all runtime behavior.
- Search policy lives behind a `Searcher` abstraction with priority, DFS, BFS, and random implementations.
- Execution state lives in typed data structures with clear ownership and lifecycle rules.
- Symbolic and concrete execution paths are independently testable.
- SHAP-driven alpha scheduling remains available through the priority search path.
- Existing CLI workflows continue to work with the same arguments and output contracts.

The detailed KLEE-inspired design remains in `01-DESIGN-PHILOSOPHY.md` through `08-KLEE-REFERENCE.md`.

### Milestone 3: Reproducible Experiment Layer

Goal: make experiments repeatable without depending on local shell history or hidden machine state.

Expected outcomes:

- Standard experiment wrappers cover tiny smoke runs, bounded benchmark runs, and paper-style batch runs.
- Each run writes a manifest with command, PyCT version or commit, dataset/model identifiers, solver settings, timeouts, environment hints, and output paths.
- Report aggregation through `pyct.stats` can consume those manifests and produce stable summaries.
- Failure modes are documented: missing solver, missing local dataset cache, missing model files, timeout, unsupported runtime dependency, and insufficient disk space.
- Dataset access remains offline by default; downloads require explicit `PYCT_ALLOW_DATASET_DOWNLOAD=1`.

### Milestone 4: Artifact and Documentation Hardening

Goal: give new users and artifact evaluators a direct path from installation to a small validated run.

Expected outcomes:

- README installation paths distinguish package users, developers, and artifact evaluators.
- Documentation includes one CPU-only quickstart, one bounded attack smoke test, one SHAP preprocessing example, and one stats/reporting example.
- Troubleshooting points users to `pyct-doctor` before heavyweight runs.
- A release checklist covers wheel build, source distribution build, clean install smoke tests, and artifact bundle validation.
- Example commands avoid network access unless explicitly marked as dataset-download workflows.

### Milestone 5: ICSE 2027 Tool-Paper Preparation

Goal: align implementation, evaluation, and artifact packaging with a tool-paper narrative.

Expected outcomes:

- The tool contribution is stated as a reproducible concolic testing pipeline for DNN robustness experiments with SHAP-guided prioritization.
- The architecture section maps directly to the runtime modules: CLI, orchestration, dataset/model adapters, predictor runtime, explainability, concolic engine, and reporting.
- Evaluation scripts reproduce a small reviewer-friendly run and a larger paper-scale run from the same interfaces.
- Artifact instructions name exact local prerequisites such as Python version, solver availability, cached Keras datasets, and model files.
- The paper artifact can be rebuilt from a tagged release or archived commit with no untracked source files required.

Official ICSE 2027 submission and artifact deadlines should be confirmed from the conference CFP when preparing the paper calendar.

## Implementation Changes

### Packaging and Entrypoints

- Keep `python -m pyct` as the canonical module entrypoint.
- Maintain console scripts for package workflows.
- Avoid importing TensorFlow, Keras, SHAP, or solver bindings in lightweight commands such as `--help` and `pyct-doctor`.
- Add packaging checks before publishing: build source distribution, build wheel, install wheel into a clean virtual environment, and run help-command smoke tests.

### Namespace and API Boundaries

- Preserve existing top-level imports until runtime refactoring is stable.
- Plan a later namespace migration away from generic top-level package names such as `datasets`.
- Keep attack CLI logic in `pyct/`, orchestration in `orchestration/`, dataset builders in `tasks/`, dataset adapters in `datasets/`, predictor wiring in `engine/`, compatibility helpers in `modeling/`, attribution utilities in `explainability/`, and aggregation in `reporting/`.

### Runtime Refactoring

- Extract state primitives first because search and execution both depend on clear state ownership.
- Introduce searcher abstractions behind tests before changing the full exploration loop.
- Extract executor behavior behind compatibility tests that compare old and new observable outputs.
- Keep each batch small enough to preserve CLI behavior and make regression failures diagnosable.
- Use the integration checklist in `07-INTEGRATION-CHECKLIST.md` as the acceptance guide for runtime work.

### Experiment Artifacts

- Add manifest-first experiment outputs before expanding benchmark scripts.
- Record command-line arguments and resolved local paths, but do not record private absolute paths when a portable relative path is available.
- Prefer deterministic tiny fixtures for CI and reserve heavyweight models or datasets for documented local smoke tests.
- Keep generated experiment outputs out of version control.

### Documentation

- Treat this roadmap as the top-level long-term plan.
- Treat `ARCH_STATUS.md` as the current implementation snapshot.
- Keep KLEE-inspired module docs focused on runtime design details.
- Update user-facing docs whenever CLI commands, packaging assumptions, or artifact workflows change.

## Test Plan

Documentation-only changes should be verified with file and search checks:

- Confirm `refactoring_docs/00-ROADMAP.md` exists and starts with the H1 title in this file.
- Confirm `refactoring_docs/00-OVERVIEW.md` lists `00-ROADMAP.md` as the first quick-navigation row.
- Run `rg "00-ROADMAP|PyPI|ICSE 2027" refactoring_docs` to confirm the roadmap and index are discoverable.

For implementation batches, expand validation according to risk:

- Run `.venv/bin/python -m pytest -q test` for core behavior changes.
- Smoke-test `python -m pyct --help`, `python -m pyct.shap --help`, and `python -m pyct.stats --help` when CLI or docs change.
- Run at least one bounded real CLI smoke test for changes touching attack pipeline, dataset adapters, model loading, orchestration, SHAP selection, or reporting.
- Build and install the package in a clean environment before PyPI release candidates.

## Assumptions

- `refactoring_docs/` is the correct home for the long-term roadmap.
- Existing `01-` through `08-` document names stay unchanged so existing relative links and external references remain valid.
- `00-OVERVIEW.md` remains the KLEE-inspired refactoring guide, while `00-ROADMAP.md` becomes the highest-level engineering and publication roadmap.
- `ARCH_STATUS.md` remains a root-level current-state snapshot.
- Dataset downloads stay opt-in and local cache behavior remains the default for tests and artifact evaluation.
- ICSE 2027 dates and artifact requirements must be rechecked against the official CFP before committing to paper-submission calendar details.
