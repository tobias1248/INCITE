---
name: build-experiment-runner
description: Use when PyCT-Influence-Transformer needs reproducible experiment wrapper scripts, batch runs, sweeps, log-path cleanup, timeout handling, or sequential and parallel bash workflows around the existing CLI entrypoints.
---

# Build Experiment Runner

Use this skill when the task is to turn repeated experiment commands into reusable scripts.

## Scope

- Wrap `python -m pyct`, `python -m pyct.shap`, and `python -m pyct.stats` in reproducible bash entrypoints.
- Standardize environment setup, arguments, output roots, log files, and fail-fast behavior.
- Support both sequential runs and bounded parallel execution.

## Workflow

1. Identify the repeated CLI shape.
2. Start from the bundled templates in `scripts/` when possible.
3. Keep parameters explicit: dataset, model name, attack mode, pixel-search, process count, first-n, timeout, solver-run-timeout, log path, and output root.
4. Print the exact command before running it.
5. Leave model logic and research analysis outside the script layer.

## Repo-Specific Guidance

- This repo currently has almost no checked-in experiment shell wrappers, so treat the bundled templates as the starting point.
- Prefer `set -euo pipefail`.
- Keep logs outside tracked experiment artifacts when possible; `.codex/logs/` is a safe default for local references.
- For correctness validation, hand off to `pyct-test`.

## References

- See `references/repo_entrypoints.md` for the repo's canonical commands and output-path notes.

## Bundled Templates

- `scripts/run_pyct_smoke.sh`
- `scripts/run_attack_batch.sh`
