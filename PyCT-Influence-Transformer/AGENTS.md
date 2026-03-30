# Repository Guidelines

## Project Structure & Module Organization
The canonical entrypoint is `python -m pyct`. Application-facing code lives in `pyct/`; workflow orchestration is in `orchestration/`; task schemas and dataset-family builders are in `tasks/`; dataset adapters are in `datasets/`; runtime wiring is in `engine/`; Keras compatibility helpers are in `modeling/`; SHAP and attribution utilities are in `explainability/`; result aggregation lives in `reporting/`. Core concolic/runtime internals remain in `libct/` and `dnnct/`. Tests live under `test/`.

## Build, Test, and Development Commands
- `uv sync`: create/update `.venv` from `pyproject.toml`.
- `uv run python -m pyct --help`: inspect the main attack CLI.
- `uv run python -m pyct.shap --help`: inspect SHAP preprocessing options.
- `uv run python -m pyct.stats --help`: inspect experiment reporting options.
- `uv run pytest`: run the full test suite.
- `.venv/bin/python -m pytest -q test`: preferred direct test command in local automation.

## Coding Style & Naming Conventions
Use Python 3.9-compatible code, 4-space indentation, and `snake_case` for modules, functions, and variables. Keep package boundaries explicit: new attack entry logic belongs in `pyct/`, orchestration in `orchestration/`, and reusable logic should not be added to root-level scripts. Prefer small functions, typed dataclasses or typed payloads in `tasks/types.py`, and descriptive module names such as `predictor_runtime.py` or `experiment_stats.py`.

## Testing Guidelines
The project uses `pytest`. Keep tests under `test/`, name them `test_*.py`, and group them by subsystem or contract (for example `test/test_engine_predictor_runtime.py` or `test/test_orchestration_launcher.py`).

Prefer fast, deterministic tests:
- Default to unit tests with `monkeypatch`, `tmp_path`, and `capsys`.
- Keep tests offline; do not require network access or on-demand dataset downloads.
- Avoid GPU-specific assumptions; model/bootstrap tests should pass on CPU-only machines.
- When dataset access is involved, use local cache helpers or patched loaders instead of real downloads.

Every behavior change should add or update regression coverage for the real failure mode. Prioritize:
- CLI validation and module entrypoints: `pyct`, `pyct.shap`, and `pyct.stats`
- Dataset cache resolution and local loader behavior
- `.h5` model bootstrap and predictor runtime wiring
- Orchestration flow across `launcher`, `runners`, `executor`, and `progress`
- SHAP/pixel-selection logic and report/stat parsing edge cases

Before committing:
- Run `uv run pytest` or `.venv/bin/python -m pytest -q test`.
- If you changed CLI or docs for commands, smoke-test:
  - `uv run python -m pyct --help`
  - `uv run python -m pyct.shap --help`
  - `uv run python -m pyct.stats --help`
- If you changed the attack pipeline, dataset adapters, model loading, orchestration, or SHAP selection, run at least one bounded smoke test through the real CLI with tiny settings such as `--first-n 1`, a small `--timeout`, and a small `--solver-run-timeout`, using cached local datasets/models.

Do not lower coverage for touched core-path modules without a clear reason. The main path that should stay exercised is:
- `pyct/`
- `orchestration/`
- `engine/`
- `datasets/`
- `explainability/`
- `reporting/`

## Commit & Pull Request Guidelines
Follow the existing history style: `type(scope): summary`, e.g. `refactor(repo): migrate attack pipeline to modular CLI packages` or `feat(shap): add patch-based selector`. Keep commits narrowly scoped and explain behavior changes, not file shuffles. PRs should describe the user-visible impact, affected datasets/models, new commands, and any required local prerequisites such as `cvc5` or cached Keras datasets.

## Security & Configuration Tips
Do not commit local artifacts from `exp/`, `shap_value/`, or `shap_value_all_layer/`. By default datasets are expected in `~/.keras/datasets`; set `PYCT_KERAS_HOME` to use a different local cache, and only set `PYCT_ALLOW_DATASET_DOWNLOAD=1` when downloads are intentionally allowed.
