# PyCT Experiment Entrypoints

This repo's canonical CLI surfaces are:

- `python -m pyct`
- `python -m pyct.shap`
- `python -m pyct.stats`

Useful local validation commands:

- `uv run python -m pyct --help`
- `uv run python -m pyct.shap --help`
- `uv run python -m pyct.stats --help`
- `.venv/bin/python -m pytest -q test`

Common attack flags:

- `--dataset`
- `--model-name`
- `--attack-mode`
- `--pixel-search`
- `--num-process`
- `--first-n`
- `--timeout`
- `--solver-run-timeout`
- `--score-alpha`
- `--symbolic-path-threshold`

Current repo notes:

- The main output tree is under `exp/`.
- `shap_value/` and `shap_value_all_layer/` are local artifact caches, not source.
- `cvc5` is the default solver.
- Datasets default to `~/.keras/datasets` unless `PYCT_KERAS_HOME` overrides it.
- The repository does not currently keep a curated set of experiment wrapper shell scripts, so new automation usually starts from scratch.
