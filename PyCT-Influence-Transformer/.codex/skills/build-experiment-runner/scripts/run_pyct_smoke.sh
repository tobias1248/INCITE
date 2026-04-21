#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/.codex/logs}"

DATASET="${DATASET:-mnist}"
MODEL_NAME="${MODEL_NAME:-simple_mnist_m6_09585}"
ATTACK_MODE="${ATTACK_MODE:-queue}"
PIXEL_SEARCH="${PIXEL_SEARCH:-1}"
FIRST_N="${FIRST_N:-1}"
NUM_PROCESS="${NUM_PROCESS:-1}"
TIMEOUT="${TIMEOUT:-30}"
SOLVER_RUN_TIMEOUT="${SOLVER_RUN_TIMEOUT:-5}"
SCORE_ALPHA="${SCORE_ALPHA:-0.8}"
SYMBOLIC_PATH_THRESHOLD="${SYMBOLIC_PATH_THRESHOLD:-2000}"

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/pyct_smoke_${DATASET}_${MODEL_NAME}_${ATTACK_MODE}_p${PIXEL_SEARCH}.log"

CMD=(
  "$PYTHON_BIN" -m pyct
  --dataset "$DATASET"
  --model-name "$MODEL_NAME"
  --attack-mode "$ATTACK_MODE"
  --pixel-search "$PIXEL_SEARCH"
  --first-n "$FIRST_N"
  --num-process "$NUM_PROCESS"
  --timeout "$TIMEOUT"
  --solver-run-timeout "$SOLVER_RUN_TIMEOUT"
  --score-alpha "$SCORE_ALPHA"
  --symbolic-path-threshold "$SYMBOLIC_PATH_THRESHOLD"
)

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" | tee "$LOG_FILE"
