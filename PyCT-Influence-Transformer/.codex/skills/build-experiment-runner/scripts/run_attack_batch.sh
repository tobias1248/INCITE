#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/.codex/logs/batch}"

DATASET="${DATASET:-cifar10}"
MODEL_NAME="${MODEL_NAME:-cifar10_concolic_transformer}"
ATTACK_MODES="${ATTACK_MODES:-shap queue}"
PIXEL_SEARCHES="${PIXEL_SEARCHES:-1 2 4}"
FIRST_N="${FIRST_N:-1}"
NUM_PROCESS="${NUM_PROCESS:-1}"
TIMEOUT="${TIMEOUT:-120}"
SOLVER_RUN_TIMEOUT="${SOLVER_RUN_TIMEOUT:-10}"
SCORE_ALPHA="${SCORE_ALPHA:-0.8}"
SYMBOLIC_PATH_THRESHOLD="${SYMBOLIC_PATH_THRESHOLD:-2000}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
FAIL_FAST="${FAIL_FAST:-1}"

mkdir -p "$LOG_DIR"
read -r -a mode_list <<< "$ATTACK_MODES"
read -r -a pixel_list <<< "$PIXEL_SEARCHES"

overall_status=0
running=0

run_one() {
  local attack_mode="$1"
  local pixel_search="$2"
  local log_file="$LOG_DIR/${attack_mode}_p${pixel_search}.log"
  local cmd=(
    "$PYTHON_BIN" -m pyct
    --dataset "$DATASET"
    --model-name "$MODEL_NAME"
    --attack-mode "$attack_mode"
    --pixel-search "$pixel_search"
    --first-n "$FIRST_N"
    --num-process "$NUM_PROCESS"
    --timeout "$TIMEOUT"
    --solver-run-timeout "$SOLVER_RUN_TIMEOUT"
    --score-alpha "$SCORE_ALPHA"
    --symbolic-path-threshold "$SYMBOLIC_PATH_THRESHOLD"
  )

  printf 'Running:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  "${cmd[@]}" >"$log_file" 2>&1
}

wait_for_slot() {
  if ! wait -n; then
    overall_status=1
    if [[ "$FAIL_FAST" == "1" ]]; then
      exit 1
    fi
  fi
  running=$((running - 1))
}

for attack_mode in "${mode_list[@]}"; do
  for pixel_search in "${pixel_list[@]}"; do
    run_one "$attack_mode" "$pixel_search" &
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then
      wait_for_slot
    fi
  done
done

while (( running > 0 )); do
  wait_for_slot
done

exit "$overall_status"
