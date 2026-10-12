#!/usr/bin/env bash
# Prepare once, then launch two serial lanes (two case workers per lane).
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/run_aces_shap_two_groups.sh"
cd "$ROOT_DIR"
MODE="${1:---launch}"
GROUP="${2:-}"
if [[ $# -gt 2 || ! "$MODE" =~ ^--(launch|prepare|group)$ ]]; then
  printf 'Usage: bash %s [--launch|--prepare|--group A|B]\n' "$SCRIPT_PATH" >&2; exit 1
fi
if [[ "$MODE" == --group && "$GROUP" != A && "$GROUP" != B ]] || [[ "$MODE" != --group && -n "$GROUP" ]]; then
  printf 'Only --group accepts a group, A or B.\n' >&2; exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
MANIFEST_PYTHON="${MANIFEST_PYTHON:-python3}"
BASE_CAMPAIGN_ID="${BASE_CAMPAIGN_ID:-aces_shap_4controls_$(date +%Y%m%d_%H%M%S)}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$ROOT_DIR/exp/$BASE_CAMPAIGN_ID}"
MODEL_NAMES="${MODEL_NAMES:-cifar10_concolic_transformer cifar10_cctlike_single_mha cifar10_cctlike_eight_mha resnet18_cifar10_clean vgg16_cifar10_clean}"
export FIRST_N="${FIRST_N:-100}" TIMEOUT="${TIMEOUT:-1800}"
export SOLVER_RUN_TIMEOUT="${SOLVER_RUN_TIMEOUT:-60}" BUILD_TIMEOUT="${BUILD_TIMEOUT:-30}"
export PATH_THRESHOLD="${PATH_THRESHOLD:-2000}" X_MIN="${X_MIN:--0.2}" X_MAX="${X_MAX:-0.2}"
export PWL_MAX_SEGMENTS="${PWL_MAX_SEGMENTS:-64}" PWL_ERROR="${PWL_ERROR:-0.00392156862745098}"
export BG_PER_CLASS="${BG_PER_CLASS:-3}" BG_SEED="${BG_SEED:-2233}"
export SCORE_ALPHA=0 SHAP_PRIORITY=1 SHAP_SENSITIVITY=1 NUM_PROCESS=2 MAX_PARALLEL=1
export MODEL_NAMES PYTHON_BIN MANIFEST_PYTHON
export CUDA_VISIBLE_DEVICES=-1 PYCT_ALLOW_DATASET_DOWNLOAD=0
export OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
DRY_RUN="${DRY_RUN:-0}" CHECK_MODELS="${CHECK_MODELS:-1}"
"$MANIFEST_PYTHON" - "$BASE_CAMPAIGN_ID" "$DRY_RUN" "$CHECK_MODELS" <<'PY'
import re,sys
name,dry,models=sys.argv[1:]
if not re.fullmatch(r'[A-Za-z0-9_-]+',name) or dry not in ('0','1') or models not in ('0','1'):
    raise SystemExit('Invalid campaign ID or dry-run/model-check flags')
PY
CAMPAIGN_ROOT="$("$MANIFEST_PYTHON" - "$CAMPAIGN_ROOT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
export BASE_CAMPAIGN_ID CAMPAIGN_ROOT
export PYCT_KERAS_HOME="$("$MANIFEST_PYTHON" - <<'PY'
import os
from pathlib import Path
print(Path(os.environ.get('PYCT_KERAS_HOME') or os.environ.get('KERAS_HOME') or Path.home()/'.keras').expanduser().resolve())
PY
)"
export KERAS_HOME="$PYCT_KERAS_HOME"
export SHAP_OUTPUT_ROOT="$CAMPAIGN_ROOT/shap_target_class"
controls_for() {
  if [[ "$1" == A ]]; then printf 'exposure highlights'; else printf 'shadows saturation'; fi
}
batch_for() {
  local lane="$1" dry="$2"
  env DRY_RUN="$dry" CHECK_MODELS="$CHECK_MODELS" CAMPAIGN_ID="${BASE_CAMPAIGN_ID}_${lane}" \
    CAMPAIGN_DIR="$CAMPAIGN_ROOT/group_$lane" CONTROLS="$(controls_for "$lane")" \
    bash "$ROOT_DIR/scripts/run_aces_no_probe_batch.sh"
}
# Validate and print both complete command sets before writing campaign files.
if [[ "$MODE" != --group ]]; then batch_for A 1; batch_for B 1; else batch_for "$GROUP" 1; fi
preflight_cmd=("$PYTHON_BIN" -m orchestration.aces_campaign_preflight
  --controls aces-exposure aces-highlights aces-shadows aces-saturation
  --first-n "$FIRST_N" --x-min "$X_MIN" --x-max "$X_MAX"
  --max-segments "$PWL_MAX_SEGMENTS" --error-tolerance "$PWL_ERROR"
  --output "$CAMPAIGN_ROOT/preflight.json")
shap_for() {
  shap_cmd=("$PYTHON_BIN" -m pyct.shap --dataset cifar10 --model-name "$1"
    --first-n "$FIRST_N" --explainer-type gradient --background-per-class "$BG_PER_CLASS"
    --background-seed "$BG_SEED" --output-root "$SHAP_OUTPUT_ROOT" --force-refresh --sleep-seconds 0)
}
read -r -a models <<< "$MODEL_NAMES"
check_preparation() {
  "$MANIFEST_PYTHON" - "$CAMPAIGN_ROOT" "$1" <<'PY'
import json,os,sys
from pathlib import Path
root=Path(sys.argv[1])
with (root/'preflight.json').open() as handle: report=json.load(handle)
expected={'controls':['aces-exposure','aces-highlights','aces-shadows','aces-saturation'],
          'first_n':int(os.environ['FIRST_N']),'x_min':float(os.environ['X_MIN']),'x_max':float(os.environ['X_MAX']),
          'max_segments':int(os.environ['PWL_MAX_SEGMENTS']),'error_tolerance':float(os.environ['PWL_ERROR'])}
count=4*expected['first_n']
if report.get('complete') is not True or report.get('settings')!=expected or report.get('checked_count')!=count or report.get('rejected_count')!=0:
    raise SystemExit('Prepared PWL checks are incomplete or incompatible')
if sys.argv[2]=='all':
    for model in os.environ['MODEL_NAMES'].split():
        for index in range(expected['first_n']):
            if not (Path(os.environ['SHAP_OUTPUT_ROOT'])/model/('shap_value_%s.json'%index)).is_file():
                raise SystemExit('Prepared SHAP cache is missing')
PY
}
if [[ "$MODE" != --group ]]; then
  printf 'Preflight:'; printf ' %q' "${preflight_cmd[@]}"; printf '\n'
  for model in "${models[@]}"; do shap_for "$model"; printf 'SHAP:'; printf ' %q' "${shap_cmd[@]}"; printf '\n'; done
fi
printf 'Groups A=Exposure/Highlights, B=Shadows/Saturation; 2 workers each; sessions=%s_A,%s_B\n' "$BASE_CAMPAIGN_ID" "$BASE_CAMPAIGN_ID"
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
[[ -x "$PYTHON_BIN" ]] || { printf 'Missing Python: %s\n' "$PYTHON_BIN" >&2; exit 1; }
if [[ "$CHECK_MODELS" == 1 ]]; then
  for model in "${models[@]}"; do [[ -f "model/$model.h5" ]] || { printf 'Missing model: %s\n' "$model" >&2; exit 1; }; done
fi
if [[ "$MODE" == --launch ]]; then
  command -v tmux >/dev/null || { printf 'tmux is required.\n' >&2; exit 1; }
  for lane in A B; do
    if tmux has-session -t "=${BASE_CAMPAIGN_ID}_${lane}" 2>/dev/null; then printf 'tmux session already exists.\n' >&2; exit 1; fi
  done
fi
if [[ "$MODE" != --group ]]; then
  if [[ -e "$CAMPAIGN_ROOT" ]]; then printf 'Campaign already exists: %s\n' "$CAMPAIGN_ROOT" >&2; exit 1; fi
  mkdir -p "$(dirname "$CAMPAIGN_ROOT")"
  mkdir "$CAMPAIGN_ROOT"
  trap 'printf "preparation_failed\n" > "$CAMPAIGN_ROOT/status.txt"' ERR
  printf 'preflight\n' > "$CAMPAIGN_ROOT/status.txt"
  "${preflight_cmd[@]}" 2>&1 | tee "$CAMPAIGN_ROOT/preflight.log"
  check_preparation pwl
  printf 'shap_refresh\n' > "$CAMPAIGN_ROOT/status.txt"
  for model in "${models[@]}"; do
    shap_for "$model"
    "${shap_cmd[@]}" 2>&1 | tee "$CAMPAIGN_ROOT/shap_${model}.log"
  done
  check_preparation all
  # Persist only experiment parameters, never a dump of the shell environment.
  "$MANIFEST_PYTHON" - "$CAMPAIGN_ROOT/ready.json" <<'PY'
import json,os,sys
keys=['BASE_CAMPAIGN_ID','MODEL_NAMES','FIRST_N','TIMEOUT','SOLVER_RUN_TIMEOUT','BUILD_TIMEOUT',
      'PATH_THRESHOLD','X_MIN','X_MAX','PWL_MAX_SEGMENTS','PWL_ERROR','BG_PER_CLASS','BG_SEED','SHAP_OUTPUT_ROOT',
      'SCORE_ALPHA','SHAP_PRIORITY','SHAP_SENSITIVITY','NUM_PROCESS','MAX_PARALLEL','PYCT_KERAS_HOME']
with open(sys.argv[1],'x') as handle: json.dump({key:os.environ[key] for key in keys},handle,indent=2)
PY
  trap - ERR
  printf 'ready\n' > "$CAMPAIGN_ROOT/status.txt"
  if [[ "$MODE" == --prepare ]]; then printf 'Preparation complete: %s\n' "$CAMPAIGN_ROOT"; exit 0; fi
fi
# The gate must exist and match this invocation before either attack lane starts.
"$MANIFEST_PYTHON" - "$CAMPAIGN_ROOT/ready.json" <<'PY'
import json,os,sys
keys=['BASE_CAMPAIGN_ID','MODEL_NAMES','FIRST_N','TIMEOUT','SOLVER_RUN_TIMEOUT','BUILD_TIMEOUT',
      'PATH_THRESHOLD','X_MIN','X_MAX','PWL_MAX_SEGMENTS','PWL_ERROR','BG_PER_CLASS','BG_SEED','SHAP_OUTPUT_ROOT',
      'SCORE_ALPHA','SHAP_PRIORITY','SHAP_SENSITIVITY','NUM_PROCESS','MAX_PARALLEL','PYCT_KERAS_HOME']
with open(sys.argv[1]) as handle: ready=json.load(handle)
if not isinstance(ready,dict) or set(ready)!=set(keys) or any(os.environ.get(key)!=value for key,value in ready.items()):
    raise SystemExit('Prepared campaign parameters differ')
PY
check_preparation all
if [[ "$MODE" == --group ]]; then
  printf 'Case logs and reports: %s/group_%s\n' "$CAMPAIGN_ROOT" "$GROUP"
  env DRY_RUN=0 CHECK_MODELS="$CHECK_MODELS" CAMPAIGN_ID="${BASE_CAMPAIGN_ID}_${GROUP}" \
    CAMPAIGN_DIR="$CAMPAIGN_ROOT/group_$GROUP" CONTROLS="$(controls_for "$GROUP")" \
    bash "$ROOT_DIR/scripts/run_aces_no_probe_batch.sh" & group_pid=$!
  cancel_group() { kill -TERM "$group_pid" 2>/dev/null || true; wait "$group_pid" 2>/dev/null || true; exit 143; }
  trap cancel_group HUP INT TERM
  code=0; wait "$group_pid" || code=$?
  trap - HUP INT TERM
  printf '%s\n' "$code" > "$CAMPAIGN_ROOT/group_${GROUP}.exit_code"
  exit "$code"
fi
group_env=()
for key in BASE_CAMPAIGN_ID CAMPAIGN_ROOT MODEL_NAMES PYTHON_BIN MANIFEST_PYTHON FIRST_N TIMEOUT SOLVER_RUN_TIMEOUT BUILD_TIMEOUT PATH_THRESHOLD X_MIN X_MAX PWL_MAX_SEGMENTS PWL_ERROR BG_PER_CLASS BG_SEED PYCT_KERAS_HOME; do
  group_env+=("$key=${!key}")
done
group_env+=("CHECK_MODELS=$CHECK_MODELS" "DRY_RUN=0")
for lane in A B; do
  printf -v launch_command '%q ' env "${group_env[@]}" bash "$SCRIPT_PATH" --group "$lane"
  if ! tmux new-session -d -s "${BASE_CAMPAIGN_ID}_${lane}" "$launch_command"; then
    if [[ "$lane" == B ]]; then tmux kill-session -t "=${BASE_CAMPAIGN_ID}_A" || true; fi
    printf 'launch_failed\n' > "$CAMPAIGN_ROOT/status.txt"; exit 1
  fi
done
printf 'groups_started\n' > "$CAMPAIGN_ROOT/status.txt"
printf 'Started tmux sessions %s_A and %s_B; root=%s\n' "$BASE_CAMPAIGN_ID" "$BASE_CAMPAIGN_ID" "$CAMPAIGN_ROOT"
