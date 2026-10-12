#!/usr/bin/env bash
# Scalar photo controls only. Run in a user-created tmux session on 530lab.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
MANIFEST_PYTHON="${MANIFEST_PYTHON:-python3}"
MODEL_NAMES="${MODEL_NAMES:-cifar10_concolic_transformer cifar10_cctlike_single_mha cifar10_cctlike_eight_mha resnet18_cifar10_clean vgg16_cifar10_clean}"
CONTROLS="${CONTROLS:-exposure contrast highlights shadows saturation vibrance curves}"
FIRST_N="${FIRST_N:-100}"
TIMEOUT="${TIMEOUT:-1800}"
SOLVER_RUN_TIMEOUT="${SOLVER_RUN_TIMEOUT:-60}"
BUILD_TIMEOUT="${BUILD_TIMEOUT:-30}"
PATH_THRESHOLD="${PATH_THRESHOLD:-2000}"
NUM_PROCESS="${NUM_PROCESS:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
X_MIN="${X_MIN:--0.2}"
X_MAX="${X_MAX:-0.2}"
SCORE_ALPHA="${SCORE_ALPHA:-0.8}"
PWL_MAX_SEGMENTS="${PWL_MAX_SEGMENTS:-32}"
PWL_ERROR="${PWL_ERROR:-0.00392156862745098}"
SHAP_PRIORITY="${SHAP_PRIORITY:-0}"
SHAP_SENSITIVITY="${SHAP_SENSITIVITY:-0}"
SHAP_OUTPUT_ROOT="${SHAP_OUTPUT_ROOT:-shap_target_class}"
BG_PER_CLASS="${BG_PER_CLASS:-3}"
BG_SEED="${BG_SEED:-2233}"
export SHAP_PRIORITY SHAP_SENSITIVITY SHAP_OUTPUT_ROOT
export PYCT_BG_PER_CLASS="$BG_PER_CLASS" PYCT_BG_SEED="$BG_SEED" PYCT_SHAP_LOOKUP_MODE=all
DRY_RUN="${DRY_RUN:-0}"
CHECK_MODELS="${CHECK_MODELS:-1}"
CAMPAIGN_ID="${CAMPAIGN_ID:-aces_noprobe_$(date +%Y%m%d_%H%M%S)}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-$ROOT_DIR/exp/$CAMPAIGN_ID}"
export CUDA_VISIBLE_DEVICES=-1
export PYCT_SMT_PATH_MODE=last
export PYCT_ATTACK_MODE_SUFFIX="$CAMPAIGN_ID"
export PYCT_ALLOW_DATASET_DOWNLOAD=0
export OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

# Validate before any directory creation or attack subprocess.
"$MANIFEST_PYTHON" - "$SHAP_PRIORITY" "$SHAP_SENSITIVITY" "$BG_PER_CLASS" "$BG_SEED" <<'PY'
import sys
priority,sensitivity,count,seed=sys.argv[1:]
if priority not in ('0','1') or sensitivity not in ('0','1'): raise SystemExit('Invalid SHAP flags')
if sensitivity=='1' and priority!='1': raise SystemExit('SHAP sensitivity requires SHAP priority')
if not count.isdigit() or int(count)<1 or not seed.isdigit(): raise SystemExit('Invalid SHAP background settings')
PY
"$MANIFEST_PYTHON" - "$CAMPAIGN_ID" "$MODEL_NAMES" "$CONTROLS" "$FIRST_N" "$TIMEOUT" "$SOLVER_RUN_TIMEOUT" "$BUILD_TIMEOUT" "$PATH_THRESHOLD" "$NUM_PROCESS" "$MAX_PARALLEL" "$X_MIN" "$X_MAX" "$SCORE_ALPHA" "$PWL_MAX_SEGMENTS" "$PWL_ERROR" "$DRY_RUN" "$CHECK_MODELS" <<'PY'
import math,re,sys
v=sys.argv[1:]
if not re.fullmatch(r'[A-Za-z0-9_-]+',v[0]): raise SystemExit('Invalid campaign ID')
models=v[1].split(); controls=v[2].split()
if not models or len(set(models))!=len(models) or any(not re.fullmatch(r'[A-Za-z0-9_-]+',m) for m in models): raise SystemExit('Invalid/duplicate models')
allowed={'exposure','contrast','highlights','shadows','saturation','vibrance','curves'}
if not controls or len(set(controls))!=len(controls) or not set(controls)<=allowed: raise SystemExit('Invalid/duplicate controls')
for s in v[3:10]+[v[13]]:
    if not s.isdigit() or int(s)<1: raise SystemExit('Budgets/counts must be positive integers')
lo,hi,alpha,err=map(float,[v[10],v[11],v[12],v[14]])
if not all(math.isfinite(x) for x in [lo,hi,alpha,err]) or not lo<=0<=hi or lo>=hi or not 0<=alpha<=1 or err<=0: raise SystemExit('Invalid range/alpha/error')
if int(v[3])>10000: raise SystemExit('FIRST_N exceeds CIFAR10 test size')
if set(controls)&{'highlights','shadows'} and (lo<-.5 or hi>.5): raise SystemExit('Selective tone range exceeds [-0.5,0.5]')
if set(controls)&{'saturation','vibrance'} and lo<-1: raise SystemExit('Chroma range starts below -1')
if 'curves' in controls and (lo<=-1 or hi>=1): raise SystemExit('Curves range must be inside (-1,1)')
if v[15] not in ('0','1') or v[16] not in ('0','1'): raise SystemExit('DRY_RUN/CHECK_MODELS must be 0 or 1')
PY
read -r -a models <<< "$MODEL_NAMES"
read -r -a controls <<< "$CONTROLS"
command_for() {
  cmd=("$PYTHON_BIN" -m pyct --dataset cifar10 --model-name "$1"
    --attack-mode global-real --global-shift-kind "aces-$2" --no-global-real-probe
    --global-x-min "$X_MIN" --global-x-max "$X_MAX" --global-x-bounds-mode clip
    --first-n "$FIRST_N" --pixel-search 1 --num-process "$NUM_PROCESS"
    --timeout "$TIMEOUT" --solver-run-timeout "$SOLVER_RUN_TIMEOUT"
    --constraint-build-timeout-seconds "$BUILD_TIMEOUT" --symbolic-path-threshold "$PATH_THRESHOLD"
    --score-alpha "$SCORE_ALPHA" --aces-pwl-max-segments "$PWL_MAX_SEGMENTS"
    --aces-pwl-error-tolerance "$PWL_ERROR" --random-seed 2024 --log-level INFO
    --shap-output-root "$SHAP_OUTPUT_ROOT")
  if [[ "$SHAP_PRIORITY" == 1 ]]; then cmd+=(--global-real-shap-priority); fi
  if [[ "$SHAP_SENSITIVITY" == 1 ]]; then cmd+=(--global-real-shap-sensitivity); fi
}
printf 'campaign=%s; models=%s; controls=%s; cases=%s; parallel=%s x workers=%s; SMT=last; Probe=off; CPU-only\n' "$CAMPAIGN_ID" "$MODEL_NAMES" "$CONTROLS" "$FIRST_N" "$MAX_PARALLEL" "$NUM_PROCESS"
for model in "${models[@]}"; do
  for control in "${controls[@]}"; do
    command_for "$model" "$control"
    printf 'Running:'; printf ' %q' "${cmd[@]}"; printf '\n'
  done
done
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
if [[ ! -x "$PYTHON_BIN" ]]; then printf 'Missing Python: %s\n' "$PYTHON_BIN" >&2; exit 1; fi
if [[ "$CHECK_MODELS" == 1 ]]; then
  for model in "${models[@]}"; do
    if [[ ! -f "model/$model.h5" ]]; then printf 'Missing model: %s\n' "$model" >&2; exit 1; fi
  done
fi
if [[ -e "$CAMPAIGN_DIR" ]]; then printf 'Campaign already exists: %s\n' "$CAMPAIGN_DIR" >&2; exit 1; fi
mkdir -p "$CAMPAIGN_DIR/logs"
CAMPAIGN_DIR="$(cd "$CAMPAIGN_DIR" && pwd)"
"$MANIFEST_PYTHON" - "$CAMPAIGN_DIR" "$ROOT_DIR" "$MODEL_NAMES" "$CONTROLS" "$FIRST_N" "$TIMEOUT" "$SOLVER_RUN_TIMEOUT" "$BUILD_TIMEOUT" "$PATH_THRESHOLD" "$NUM_PROCESS" "$MAX_PARALLEL" "$X_MIN" "$X_MAX" "$SCORE_ALPHA" "$PWL_MAX_SEGMENTS" "$PWL_ERROR" "$PYTHON_BIN" <<'PY'
import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
v=sys.argv[1:];root=Path(v[1]); hashes={}
for model in v[2].split():
    p=root/'model'/(model+'.h5')
    hashes[model]=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
git=lambda *a: subprocess.run(['git','-C',str(root),*a],text=True,capture_output=True).stdout.strip()
commit=git('rev-parse','HEAD') or None
source_paths=list(root.glob('*.py'))
for package in ['pyct','orchestration','tasks','datasets','engine','modeling','explainability','reporting','libct','dnnct']:
    source_paths.extend((root/package).rglob('*.py'))
source_paths.extend([root/'scripts/run_aces_no_probe_batch.sh',root/'docs/aces-like-controls.md'])
source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths) if p.is_file()}
manifest={'schema_version':'aces-no-probe-campaign-v1','created_at':datetime.now(timezone.utc).isoformat(),
 'source_commit':commit,'source_dirty':bool(git('status','--porcelain')) if commit else None,'source_files_sha256':source_hashes,
 'python':v[16],'host':platform.node(),'model_sha256':hashes,
 'settings':dict(zip(['first_n','timeout','solver_timeout','build_timeout','path_threshold','num_process','max_parallel','x_min','x_max','score_alpha','pwl_max_segments','pwl_error'],v[4:16])),
'environment':{k:os.environ.get(k) for k in ['PYCT_SMT_PATH_MODE','PYCT_ATTACK_MODE_SUFFIX','CUDA_VISIBLE_DEVICES','PYCT_ALLOW_DATASET_DOWNLOAD','OMP_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','PYCT_BG_PER_CLASS','PYCT_BG_SEED','PYCT_SHAP_LOOKUP_MODE']},
 'experiments':[{'model':m,'control':'aces-'+c,'case_ids':list(range(int(v[4])))} for m in v[2].split() for c in v[3].split()]}
manifest['settings'].update(shap_priority_enabled=os.environ['SHAP_PRIORITY'],
 shap_sensitivity_enabled=os.environ['SHAP_SENSITIVITY'],shap_output_root=str(Path(os.environ['SHAP_OUTPUT_ROOT']).resolve()),
 bg_per_class=os.environ['PYCT_BG_PER_CLASS'],bg_seed=os.environ['PYCT_BG_SEED'])
Path(v[0],'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
run_one() {
  local model="$1" control="$2" code=0
  command_for "$model" "$control"
  # Separate session makes cancellation terminate the CLI and its case workers.
  setsid "${cmd[@]}" > "$CAMPAIGN_DIR/logs/${model}_${control}.log" 2>&1 &
  local attack_pid=$!
  printf '%s\n' "$attack_pid" > "$CAMPAIGN_DIR/logs/${model}_${control}.pid"
  wait "$attack_pid" || code=$?
  rm -f "$CAMPAIGN_DIR/logs/${model}_${control}.pid"
  "$MANIFEST_PYTHON" - "$CAMPAIGN_DIR" "$ROOT_DIR" "$model" "$control" "$code" "$CAMPAIGN_ID" <<'PY'
import json,sys
from pathlib import Path
campaign,root,model,control,code,tag=sys.argv[1:]
dirs=list((Path(root)/'exp').glob(model+'_global-real_smtlast_'+tag+'_clip_aces-'+control+'_*'))
record={'model':model,'control':'aces-'+control,'exit_code':int(code),
        'result_dir':str(dirs[0]) if len(dirs)==1 else None,'matching_directories':len(dirs)}
Path(campaign,model+'_'+control+'.completion.json').write_text(json.dumps(record,indent=2)+'\n')
PY
  return "$code"
}
pids=()
cleanup() {
  local pidfile attack_pid
  local groups=()
  for pidfile in "$CAMPAIGN_DIR"/logs/*.pid; do
    [[ -f "$pidfile" ]] || continue
    read -r attack_pid < "$pidfile"
    groups+=("$attack_pid")
    kill -TERM -- "-$attack_pid" 2>/dev/null || true
  done
  # Native work can delay SIGTERM: enforce a bounded cancellation grace period.
  if (( ${#groups[@]} )); then
    sleep 2
    for attack_pid in "${groups[@]}"; do kill -KILL -- "-$attack_pid" 2>/dev/null || true; done
  fi
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_wave() {
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  pids=()
  # CLI exit 0 alone does not prove all cases have terminal, valid stats.
  "$PYTHON_BIN" -m pyct.aces_stats --path "$CAMPAIGN_DIR" --output-dir "$CAMPAIGN_DIR/report" --check-launched || failed=1
  if (( failed )); then printf 'Wave failed; inspect report/errors and logs.\n' >&2; exit 1; fi
}
for model in "${models[@]}"; do
  for control in "${controls[@]}"; do
    run_one "$model" "$control" & pids+=("$!")
    if (( ${#pids[@]} >= MAX_PARALLEL )); then wait_wave; fi
  done
done
if (( ${#pids[@]} )); then wait_wave; fi
"$PYTHON_BIN" -m pyct.aces_stats --path "$CAMPAIGN_DIR" --output-dir "$CAMPAIGN_DIR/report" --require-complete
trap - INT TERM EXIT
