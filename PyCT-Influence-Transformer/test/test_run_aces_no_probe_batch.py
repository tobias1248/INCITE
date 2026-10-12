from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_aces_no_probe_batch.sh"


def _environment(tmp_path, **overrides):
    environment = dict(os.environ)
    environment.update({
        "CAMPAIGN_ID": "unit_no_probe", "CAMPAIGN_DIR": str(tmp_path / "campaign"),
        "MODEL_NAMES": "model_a model_b", "CONTROLS": "exposure curves",
        "FIRST_N": "3", "TIMEOUT": "1800", "SOLVER_RUN_TIMEOUT": "60",
        "BUILD_TIMEOUT": "30", "PATH_THRESHOLD": "2000", "NUM_PROCESS": "2",
        "MAX_PARALLEL": "2", "X_MIN": "-0.2", "X_MAX": "0.2",
        "SCORE_ALPHA": "0.8", "PWL_MAX_SEGMENTS": "32",
        "PWL_ERROR": str(1.0 / 255.0), "CHECK_MODELS": "0", "DRY_RUN": "1",
        "MANIFEST_PYTHON": sys.executable, "PYTHON_BIN": sys.executable,
    })
    environment.update(overrides)
    return environment


def _run(environment):
    return subprocess.run(
        ["bash", str(RUNNER)], env=environment, capture_output=True, text=True, timeout=20
    )


def test_runner_dry_run_generates_scalar_no_probe_commands_without_files(tmp_path):
    result = _run(_environment(tmp_path))
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line.removeprefix("Running:")) for line in result.stdout.splitlines() if line.startswith("Running:")]
    assert len(commands) == 4
    for command in commands:
        assert command[:3] == [sys.executable, "-m", "pyct"]
        assert "--no-global-real-probe" in command
        assert command[command.index("--attack-mode") + 1] == "global-real"
        assert command[command.index("--num-process") + 1] == "2"
        assert command[command.index("--symbolic-path-threshold") + 1] == "2000"
        assert command[command.index("--timeout") + 1] == "1800"
        assert "--ternary-simplification" not in command
        assert "--de-maxiter" not in command
    assert not (tmp_path / "campaign").exists()


def test_runner_passes_shap_flags_and_zero_alpha_in_dry_run(tmp_path):
    result = _run(_environment(tmp_path, SHAP_PRIORITY="1", SHAP_SENSITIVITY="1",
                               SHAP_OUTPUT_ROOT=str(tmp_path / "shap"), SCORE_ALPHA="0"))
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line.removeprefix("Running:")) for line in result.stdout.splitlines() if line.startswith("Running:")]
    assert len(commands) == 4
    for command in commands:
        assert "--global-real-shap-priority" in command
        assert "--global-real-shap-sensitivity" in command
        assert command[command.index("--score-alpha") + 1] == "0"
    assert not (tmp_path / "campaign").exists()


@pytest.mark.parametrize("overrides", [
    {"SHAP_SENSITIVITY": "1", "SHAP_PRIORITY": "0"},
    {"SHAP_PRIORITY": "maybe"},
    {"SHAP_SENSITIVITY": "2"},
])
def test_runner_rejects_invalid_shap_mode_before_files(tmp_path, overrides):
    result = _run(_environment(tmp_path, **overrides))
    assert result.returncode != 0
    assert not (tmp_path / "campaign").exists()


@pytest.mark.parametrize("overrides", [
    {"CONTROLS": "exposure exposure"}, {"CONTROLS": "unknown"},
    {"MODEL_NAMES": "model_a model_a"}, {"CAMPAIGN_ID": "../escape"},
    {"MAX_PARALLEL": "0"}, {"FIRST_N": "10001"}, {"X_MIN": "nan"},
    {"CONTROLS": "highlights", "X_MAX": "0.6"},
    {"CONTROLS": "saturation", "X_MIN": "-1.1"},
    {"CONTROLS": "curves", "X_MAX": "1"},
])
def test_runner_rejects_invalid_configuration_before_creating_files(tmp_path, overrides):
    result = _run(_environment(tmp_path, **overrides))
    assert result.returncode != 0
    assert not (tmp_path / "campaign").exists()


def _fake_python(tmp_path):
    fake = tmp_path / "fake_python"
    fake.write_text("#!" + sys.executable + "\n" + '''
import fcntl,json,os,sys,time
from pathlib import Path
state_path=Path(os.environ["FAKE_STATE"])
def update(start=False,finish=False):
    with state_path.open("a+") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        handle.seek(0); text=handle.read()
        state=json.loads(text) if text else {"active":0,"peak":0,"attacks":[],"reports":[]}
        if start:
            state["active"]+=1;state["peak"]=max(state["peak"],state["active"])
            state["attacks"].append({"args":sys.argv[1:],"mode":os.environ.get("PYCT_SMT_PATH_MODE"),"gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),"downloads":os.environ.get("PYCT_ALLOW_DATASET_DOWNLOAD")})
        elif finish:state["active"]-=1
        else:state["reports"].append(sys.argv[1:])
        handle.seek(0);handle.truncate();json.dump(state,handle);handle.flush()
if sys.argv[2]=="pyct":
    if os.environ.get("FAKE_RESULT_ROOT"):
        argv=sys.argv[1:]
        model=argv[argv.index("--model-name")+1]
        control=argv[argv.index("--global-shift-kind")+1]
        tag=os.environ["PYCT_ATTACK_MODE_SUFFIX"]
        (Path(os.environ["FAKE_RESULT_ROOT"])/(model+"_global-real_smtlast_"+tag+"_clip_"+control+"_xm0p2_0p2")).mkdir(parents=True)
    update(start=True);time.sleep(.15);update(finish=True)
    sys.exit(int(os.environ.get("FAKE_ATTACK_EXIT","0")))
update()
sys.exit(int(os.environ.get("FAKE_REPORT_EXIT","0")))
''')
    fake.chmod(0o755)
    return fake


def test_runner_records_manifest_limits_parallelism_and_checks_each_wave(tmp_path):
    fake = _fake_python(tmp_path)
    state_path = tmp_path / "state.json"
    result = _run(_environment(tmp_path, DRY_RUN="0", PYTHON_BIN=str(fake), FAKE_STATE=str(state_path)))
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "campaign" / "manifest.json").read_text())
    assert len(manifest["experiments"]) == 4
    assert all(item["case_ids"] == [0, 1, 2] for item in manifest["experiments"])
    assert manifest["environment"]["PYCT_SMT_PATH_MODE"] == "last"
    assert manifest["settings"]["path_threshold"] == "2000"
    state = json.loads(state_path.read_text())
    assert state["peak"] == 2
    assert state["active"] == 0
    assert len(state["attacks"]) == 4
    assert all(item["mode"] == "last" and item["gpu"] == "-1" and item["downloads"] == "0" for item in state["attacks"])
    assert sum("--check-launched" in args for args in state["reports"]) == 2
    assert "--require-complete" in state["reports"][-1]
    assert len(list((tmp_path / "campaign").glob("*.completion.json"))) == 4


def test_runner_manifest_records_shap_and_background_provenance(tmp_path):
    fake = _fake_python(tmp_path)
    result = _run(_environment(tmp_path, DRY_RUN="0", PYTHON_BIN=str(fake),
        FAKE_STATE=str(tmp_path / "state.json"), MODEL_NAMES="model_a", CONTROLS="exposure",
        SHAP_PRIORITY="1", SHAP_SENSITIVITY="1", SHAP_OUTPUT_ROOT=str(tmp_path / "cache"),
        BG_PER_CLASS="3", BG_SEED="2233", SCORE_ALPHA="0"))
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "campaign" / "manifest.json").read_text())
    settings = manifest["settings"]
    assert settings["shap_priority_enabled"] == settings["shap_sensitivity_enabled"] == "1"
    assert settings["shap_output_root"] == str(tmp_path / "cache")
    assert settings["bg_per_class"] == "3"
    assert settings["bg_seed"] == "2233"
    assert manifest["environment"]["PYCT_BG_PER_CLASS"] == "3"
    assert manifest["environment"]["PYCT_BG_SEED"] == "2233"
    assert manifest["environment"]["PYCT_SHAP_LOOKUP_MODE"] == "all"


@pytest.mark.parametrize("failure", ["FAKE_ATTACK_EXIT", "FAKE_REPORT_EXIT"])
def test_runner_stops_after_failed_wave(tmp_path, failure):
    fake = _fake_python(tmp_path)
    state_path = tmp_path / "state.json"
    result = _run(_environment(tmp_path, DRY_RUN="0", PYTHON_BIN=str(fake), FAKE_STATE=str(state_path), **{failure: "1"}))
    assert result.returncode != 0
    assert "Wave failed" in result.stderr
    state = json.loads(state_path.read_text())
    assert len(state["attacks"]) == 2
    assert not any("--require-complete" in args for args in state["reports"])


def test_runner_discovers_real_smtlast_result_directory(tmp_path):
    isolated_root = tmp_path / "checkout"
    scripts = isolated_root / "scripts"
    scripts.mkdir(parents=True)
    copied_runner = scripts / RUNNER.name
    copied_runner.write_text(RUNNER.read_text())
    fake = _fake_python(tmp_path)
    environment = _environment(tmp_path, DRY_RUN="0", PYTHON_BIN=str(fake),
                               MODEL_NAMES="model_a", CONTROLS="exposure",
                               FAKE_STATE=str(tmp_path / "state.json"),
                               FAKE_RESULT_ROOT=str(isolated_root / "exp"))
    result = subprocess.run(["bash", str(copied_runner)], env=environment,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    completion = json.loads((tmp_path / "campaign" / "model_a_exposure.completion.json").read_text())
    assert completion["matching_directories"] == 1
    assert completion["result_dir"] == str(isolated_root / "exp" / "model_a_global-real_smtlast_unit_no_probe_clip_aces-exposure_xm0p2_0p2")


def _running(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False


def test_runner_sigterm_stops_attack_and_its_descendant(tmp_path):
    marker = tmp_path / "child_pids.json"
    fake = tmp_path / "sleeping_python"
    fake.write_text("#!" + sys.executable + "\n" + '''
import json,os,signal,subprocess,sys,time
from pathlib import Path
signal.signal(signal.SIGTERM,signal.SIG_IGN)
ready=Path(os.environ["CHILD_PIDS"]+".ready")
child=subprocess.Popen([sys.executable,"-c","import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(os.environ['CHILD_PIDS']+'.ready').touch(); time.sleep(60)"])
while not ready.exists():time.sleep(.01)
Path(os.environ["CHILD_PIDS"]).write_text(json.dumps([os.getpid(),child.pid]))
time.sleep(60)
''')
    fake.chmod(0o755)
    environment = _environment(tmp_path, DRY_RUN="0", PYTHON_BIN=str(fake),
                               CHILD_PIDS=str(marker), MODEL_NAMES="model_a", CONTROLS="exposure")
    process = subprocess.Popen(["bash", str(RUNNER)], env=environment,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    child_pids = []
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "Runner never started the attack"
        child_pids = json.loads(marker.read_text())
        assert all(_running(pid) for pid in child_pids)
        process.terminate()
        _, stderr = process.communicate(timeout=5)
        assert process.returncode != 0, stderr
        deadline = time.monotonic() + 2
        while any(_running(pid) for pid in child_pids) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(_running(pid) for pid in child_pids), "Attack subprocess survived runner SIGTERM"
    finally:
        if child_pids:
            try:
                os.killpg(child_pids[0], signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
