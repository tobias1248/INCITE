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
RUNNER = ROOT / 'scripts' / 'run_aces_shap_two_groups.sh'


def _environment(tmp_path, **overrides):
    environment = dict(os.environ)
    # Explicit settings prevent the invoking shell from changing this experiment contract.
    environment.update(BASE_CAMPAIGN_ID='unit_two_groups', CAMPAIGN_ROOT=str(tmp_path / 'campaign'),
                       PYTHON_BIN=sys.executable, MANIFEST_PYTHON=sys.executable,
                       MODEL_NAMES='cifar10_concolic_transformer cifar10_cctlike_single_mha '
                       'cifar10_cctlike_eight_mha resnet18_cifar10_clean vgg16_cifar10_clean',
                       FIRST_N='100', TIMEOUT='1800', SOLVER_RUN_TIMEOUT='60', BUILD_TIMEOUT='30',
                       PATH_THRESHOLD='2000', X_MIN='-0.2', X_MAX='0.2', PWL_MAX_SEGMENTS='64',
                       PWL_ERROR=str(1 / 255), BG_PER_CLASS='3', BG_SEED='2233',
                       DRY_RUN='1', CHECK_MODELS='0')
    environment.update(overrides)
    return environment


def _run(environment, *args):
    return subprocess.run(['bash', str(RUNNER), *args], env=environment, capture_output=True,
                          text=True, timeout=30)


def _commands(stdout, prefix):
    return [shlex.split(line[len(prefix):]) for line in stdout.splitlines() if line.startswith(prefix)]


def _option(command, name):
    return command[command.index(name) + 1]


def test_dry_run_has_twenty_scalar_commands_four_controls_and_no_files(tmp_path):
    result = _run(_environment(tmp_path))
    assert result.returncode == 0, result.stderr
    commands = _commands(result.stdout, 'Running:')
    assert len(commands) == 20
    expected = {'--first-n': '100', '--timeout': '1800', '--solver-run-timeout': '60',
                '--constraint-build-timeout-seconds': '30', '--symbolic-path-threshold': '2000',
                '--score-alpha': '0', '--aces-pwl-max-segments': '64', '--num-process': '2',
                '--global-x-min': '-0.2', '--global-x-max': '0.2', '--global-x-bounds-mode': 'clip'}
    for command in commands:
        assert command[:3] == [sys.executable, '-m', 'pyct']
        assert all(_option(command, name) == value for name, value in expected.items())
        assert {'--global-real-shap-priority', '--global-real-shap-sensitivity',
                '--no-global-real-probe'} <= set(command)
        assert '--de-maxiter' not in command
    assert {_option(command, '--global-shift-kind') for command in commands} == {
        'aces-exposure', 'aces-highlights', 'aces-shadows', 'aces-saturation'}
    assert 'parallel=1 x workers=2' in result.stdout
    preflight, = _commands(result.stdout, 'Preflight:')
    assert _option(preflight, '--first-n') == '100'
    assert preflight[preflight.index('--controls') + 1:preflight.index('--first-n')] == [
        'aces-exposure', 'aces-highlights', 'aces-shadows', 'aces-saturation']
    shap = _commands(result.stdout, 'SHAP:')
    assert len(shap) == 5
    assert all(_option(command, '--first-n') == '100' and '--force-refresh' in command for command in shap)
    assert not (tmp_path / 'campaign').exists()


def _fake_tools(tmp_path):
    binaries = tmp_path / 'fake tools'
    binaries.mkdir()
    python = binaries / 'python'
    python.write_text('#!' + sys.executable + '\n' + '''
import json,os,signal,sys,time
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['EVENTS'],'a') as handle:
    handle.write(json.dumps({'tool':'python','args':args,'downloads':os.environ.get('PYCT_ALLOW_DATASET_DOWNLOAD'),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'keras_home':os.environ.get('KERAS_HOME'),'pyct_keras_home':os.environ.get('PYCT_KERAS_HOME')})+'\\n')
module=args[args.index('-m')+1]
if module=='pyct' and os.environ.get('ALLOW_FAKE_ATTACK')=='1':
    Path(os.environ['ATTACK_PID']).write_text(str(os.getpid()))
    def terminated(signum, frame):
        Path(os.environ['ATTACK_TERMINATED']).write_text(str(signum))
        raise SystemExit(143)
    signal.signal(signal.SIGTERM,terminated)
    while True: time.sleep(.1)
if module not in ('orchestration.aces_campaign_preflight','pyct.shap'):
    raise SystemExit('Unexpected real attack: '+module)
if module=='orchestration.aces_campaign_preflight':
    first_n=int(args[args.index('--first-n')+1])
    settings={'controls':args[args.index('--controls')+1:args.index('--first-n')],
              'first_n':first_n,'x_min':float(args[args.index('--x-min')+1]),
              'x_max':float(args[args.index('--x-max')+1]),
              'max_segments':int(args[args.index('--max-segments')+1]),
              'error_tolerance':float(args[args.index('--error-tolerance')+1])}
    report={'complete':True,'settings':settings,'checked_count':4*first_n,'rejected_count':0}
    if os.environ.get('INVALID_PREFLIGHT')=='1': report['complete']=False
    Path(args[args.index('--output')+1]).write_text(json.dumps(report))
    raise SystemExit(int(os.environ.get('FAIL_PREFLIGHT','0')))
if os.environ.get('SKIP_SHAP_CACHE')!='1':
    root=Path(args[args.index('--output-root')+1])/args[args.index('--model-name')+1]
    root.mkdir(parents=True,exist_ok=True)
    for index in range(int(args[args.index('--first-n')+1])):
        (root/('shap_value_%s.json'%index)).write_text('{}')
raise SystemExit(int(os.environ.get('FAIL_SHAP','0')))
''')
    python.chmod(0o755)
    tmux = binaries / 'tmux'
    tmux.write_text('#!' + sys.executable + '\n' + '''
import json,os,sys
args=sys.argv[1:]
with open(os.environ['EVENTS'],'a') as handle:
    handle.write(json.dumps({'tool':'tmux','args':args})+'\\n')
if args[0]=='has-session': raise SystemExit(int(os.environ.get('SESSION_MISSING','1')))
if args[0]=='new-session' and os.environ.get('FAIL_SECOND_LAUNCH')=='1' and args[args.index('-s')+1].endswith('_B'):
    raise SystemExit(1)
raise SystemExit(0)
''')
    tmux.chmod(0o755)
    return python, binaries


def _fake_environment(tmp_path, **overrides):
    python, binaries = _fake_tools(tmp_path)
    return _environment(tmp_path, DRY_RUN='0', PYTHON_BIN=str(python),
                        EVENTS=str(tmp_path / 'events.jsonl'),
                        PATH=str(binaries) + os.pathsep + os.environ['PATH'], **overrides)


def _events(environment):
    path = Path(environment['EVENTS'])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize('failure', ['FAIL_PREFLIGHT', 'FAIL_SHAP'])
def test_failed_preparation_never_marks_ready_or_starts_groups(tmp_path, failure):
    environment = _fake_environment(tmp_path, **{failure: '1'})
    result = _run(environment)
    assert result.returncode != 0
    root = Path(environment['CAMPAIGN_ROOT'])
    assert not (root / 'ready.json').exists()
    assert not (root / 'group_A').exists()
    assert not (root / 'group_B').exists()
    assert (root / 'status.txt').read_text().strip() == 'preparation_failed'
    events = _events(environment)
    assert not any(event['tool'] == 'tmux' and event['args'][0] == 'new-session' for event in events)
    python = [event for event in events if event['tool'] == 'python']
    assert len(python) == (1 if failure == 'FAIL_PREFLIGHT' else 2)


def test_launch_prepares_once_then_starts_two_tmux_sessions_with_quoted_absolute_paths(tmp_path):
    # These characters must survive shell quoting without creating a nested command.
    root = tmp_path / "campaign with spaces ' and $(touch injected)"
    environment = _fake_environment(tmp_path, CAMPAIGN_ROOT=str(root))
    result = _run(environment)
    assert result.returncode == 0, result.stderr
    events = _events(environment)
    preparation = [event for event in events if event['tool'] == 'python']
    assert [event['args'][1] for event in preparation] == [
        'orchestration.aces_campaign_preflight'] + ['pyct.shap'] * 5
    assert all(event['downloads'] == '0' and event['gpu'] == '-1' for event in preparation)
    launches = [event for event in events if event['tool'] == 'tmux' and event['args'][0] == 'new-session']
    assert len(launches) == 2
    assert events.index(launches[0]) > events.index(preparation[-1])
    for lane, event in zip(['A', 'B'], launches):
        assert _option(event['args'], '-s') == 'unit_two_groups_' + lane
        command = shlex.split(event['args'][-1])
        assert command[0] == 'env'
        assert 'CAMPAIGN_ROOT=' + str(root.resolve()) in command
        assert 'PYTHON_BIN=' + environment['PYTHON_BIN'] in command
        assert command[-4:] == ['bash', str(RUNNER), '--group', lane]
    ready = json.loads((root / 'ready.json').read_text())
    assert ready['NUM_PROCESS'] == '2' and ready['MAX_PARALLEL'] == '1'
    assert ready['SHAP_OUTPUT_ROOT'] == str(root / 'shap_target_class')
    assert (root / 'status.txt').read_text().strip() == 'groups_started'
    assert not (ROOT / 'injected').exists()
    assert not (root / 'group_A').exists()  # fake tmux did not invoke the real attack.


def test_group_parameter_mismatch_rejected_before_attack(tmp_path):
    environment = _fake_environment(tmp_path)
    result = _run(environment, '--prepare')
    assert result.returncode == 0, result.stderr
    before = _events(environment)
    environment['TIMEOUT'] = '900'
    result = _run(environment, '--group', 'A')
    assert result.returncode != 0
    assert 'Prepared campaign parameters differ' in result.stderr
    assert _events(environment) == before
    assert not (Path(environment['CAMPAIGN_ROOT']) / 'group_A').exists()


def test_session_collision_rejected_before_any_campaign_writes(tmp_path):
    environment = _fake_environment(tmp_path, SESSION_MISSING='0')
    result = _run(environment)
    assert result.returncode != 0
    assert 'session already exists' in result.stderr
    assert not Path(environment['CAMPAIGN_ROOT']).exists()
    assert all(event['tool'] == 'tmux' and event['args'][0] == 'has-session'
               for event in _events(environment))


def test_failed_second_launch_cancels_first_session(tmp_path):
    environment = _fake_environment(tmp_path, FAIL_SECOND_LAUNCH='1')
    result = _run(environment)
    assert result.returncode != 0
    assert any(event['tool'] == 'tmux' and event['args'] == ['kill-session', '-t', '=unit_two_groups_A']
               for event in _events(environment))
    assert (Path(environment['CAMPAIGN_ROOT']) / 'status.txt').read_text().strip() == 'launch_failed'


def test_existing_preparation_root_is_not_overwritten(tmp_path):
    environment = _fake_environment(tmp_path)
    root = Path(environment['CAMPAIGN_ROOT'])
    root.mkdir()
    marker = root / 'preserved'
    marker.write_text('keep')
    result = _run(environment, '--prepare')
    assert result.returncode != 0
    assert marker.read_text() == 'keep'
    assert not (root / 'status.txt').exists()
    assert not _events(environment)


@pytest.mark.parametrize('failure', ['SKIP_SHAP_CACHE', 'INVALID_PREFLIGHT'])
def test_false_successful_preparation_rejects_incomplete_artifacts_before_ready(tmp_path, failure):
    environment = _fake_environment(tmp_path, **{failure: '1'})
    result = _run(environment, '--prepare')
    assert result.returncode != 0
    root = Path(environment['CAMPAIGN_ROOT'])
    assert not (root / 'ready.json').exists()
    assert not any(event['tool'] == 'tmux' and event['args'][0] == 'new-session'
                   for event in _events(environment))


@pytest.mark.parametrize('tamper', ['ready_keys', 'preflight_count', 'missing_cache'])
def test_group_gate_revalidates_all_prepared_artifacts_before_attack(tmp_path, tamper):
    environment = _fake_environment(tmp_path)
    result = _run(environment, '--prepare')
    assert result.returncode == 0, result.stderr
    root = Path(environment['CAMPAIGN_ROOT'])
    if tamper == 'ready_keys':
        ready = json.loads((root / 'ready.json').read_text())
        ready.pop('TIMEOUT')
        (root / 'ready.json').write_text(json.dumps(ready))
    elif tamper == 'preflight_count':
        preflight = json.loads((root / 'preflight.json').read_text())
        preflight['checked_count'] -= 1
        (root / 'preflight.json').write_text(json.dumps(preflight))
    else:
        (root / 'shap_target_class' / 'vgg16_cifar10_clean' / 'shap_value_99.json').unlink()
    before = _events(environment)
    result = _run(environment, '--group', 'B')
    assert result.returncode != 0
    assert not (root / 'group_B').exists()
    assert _events(environment) == before


def test_group_cancellation_reaches_batch_and_its_attack_process(tmp_path):
    environment = _fake_environment(tmp_path, FIRST_N='1', MODEL_NAMES='model_a',
                                    ALLOW_FAKE_ATTACK='1', ATTACK_PID=str(tmp_path / 'attack.pid'),
                                    ATTACK_TERMINATED=str(tmp_path / 'attack.terminated'))
    result = _run(environment, '--prepare')
    assert result.returncode == 0, result.stderr
    process = subprocess.Popen(['bash', str(RUNNER), '--group', 'A'], env=environment,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    attack_pid = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if Path(environment['ATTACK_PID']).exists():
                attack_pid = int(Path(environment['ATTACK_PID']).read_text())
                break
            if process.poll() is not None:
                pytest.fail('group exited before attack: ' + process.stderr.read())
            time.sleep(.05)
        assert attack_pid is not None, 'fake attack did not start'
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 143, stderr
        assert Path(environment['ATTACK_TERMINATED']).read_text() == str(int(signal.SIGTERM))
        with pytest.raises(ProcessLookupError):
            os.kill(attack_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if attack_pid is not None:
            try:
                os.kill(attack_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_keras_home_fallback_is_shared_with_preparation_tmux_and_ready_gate(tmp_path):
    cache = tmp_path / 'keras cache with spaces'
    environment = _fake_environment(tmp_path, KERAS_HOME=str(cache))
    environment.pop('PYCT_KERAS_HOME', None)
    result = _run(environment)
    assert result.returncode == 0, result.stderr
    events = _events(environment)
    preparation = [event for event in events if event['tool'] == 'python']
    assert all(event['keras_home'] == event['pyct_keras_home'] == str(cache.resolve())
               for event in preparation)
    launches = [event for event in events if event['tool'] == 'tmux' and event['args'][0] == 'new-session']
    assert all('PYCT_KERAS_HOME=' + str(cache.resolve()) in shlex.split(event['args'][-1])
               for event in launches)
    ready = json.loads((Path(environment['CAMPAIGN_ROOT']) / 'ready.json').read_text())
    assert ready['PYCT_KERAS_HOME'] == str(cache.resolve())
    environment['KERAS_HOME'] = str(tmp_path / 'other cache')
    before = _events(environment)
    result = _run(environment, '--group', 'A')
    assert result.returncode != 0
    assert 'Prepared campaign parameters differ' in result.stderr
    assert _events(environment) == before
