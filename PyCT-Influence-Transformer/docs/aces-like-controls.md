# Scalar ACES-like controls: PyCT without Probe

These are deterministic photo controls inspired by perceptual colour editing,
not the full ACES rendering transform. Each run searches one scalar `X` from
the original image. DE and reference Probe are disabled. Legacy
`aces-brightness` and the brightness/contrast hybrid pipeline remain available.

## Exact control definitions

All controls preserve the source image exactly at `X=0`. The seven primary
controls use version `scalar-photo-controls-v1`.

| CLI shift kind | Exact transform | Unit |
| --- | --- | --- |
| `aces-exposure` | Decode sRGB, multiply linear RGB by `2**X`, clip channels to `[0,1]`, encode sRGB | EV |
| `aces-contrast` | Multiply the OKLab lightness logit by `2**X` | dimensionless |
| `aces-highlights` | Add `2*X*smoothstep(0.5,0.9,L)` to the lightness logit | dimensionless |
| `aces-shadows` | Add `2*X*(1-smoothstep(0.1,0.5,L))` to the lightness logit | dimensionless |
| `aces-saturation` | Multiply OKLCh chroma by `1+X` | dimensionless |
| `aces-vibrance` | Multiply chroma by `1+X*(1-clip(C/0.2,0,1))` | dimensionless |
| `aces-curves` | Linearly interpolate lightness anchors `[0,.25,.5,.75,1]` to those anchors plus `X*[0,.25,.5,.25,0]` | dimensionless |

Tone/chroma controls use the existing OKLCh gamut mapper. Exposure instead
clips linear RGB channels. Selective tone ranges are restricted to `[-.5,.5]`,
chroma controls require `X>=-1`, and curves require `-1<X<1`.

## Search and validation

Symbolic forward uses an adaptive PWL approximation of the exact transformed
image. The default limit is 32 segments with sampled maximum absolute RGB
error tolerance `1/255`. This sampled tolerance is not a uniform error proof.
The runtime also checks the approximation error at a candidate and rejects
candidates outside the tolerance.

Reference prediction and saved images now use the exact transform for
standalone ACES-like runs as well as hybrids. Previously standalone paths
could use PWL for reference prediction and artifacts; those historical results
need exact replay before being compared as validated attacks.

With `--no-global-real-probe`, only the SMT candidate `X` is passed through the
exact transform and Keras model. A PWL prediction flip that does not flip the
exact prediction is rejected. No neighbouring `X` values are searched.

## Planned campaign defaults

The batch runner prepares five CIFAR10 models times seven controls times
100 cases (indices 0–99): 3500 case executions. The models are
`cifar10_concolic_transformer`, `cifar10_cctlike_single_mha`,
`cifar10_cctlike_eight_mha`, `resnet18_cifar10_clean`, and
`vgg16_cifar10_clean`.

| Setting | Default |
| --- | --- |
| Attack | standalone `global-real`, no DE, no Probe |
| Range | `[-0.2,0.2]`, clip bounds |
| PyCT timeout | 1800 seconds per case |
| Solver timeout | 60 seconds per solver call |
| Constraint build timeout | 30 seconds |
| Symbolic path threshold | 2000 |
| SMT path mode | `last` (bounds retained) |
| Priority score alpha | 0.8 |
| Pixel search | 1 |
| PWL | 32 segments, sampled tolerance `1/255` |
| Concurrent groups | 2 |
| Workers per group | 2 |
| CPU threads per worker | OMP/TF intra/inter: 1 |
| Ternary simplification / GPU | off / off |
| Random seed | 2024 |

The runner fixes these environment settings and records parameters, model
hashes, host, source commit, and whether the checkout is dirty in a manifest.
Two groups run as a wave; the next wave starts after both finish and pass the
result audit. Solver subprocesses and the parent process are additional to
the four concurrent case workers. PWL preprocessing/model startup are outside
the per-case exploration timeout, so this is not a total campaign wall-clock
limit. Cancellation signals are sent to each CLI process group and its workers.

Preview commands without creating campaign files:

```bash
DRY_RUN=1 bash scripts/run_aces_no_probe_batch.sh
```

After reviewing parameters and approving the campaign, use a committed,
pushed checkout and pull it on 530lab. Launch from that repository inside
tmux (all five `.h5` models and the cached CIFAR10 dataset must already exist):

```bash
tmux new-session -s aces-no-probe
CAMPAIGN_ID=aces_noprobe_reviewed bash scripts/run_aces_no_probe_batch.sh
```

Optional environment overrides include `MODEL_NAMES`, `CONTROLS`, `FIRST_N`,
`TIMEOUT`, `SOLVER_RUN_TIMEOUT`, `BUILD_TIMEOUT`, `PATH_THRESHOLD`, `X_MIN`,
`X_MAX`, `NUM_PROCESS`, and `MAX_PARALLEL`. The runner refuses duplicate names,
invalid ranges, and reuse of an existing campaign directory. This change does
not itself launch the campaign.

## Reporting

```bash
.venv/bin/python -m pyct.aces_stats \
  --path exp/aces_noprobe_reviewed \
  --output-dir exp/aces_noprobe_reviewed/report --require-complete
```

Outputs are `report.json` and `cases.csv`. The report verifies case indices
against the cached CIFAR10 test images, explicit No Probe/exact metadata,
absence of probe calls, model/control consistency, manifest parameter
consistency, duplicate cases, missing cases, and terminal statuses. It does
not independently rerun Keras prediction; recorded predictions come from the
exact runtime reference. `--dataset-cache` can select a local `test_batch`
file; no downloads occur.

Primary ASR is **originally correct → wrong / originally correct** using fixed
CIFAR10 ground truth. Prediction flips are secondary: wrong → correct and
wrong → another wrong class do not count as newly induced errors. Observed
partial ASR is labelled separately; complete ASR is withheld when coverage or
original labels are unknown. An initial-forward timeout may lack an original
prediction and therefore cannot supply the full denominator.

`success`, `timeout`, `exhausted`, and `incomplete` are valid final case
outcomes; a CLI error, missing stats/artifacts, or failed audit stops the
runner. Campaign manifests define expected coverage. A raw result directory
can be audited but cannot establish complete coverage without expected IDs.
Solver SAT/UNSAT/constraint totals are cumulative across recorded iterations;
an interrupted iteration can be omitted by the existing recorder.

Experiment artifacts belong under ignored `exp/` and are not committed.

## Target SHAP and scalar-X sensitivity priority

Standalone ACES runs can explicitly enable `--global-real-shap-priority` and
`--global-real-shap-sensitivity`. Both default to off. Sensitivity requires
SHAP priority; these flags are restricted to standalone ACES photo controls.
Priority requires an existing compatible all-layer target-class cache and
fails on missing/incompatible/input-only caches. The attack never recomputes
SHAP. Model, input, background and case fingerprints must match.

Refresh a separate cache for each model before running the reviewed campaign:

```bash
.venv/bin/python -m pyct.shap --dataset cifar10 \
  --model-name cifar10_concolic_transformer --first-n 100 \
  --explainer-type gradient --background-per-class 3 --background-seed 2233 \
  --output-root shap_target_class --force-refresh
```

The target is the original image's predicted class, fixed across controls.
Sequential models use layerwise Gradient SHAP. Functional/DAG models use input
Gradient SHAP and internal `gradient * (activation - background_mean)` influence.
The sensitivity multiplier changes constraint priority, not these attributions.

For a recorded comparison, define `g(X) = lhs(X) - rhs(X)`. Evaluate its existing
PWL expression at the current candidate X and two clipped neighbors separated
by 1% of the allowed range. Let `slope` be the largest absolute one-sided slope
and `distance` the largest displacement from X to either bound:

```text
S = min(1, distance * slope / (abs(g(X)) + 1e-8))
influence = abs(target_SHAP) * S
score = (1-alpha) * log10(influence + score_epsilon)
        - alpha * log10(path_length + 1)
```

With `alpha=0`, depth does not affect score. This is a branch reachability
heuristic; it does not guarantee SAT or an adversarial example. No additional
neural-network forward, reference probe, or attack candidate is produced.
Expressions are evaluated iteratively without mutation, with shared per-X
memoization. Each forward has a cumulative 200,000-node / 2-second estimator
budget. Unsupported expressions or budget exhaustion use raw target SHAP;
such branches can outrank branches whose sensitivity multiplier is below one.
The estimator resets at each new candidate X.

Preview the approved six-control settings (repeat cache refresh for all five
models; this preview does not launch the campaign):

```bash
DRY_RUN=1 CONTROLS='exposure highlights shadows saturation vibrance curves' \
  FIRST_N=100 TIMEOUT=1800 PATH_THRESHOLD=2000 SCORE_ALPHA=0 \
  PWL_MAX_SEGMENTS=64 SHAP_PRIORITY=1 SHAP_SENSITIVITY=1 \
  SHAP_OUTPUT_ROOT=shap_target_class bash scripts/run_aces_no_probe_batch.sh
```

The runner records flags, background settings and an absolute cache root.
Results retain the cache SHA256 and metadata; the report checks model/input/
background/case identity and original target when available. It reports SHAP
lookup/nonzero counts, sensitivity coverage, fallback reasons, evaluated nodes
and estimator seconds. Counters are cumulative over returned child envelopes;
a hard-killed forward may not return its counters. Nonzero lookup counts should
be checked before attributing an experiment effect to SHAP priority.

## Four controls in two tmux groups

`scripts/run_aces_shap_two_groups.sh` prepares one shared campaign and launches
two independent tmux sessions:

| Group | Controls, run sequentially for each model | Case workers |
| --- | --- | --- |
| A | Exposure, Highlights | 2 |
| B | Shadows, Saturation | 2 |

Both groups cover the five CIFAR10 models listed above, indices 0–99. This
gives 20 model/control experiments and 2,000 case executions. Each group has
`MAX_PARALLEL=1`, so only one attack CLI is active per group. There are at most
four concurrent case workers across both groups, plus their parents/solvers.

Defaults are timeout 1800s, solver 60s, build 30s, threshold 2000, X in
`[-0.2,0.2]`, PWL cap 64 and tolerance `1/255`. The wrapper fixes alpha 0
and target SHAP priority on. Sensitivity defaults to on and can be disabled
with `SHAP_SENSITIVITY=0`. DE, Probe, ternary simplification,
GPU and dataset downloads are off. CPU thread limits remain one per worker.
The sensitivity fallback and cumulative budgets described above still apply.

Preparation runs these stages before either attack group starts:

1. Check all four controls on all 100 cached test images using the runtime PWL
   builder (400 checks, shared across models). Save every rejection and stop
   if any check fails; never change the segment cap or tolerance automatically.
2. Refresh target SHAP for 100 images on each of the five models, sequentially,
   with three background samples per class and background seed 2233.
3. Save the prepared parameter gate, then create group A/B tmux sessions with
   those exact settings. A group refuses to run with changed prepared settings.

### SHAP-only comparison with the same attribution caches

Use `SHAP_SENSITIVITY=0` for raw target SHAP ordering; alpha remains zero, so
depth does not affect priority. All model/control/case/search settings above
remain the same. This comparison measures the effect of sensitivity weighting
with the same SHAP attributions.

`SHAP_CACHE_MODE=refresh` is the default and recomputes SHAP as before.
`SHAP_CACHE_MODE=reuse` requires `SHAP_REUSE_ROOT`, pointing at the old
campaign's `shap_target_class/` directory. Reuse never runs `pyct.shap` and
never silently regenerates missing or incompatible caches. It copies the
requested files into the new campaign's own `shap_target_class/` directory,
without symlinks or hard links. Source and destination trees must not overlap.
The old campaign's caches are only read.

Both modes validate the local model hash, float32 input, balanced background,
case index, explainer, background settings and internal-layer values. This
offline check does not load the neural network; the attack runtime/report
still verifies the target against the original reference prediction. A
failure stops preparation; partial copies/logs remain for inspection and no
successful receipt or ready gate is published. Use a new campaign ID/root
after a failed preparation.

`shap_cache_manifest.json` records the source path and every cache SHA256.
The ready gate includes its digest, the cache mode, source path and sensitivity
flag. Before each group starts, the runner verifies those parameters, current
model/dataset identity and all snapshot hashes. The source directory need not
remain available after preparation. Do not edit the snapshot during a run;
the existing case/report audit also checks recorded cache provenance.

Preview the SHAP-only campaign without writing files:

```bash
DRY_RUN=1 CHECK_MODELS=0 BASE_CAMPAIGN_ID=aces_shap_only_100c_reviewed \
  SHAP_SENSITIVITY=0 SHAP_CACHE_MODE=reuse \
  SHAP_REUSE_ROOT=/absolute/path/to/baseline/shap_target_class \
  bash scripts/run_aces_shap_two_groups.sh
```

### Isolated checkout on 530lab

After separately approving commits and pushing locally, fetch on 530lab and
create a new worktree at the exact approved commit. The repository root is
`INCITE`; the project is its `PyCT-Influence-Transformer/` subdirectory.
The active baseline worktree must keep its current code and environment.

```bash
# Run on 530lab; replace these two values with the reviewed SHA and new name.
APPROVED_SHA=REPLACE_WITH_APPROVED_COMMIT_SHA
EXP_WORKTREE=/home/soslab-tobias/INCITE-worktrees/aces-shap-only-100-20261018
git -C /home/soslab-tobias/INCITE fetch origin
git -C /home/soslab-tobias/INCITE worktree add --detach "$EXP_WORKTREE" "$APPROVED_SHA"
cd "$EXP_WORKTREE/PyCT-Influence-Transformer"
```

Provide the five existing model files in this checkout and verify their hashes
against the baseline manifest. Tracked models are already checked out; link
only missing model assets from the existing model cache. Use an explicit
`PYTHON_BIN` for the existing venv and do not install/update dependencies while
either campaign is running. Record the Python/package versions with the
experiment. Worktrees isolate tracked code; they do not isolate shared venvs,
model assets, dataset files or CPU resources.

Start a preparation tmux session in the new project directory, then export:

```bash
export PYTHON_BIN=/home/soslab-tobias/INCITE/PyCT-Influence-Transformer/.venv/bin/python
export PYCT_KERAS_HOME=/home/soslab-tobias/.keras
export BASE_CAMPAIGN_ID=aces_shap_only_100c_1800s_20261018
export SHAP_SENSITIVITY=0 SHAP_CACHE_MODE=reuse
export SHAP_REUSE_ROOT=/home/soslab-tobias/INCITE-worktrees/aces-shap-four-controls-100-20261012/PyCT-Influence-Transformer/exp/aces_shap_four_100c_1800s_20261012/shap_target_class
bash scripts/run_aces_shap_two_groups.sh --prepare
```

Preparation alone starts no attack workers. Once the baseline campaign has
finished, start `--group A` and `--group B` in separate tmux sessions using
the same exported settings and this worktree. Each group checks the ready gate
before running. Alternatively, `--launch` prepares and starts both groups at
once, but requires a fresh campaign root; it cannot reuse a prepared root.
Do not run it after `--prepare` with the same campaign ID. Each campaign has
four case workers, so overlapping baseline and SHAP-only campaigns would use
up to eight workers and could change timing under the fixed case deadline.

The wrapper resolves `PYCT_KERAS_HOME`, falling back to `KERAS_HOME` and then
`~/.keras`, and forwards that absolute path to both tmux sessions. The same
dataset cache location is included in the prepared parameter gate.

Preview without writing files or starting tmux/SHAP/attacks:

```bash
DRY_RUN=1 CHECK_MODELS=0 BASE_CAMPAIGN_ID=aces_shap_four_reviewed \
  bash scripts/run_aces_shap_two_groups.sh
```

After commit approval, push locally and update the experiment checkout on
530lab through git. Start preparation inside tmux so it survives SSH exit:

```bash
tmux new-session -s aces_shap_four_prepare
BASE_CAMPAIGN_ID=aces_shap_four_reviewed \
  bash scripts/run_aces_shap_two_groups.sh --launch
```

The wrapper creates `aces_shap_four_reviewed_A` and
`aces_shap_four_reviewed_B` after preparation succeeds. Preparation runs in
the foreground of the caller's session. `--prepare` runs only preparation;
to start a prepared group manually, keep the same exported campaign parameters
and use `--group A` or `--group B` in its own tmux session.

Artifacts default to `exp/<BASE_CAMPAIGN_ID>/`: `preflight.json`,
`preflight.log`, per-model SHAP logs (refresh mode), `shap_cache.log`,
`shap_cache_manifest.json`, `shap_target_class/`, `ready.json` and
`status.txt`. Each `group_A/` or `group_B/` has its own manifest, case CLI logs,
completion records and `report/{report.json,cases.csv}`. Actual case arrays
and stats remain in the existing `exp/<model>_global-real_smtlast_...` layout,
linked by the group's completion records. `group_A.exit_code` and
`group_B.exit_code` record normal lane completion; interruption may omit them.
`groups_started` records launch, not experiment completion. Attach to a lane
with `tmux attach -t aces_shap_four_reviewed_A`; during a long case, inspect
the current file under `group_A/logs/` for worker output.

Campaign roots and tmux names cannot be reused by `--launch`; failed preparation
remains available for inspection. A group failure stops that group and is
reported separately; the other group continues. If session B cannot be created,
the wrapper cancels session A. Closing a lane sends termination to its batch
runner so its worker process groups are cleaned up.

At full timeout usage, the case search budgets total about 10.4 days at four
workers. SHAP refresh, PWL preprocessing and model startup add to that time.

## Joint photo controls: two or three independent sliders

`--global-shift-kind aces-multi` selects standalone joint PyCT search. Each
selected control has its own real variable, starts at zero and can change in
the same SMT solution. Choose two or three unique short names from
`exposure contrast highlights shadows saturation vibrance curves`.
The pipeline applies selected controls in that canonical order, regardless of
CLI ordering. Each candidate is rendered afresh from the original image.
All zeros preserve the source exactly; one nonzero slider reduces to its
existing exact scalar control. Curves uses the existing scalar five-anchor
curve family.

This mode enables target SHAP priority, disables Probe and uses alpha zero.
Sensitivity, hybrid DE and strict bounds are rejected. Target SHAP caches
remain tied to the original prediction and can be reused across combinations.
Prepare compatible all-layer caches with `python -m pyct.shap` as above.

```bash
PYCT_SMT_PATH_MODE=last PYCT_BG_PER_CLASS=3 PYCT_BG_SEED=2233 \
  .venv/bin/python -m pyct --dataset cifar10 \
  --model-name cifar10_concolic_transformer --attack-mode global-real \
  --global-shift-kind aces-multi --aces-controls exposure highlights saturation \
  --aces-control-bounds=highlights:-0.1:0.1 \
  --global-x-min -0.2 --global-x-max 0.2 \
  --aces-pwl-max-regions 512 --aces-pwl-error-tolerance 0.00392156862745098 \
  --shap-output-root /absolute/path/to/shap_target_class \
  --first-n 1 --pixel-search 1 --num-process 1 \
  --timeout 1800 --solver-run-timeout 60 \
  --constraint-build-timeout-seconds 30 --symbolic-path-threshold 2000
```

The common bounds default to `[-0.2,0.2]` for this mode. Repeat
`--aces-control-bounds=CONTROL:MIN:MAX` to override individual axes; unselected
and duplicate names, nonfinite bounds, ranges excluding zero and invalid
control domains fail validation. Scalar/hybrid default bounds remain ±0.1.

The search image uses an adaptive conforming simplex mesh over the selected
parameter box. Within each triangle/tetrahedron each RGB element is affine
in all selected variables. In last mode, every variable's bounds remains in
the query together with the final negated predicate. A new forward uses the
entire returned parameter vector. Candidate reference predictions and saved
images use the exact ordered transform.

`--aces-pwl-max-regions` is a separate mesh budget, default 512. It does not
replace path threshold or the scalar segment cap. Validation samples a
barycentric quarter grid plus centroid in every simplex, measuring maximum
absolute RGB error against the exact renderer. It is a sampled check, not a
uniform error certificate. Refinement inserts a full axis plane to maintain
shared-face continuity; the mesh always contains zero. Failure to satisfy
the configured tolerance within the cap stops task preparation. The runtime
also rejects a candidate whose exact/PWL error exceeds tolerance. Neither
bounds, tolerance nor region cap is automatically relaxed. Larger joint
domains may require substantially more preparation and solver work.

Results include canonical controls/order, per-axis bounds, variable names,
mesh knots/count/cap/version and a source/mesh SHA256 receipt. Successful
parameters are saved as named `global_real_solved_params` and an ordered
`global_real_solved_parameter_vector`. `sat_aces_params.npy` has shape
`(SAT candidates, number of controls)` aligned with `sat_inputs.npy`.
`ori_input.npy`/`ori_input.jpg` and `adv_input.npy`/`adv_<labels>.jpg` retain
the before/after images. Result paths contain short control names and a
configuration digest covering bounds, region budget, tolerance and version.

For `python -m pyct.aces_stats`, joint runs require a campaign `manifest.json`
with the existing SHAP cache/model provenance settings and explicit expected
cases. Each experiment adds `controls` and `control_bounds`, for example:

```json
{"model": "cifar10_concolic_transformer", "control": "aces-multi",
 "controls": ["exposure", "highlights", "saturation"],
 "control_bounds": [[-0.2, 0.2], [-0.1, 0.1], [-0.2, 0.2]],
 "case_ids": [0], "result_dir": "/absolute/path/to/model_result_directory"}
```

The campaign settings must include `shap_priority_enabled="1"`,
`shap_sensitivity_enabled="0"`, `score_alpha="0"`, `shap_output_root`,
`bg_per_class`, `bg_seed`, and the usual `model_sha256` map. Reports keep
different control combinations separate, verify cache and mesh receipts,
check parameter bounds, and replay every SAT/success image with the exact
ordered renderer. They do not repeat neural-network inference. Existing
two-group scalar wrappers continue to run scalar-control experiments; use
the joint CLI explicitly for this mode.
## Staged multi-control recipes

`aces-multi` keeps `--aces-recipe-policy fixed` as its default. Opt into
`--aces-recipe-policy staged` to try canonical pairs of the selected controls,
then the triple when three controls were selected. For example, exposure,
highlights and saturation produce these attempts:

1. Exposure + highlights.
2. Exposure + saturation.
3. Highlights + saturation.
4. Exposure + highlights + saturation.

Every attempt starts at zero parameters on the original image. A new PyCT
engine and constraint queue are created for each recipe, while the original
target-class SHAP cache is reused. SHAP-only priority, alpha zero, independent
bounds, No Probe and exact reference verification apply to every attempt.
Selection order is deterministic; this is a staged search, not a learned
recipe-ranking algorithm.

The entire case shares `--timeout`, including PWL construction, model setup,
forward execution and solver work. Each recipe receives the remaining case
time divided by the number of remaining recipes. An early finish leaves its
unused time for later recipes. A successful exact prediction flip stops the
case immediately. An outer supervisor stops an overdue attempt's process
session, including forward and solver descendants; process cleanup and writing
the final receipt can add a small amount of wall time after the deadline.

In staged mode, `--aces-pwl-region-budgets` defaults to `512 1024 2048`.
The builder increases its region cap only when refinement cannot continue at
the current cap. It retains previously computed transform samples and region
errors. This never changes the requested X bounds or the RGB error tolerance
(normally `1/255`). An explicitly supplied `--aces-pwl-max-regions` must match
the final ladder entry. `--aces-pwl-build-timeout` defaults to 30 seconds for
each recipe and is also constrained by that recipe's remaining time. This is
separate from the existing constraint-build timeout.

Example (choose `X_MIN` and `X_MAX` explicitly before launching):

```bash
.venv/bin/python -m pyct \
  --dataset cifar10 --model-name cifar10_concolic_transformer \
  --attack-mode global-real --global-shift-kind aces-multi \
  --aces-controls exposure highlights saturation \
  --aces-recipe-policy staged \
  --aces-pwl-region-budgets 512 1024 2048 \
  --aces-pwl-build-timeout 30 \
  --global-x-min "${X_MIN}" --global-x-max "${X_MAX}" \
  --timeout 1800 --solver-run-timeout 60 \
  --symbolic-path-threshold 2000 --first-n 1 --num-process 1
```

Each case saves `recipe_summary.json` and an aggregate `stats.json`. Individual
attempts save their ordinary PyCT artifacts under `recipes/01-<controls>/`,
including source/adversarial pictures and SAT parameter vectors when available.
The recipe summary records allocation, elapsed time, PWL budgets and errors,
transition reasons, and the winning recipe. Region exhaustion or a PWL build
timeout advances to the next recipe; unexpected errors stop the case.

If no recipe reaches search because all PWL builds fail, the case is `invalid`,
not `exhausted`. Partial coverage is `incomplete`; search/case deadline expiry
is `timeout`. Only exhaustion of every planned recipe is `exhausted`.

Use `python -m pyct.aces_stats --recipes --path <campaign> --output-dir <report>`
with the campaign manifest and local CIFAR10 cache to audit the receipts.
The report replays exact candidate images and counts each model/case once,
with prediction flips separate from ground-truth correct-to-wrong outcomes.
Invalid/error cases do not enter observed ASR denominators, and incomplete
coverage prevents a complete-campaign ASR claim. PWL error remains sampled
validation rather than a uniform mathematical bound.
