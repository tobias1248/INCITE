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
