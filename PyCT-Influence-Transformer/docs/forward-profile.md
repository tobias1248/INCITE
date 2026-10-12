# One-case forward profiling

This diagnostic uses an existing `.h5` model and one cached CIFAR10 test image.
It measures computation and path-management cost, without running an attack,
solver, DE, or reference Probe. It does not train or generate a toy model.

## Defaults

| Setting | Value |
| --- | --- |
| Model | `cifar10_concolic_transformer` |
| Dataset case | CIFAR10 test index 0 |
| Control | `aces-contrast`, `X=0`, clip range `[-0.2,0.2]` |
| Mode selection | `all`; `--mode` can select a single mode |
| PWL | maximum 32 segments, sampled tolerance `1/255` |
| Symbolic branch threshold | 2000 |
| Forward deadline | 1800s per mode |
| Setup watchdog | 300s per mode, excludes forward |
| Execution | four modes sequentially, one fresh process per mode |
| Resources | CPU only; OMP and TF intra/inter threads each 1 |
| Queue priority | existing scheduler, alpha 0.8, no SHAP comparator |
| Ternary simplification | off |

Available modes use the same normalized source image:

1. **`keras_concrete`**: native Keras reference prediction. One warmup is
   recorded separately; measured forward includes the prediction API overhead.
2. **`nnmodel_concrete`**: existing NNModel with ordinary float inputs, measuring
   its Python layer loops and concrete computation.
3. **`symbolic_path`**: existing concolic arithmetic, path tree, and priority
   queue. The production path/scheduler code is reused by a forward-only engine
   adapter; there is no attack executor or child-payload serialization.
4. **`symbolic_no_path`**: identical symbolic inputs and arithmetic, with branch
   visits counted instead of building path nodes and queue entries. The same
   ConcolicBool guard disables symbolic tracking at the same threshold. This
   diagnostic discards every attack candidate and cannot replace normal PyCT.

`--global-shift-kind` selects `aces-brightness` or `aces-contrast`. At `X=0`
both preserve the source image, but their symbolic input expressions describe
different controls. The selected control is recorded in the manifest and each
successful worker report. A single mode measures its bottlenecks; it cannot
supply a path-versus-no-path speedup comparison.

## Concrete computation after tracking stops

`NNModel.forward` remembers the engines owning its input for each forward.
When every owning engine explicitly reports `symbolic_enabled=False`, the
next layer receives a copy containing ordinary values instead of concolic
objects. Cached inputs are converted when consumed, once per cache entry,
including residual skip inputs used by `Add`. Original inputs, layer output
objects and recorded predicates are not rewritten. Each new forward starts
with fresh conversion state.

This applies to the shared VGG, ResNet and Transformer execution path.
Conversion happens at a layer boundary: the layer that disables tracking
finishes normally. The first convolution and work before the threshold are
still symbolic, and subsequent concrete convolutions still use Python loops.
The existing streaming attention conversion can also act within attention.
Missing engine metadata or a missing/active tracking flag does not authorize
conversion. No additional CLI setting is required.

Compare the same model, image, control, threshold and profiling settings on
the two source revisions. Check prediction, branch prefix and disable depth
as well as time; Transformer speedups do not establish a VGG or ResNet speedup.

All modes include **cProfile overhead**. These are diagnostic timings, not
unprofiled experiment timings. Native Keras and NNModel can have different
predictions because NNModel contains numerical/activation approximations.
The two symbolic modes must agree on prediction, branch count and disable
depth before the report presents a timing ratio. A single image and one run
per mode cannot establish a general speedup.

## Run locally

The cached model and `~/.keras/datasets/cifar-10-batches-py/test_batch` must
already exist. `PYCT_KERAS_HOME` selects a different local Keras cache.
Downloads and GPU use are disabled by this CLI.

```bash
.venv/bin/python -m pyct.profile_forward --help

# Bounded integration smoke, not an attack effectiveness experiment.
.venv/bin/python -m pyct.profile_forward \
  --mode symbolic_path --global-shift-kind aces-brightness \
  --timeout 10 --output-dir exp/forward_profile/smoke_case0

# Full diagnostic; up to four 1800s forward budgets plus setup.
.venv/bin/python -m pyct.profile_forward \
  --mode all --global-shift-kind aces-contrast \
  --timeout 1800 --symbolic-path-threshold 2000 \
  --output-dir exp/forward_profile/transformer_case0
```

An existing output directory is refused. Optional arguments are
`--model-name`, `--case-index`, `--dataset-cache` (a local `test_batch` file),
`--timeout`, `--setup-timeout`, `--symbolic-path-threshold`, `--mode`,
`--global-shift-kind`, and `--output-dir`.

## Run on 530lab

After committing and pushing, deploy the branch through Git on 530lab before
running the diagnostic. The main checkout has other development in progress;
create an independent worktree and experiment branch for this measurement.
Do not deploy an uncommitted working tree.

```bash
ssh 530lab
git -C /home/soslab-tobias/INCITE fetch origin feat/de-pyct-hybrid
git -C /home/soslab-tobias/INCITE worktree add \
  -b exp/forward-profile-brightness-900s \
  /home/soslab-tobias/INCITE-worktrees/forward-profile-brightness-900s \
  origin/feat/de-pyct-hybrid
cd /home/soslab-tobias/INCITE-worktrees/forward-profile-brightness-900s/PyCT-Influence-Transformer
git pull --ff-only origin feat/de-pyct-hybrid
ln -s /home/soslab-tobias/INCITE/PyCT-Influence-Transformer/.venv .venv
sha256sum model/cifar10_concolic_transformer.h5
tmux new-session -s forward-brightness-900s
.venv/bin/python -m pyct.profile_forward \
  --mode symbolic_path --global-shift-kind aces-brightness \
  --case-index 0 --timeout 900 --symbolic-path-threshold 2000 \
  --output-dir exp/forward_profile/brightness_case0_900s
```

## Results and timeout handling

The output contains:

- `manifest.json`: selected mode/control, parameters, transform settings, resource environment,
  and model SHA256.
- `summary.json`: selected mode results, comparability, and timing ratio when
  both symbolic forwards finish with matching prediction/count/disable depth.
- `<mode>/result.json`: model load/conversion, dataset/PWL/input preparation,
  forward wall/CPU duration, branch count, queue/node count, and layer timings.
- `<mode>/events.jsonl`: progress flushed by the parent while the worker runs.
- `<mode>/forward.prof`: cProfile data when graceful profiling finishes.

Self-time groups partition the profiler's function self-times into path
management, symbolic arithmetic/object utilities, model Python code, and
other functions. Utilities include engine lookup and concrete unwrapping.
They are not pure arithmetic categories. Function cumulative time and layer
time overlap those groups and must not be added to them. TensorFlow background
thread work is not individually captured by cProfile in the calling thread.

A graceful forward timeout preserves partial cProfile data and marks the
active layer as `timeout`. A hard watchdog retains only progress already sent
to the parent; unfinished layers are marked and missing times stay null.
Setup errors are reported separately. A timeout is never presented as a
completed forward time or evidence that one mode is faster. The CLI exits
nonzero for execution errors; recorded timeout results are expected diagnostics.
Keyboard interruption terminates the active worker.

Use the per-layer and self-time data to decide whether to optimize path
management, symbolic expression construction, or the concrete Python loops.
Generated experiment artifacts stay under ignored `exp/`.
