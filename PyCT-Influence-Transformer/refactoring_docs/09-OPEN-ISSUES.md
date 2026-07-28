# Refactoring Open Issues

This file tracks issues discovered during the `libct/explore.py` refactoring.
When a new risk or cleanup item is found, record it here before or during the
next implementation batch.

## How to Record Issues

Use this format for new entries:

```markdown
### ISSUE-N: Short title

- **Status**: Open | In Progress | Resolved | Deferred
- **Area**: Runtime | Observability | Compatibility | Testing | Packaging
- **Found during**: Phase or commit/context
- **Problem**: What is wrong or risky.
- **Impact**: Why it matters.
- **Suggested follow-up**: Smallest reasonable next action.
```

## Open Issues

### ISSUE-7: Ternary search path can trigger duplicate model execution per candidate

- **Status**: Resolved
- **Area**: Runtime
- **Priority**: High
- **Found during**: Post-refactor timing review of `total_time`, `solve_time`, and `forward_time` in `exp_storage`
- **Problem**: A SAT candidate needs both authoritative label verification and, when it is not adversarial, another concolic forward to generate constraints. Treating both executions as interchangeable allowed a search result to become an attack label.
- **Resolution**: The responsibilities are now explicit for ternary and non-ternary modes. The cached Keras model alone produces `original_label` and `attack_label`; `NNModel` alone performs concolic search. Every unique SAT candidate is checked by Keras first and is sent to `NNModel` only when its Keras label is unchanged.
- **Observability**: Reference prediction count, phase, and wall time are recorded separately, so the required Keras verification is no longer hidden inside solver or search timing.

### ISSUE-1: `child_event_message` can become too large

- **Status**: Open
- **Area**: Observability
- **Found during**: Phase 2A bounded SHAP smoke after extracting the concolic runner
- **Problem**: Soft-timeout child events currently store a full message that embeds the expanded `all_args` dictionary. For image inputs this can include thousands of scalar pixel variables.
- **Impact**: `stats.json`, logs, and diagnostics become noisy and hard to inspect. Large metadata also makes downstream parsing and experiment summaries less stable.
- **Suggested follow-up**: Store a short `child_event_message` summary in recorder metadata, and write full details to a diagnostic side file when needed. Preserve `child_event_type`, `child_event_phase`, and `child_pid`.

### ISSUE-2: `libct.explore` still depends on module-level runtime globals

- **Status**: Open
- **Area**: Runtime
- **Found during**: Phase 2A concolic runner extraction
- **Problem**: `module`, `execute`, and `recorder` remain module-level globals in `libct.explore`.
- **Impact**: Global mutable runtime state makes multiprocessing, re-entrant execution, and isolated tests more fragile. It also keeps new executor modules tied to compatibility accessors on `ExplorationEngine`.
- **Suggested follow-up**: Gradually move these values into an explicit runtime/session object or engine fields. Keep compatibility accessors such as `_get_execute()` while migrating.

### ISSUE-3: Primitive execution still uses a raw multi-pipe protocol

- **Status**: Open
- **Area**: Runtime
- **Found during**: Planning Phase 2B after concolic runner extraction
- **Problem**: `_one_execution_primitive()` still coordinates child execution with multiple raw pipes for line ranges, ready signal, answer, and coverage payload.
- **Impact**: The protocol is harder to test and less observable than the child envelope path. Parent/child failures can be ambiguous, especially around timeouts or missing payloads.
- **Suggested follow-up**: Extract a `PrimitiveExecutionRunner` first while preserving the raw protocol. After compatibility is covered, consider a structured envelope for primitive coverage results.

### ISSUE-4: Primitive child exceptions are mostly swallowed

- **Status**: Open
- **Area**: Observability
- **Found during**: Planning Phase 2B after inspecting `_one_execution_primitive()`
- **Problem**: Primitive execution catches broad exceptions and leaves the answer as the `Exception` sentinel without preserving traceback or error class.
- **Impact**: Runtime failures can look like ordinary target exceptions, making debugging and experiment triage difficult.
- **Suggested follow-up**: Preserve current sentinel behavior for compatibility, but add a diagnostic side file or recorder metadata for unexpected primitive child failures.

### ISSUE-5: `explore.py` remains a god object after child/concolic extraction

- **Status**: Open
- **Area**: Runtime
- **Found during**: Phase 2A completion review; updated after the 2026-06-02 compatibility extraction batch
- **Problem**: `libct/explore.py` no longer owns the raw search scheduler,
  candidate execution pair, or concolic argument wrapping, but it still owns the
  main `_execution_loop()`, `explore()` setup/teardown, coverage reporting,
  stats artifact finalization, and module-level runtime globals.
- **Impact**: The file is smaller and more adapter-shaped, but it is still the
  highest-risk runtime module. Moving the main loop before isolating setup,
  teardown, recorder state, and solver/result accounting would make regression
  failures hard to diagnose.
- **Suggested follow-up**: Extract a runtime session/environment helper first
  for `sys.path` mutation, working-directory switching, coverage setup, timeout
  deadline setup, and final stats artifact writes. After that, split the main
  loop into an explicit coordinator around solver attempts and candidate
  execution.

### ISSUE-6: Python version contract is inconsistent

- **Status**: Open
- **Area**: Packaging
- **Found during**: Phase 2A test additions
- **Problem**: Repository guidance says Python 3.9-compatible code, but some existing code uses newer syntax such as `float | None`.
- **Impact**: Packaging or artifact users on Python 3.9 may fail before runtime tests execute.
- **Suggested follow-up**: Decide whether the project targets Python 3.9 or a newer version. If Python 3.9 remains the target, replace newer syntax and add a compatibility check.

### ISSUE-7: `predict_compare` first layer diff can identify amplification, not root cause

- **Status**: Open
- **Area**: Observability
- **Found during**: ResNet18 CIFAR-10 predict comparison investigation
- **Problem**: After fixing BatchNorm output cache contamination from in-place ReLU, the first failing ResNet18 case moved to `conv2d_3`. Isolating `conv2d_3` with the Keras `re_lu_2` output as input showed the myDNN Conv2D result stayed within the strict tolerance, while the full forward path crossed the threshold at `conv2d_3` because earlier small differences were amplified by convolution accumulation.
- **Impact**: The current first-layer diff can be read as "this layer is semantically wrong" even when the layer is only the first point where accumulated numerical error exceeds `np.allclose(atol=1e-5, rtol=1e-5)`.
- **Suggested follow-up**: Extend `predict_compare` diagnostics to log both input and output diff for the first failing layer, use scientific notation for small values, and include top-k absolute diff locations so semantic bugs can be separated from numerical amplification.

### ISSUE-8: `explore()` setup and teardown still mix environment and artifact side effects

- **Status**: Open
- **Area**: Runtime
- **Found during**: 2026-06-02 post-extraction review
- **Problem**: `ExplorationEngine.explore()` still mutates process-wide state
  (`sys.path`, current working directory), configures coverage, creates SHAP
  comparator state, computes the global deadline, invokes the main loop, and
  writes final `statsdir` artifacts in one method.
- **Impact**: This makes the remaining coordinator hard to test in isolation
  and keeps process-state cleanup coupled to exploration-loop behavior.
- **Suggested follow-up**: Introduce a compatibility runtime session or
  environment helper that owns setup/teardown with deterministic tests for
  path restoration, cwd restoration, coverage setup, and final artifact writes.
