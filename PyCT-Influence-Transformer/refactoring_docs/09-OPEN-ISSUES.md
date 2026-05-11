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
- **Found during**: Phase 2A completion review
- **Problem**: `libct/explore.py` still owns the main exploration loop, `explore()` setup/teardown, primitive coverage execution, argument wrapping, coverage reporting, and search compatibility wrappers.
- **Impact**: The file remains difficult to reason about and is still far from the target coordinator/facade shape.
- **Suggested follow-up**: Continue with Phase 2B primitive runner extraction, then Phase 2C execution-pair adapter, then Phase 3 runtime/session loop extraction.

### ISSUE-6: Python version contract is inconsistent

- **Status**: Open
- **Area**: Packaging
- **Found during**: Phase 2A test additions
- **Problem**: Repository guidance says Python 3.9-compatible code, but some existing code uses newer syntax such as `float | None`.
- **Impact**: Packaging or artifact users on Python 3.9 may fail before runtime tests execute.
- **Suggested follow-up**: Decide whether the project targets Python 3.9 or a newer version. If Python 3.9 remains the target, replace newer syntax and add a compatibility check.
