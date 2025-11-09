# AGENT.md — PyCT‑Influence‑Transformer

> Handbook for the Codex agent working on **SHAP‑guided concolic testing for transformer models** extending PyCT. Default to **read‑only** and **plan‑only** changes.

---

## 1) Mission & Scope

* **Mission**: Analyze and incrementally refactor the repo to improve determinism, safety, and maintainability while preserving behavior. Prioritize:

  1. orchestration/CLI config sanity (solver/queue/timeouts/paths),
  2. elimination of cross‑process global state,
  3. modularization of oversized modules (esp. `dnnct/myDNN.py`, `libct/concolic/*`),
  4. SHAP cache hygiene & concurrency,
  5. numeric stability for single‑layer MHA.
* **Out of scope**: shipping binaries, dataset/model redistribution, long stress or performance benchmarks unless explicitly requested.

## 2) Guardrails (Global)

* **Read‑only by default**: Do **not** modify files, commit, push, or delete. Use plan‑only diffs unless Apply mode is explicitly enabled for a single file.
* **No external network** and do not execute long‑running jobs. Respect environment constraints in `environment.yml` but do not create/activate conda env by yourself.
* **Privacy/exclude**: Ignore `exp/`, `shap_value_all_layer/`, `popped_constraint_position/`, `model/`, `**/secrets/**`, `.venv/`, `__pycache__/` in analysis and reports.
* **Solver tooling**: Treat **cvc5** as default; detect availability and fail fast with actionable guidance. Never silently fall back to cvc4.

## 3) Repository Layout (reference)

```
libct/               # core concolic engine, solver interface, SHAP comparator
libct/concolic/*     # primitives; large, scheduled for modular split
libct/shapInfl.py    # SHAP loaders/caches
libct/solver.py      # solver subprocess orchestration (cvc5)
libct/explore.py     # ExplorationEngine, queues, MP
utils/               # datasets, orchestration helpers, experiments
utils/dataset.py     # external data expectations
utils/pyct_attack_exp*.py  # runners and multi-process wrappers
dnnct/               # DNN wrappers; myDNN contains transformer logic
dnnct/myDNN.py       # attention & NNModel mapping (oversized)
dnn_predict_common.py
run_dnnct.py         # CLI orchestration
```

## 4) Phases & Deliverables

* **Phase 1 – Engine/Core Health**
  Scope: `libct/**`, `utils/**` (excluding heavy artifact dirs).
  Deliverable: `reports/phase1.md` (deps/cycles, hotspot map, queue/solver config normalization plan, global‑state audit).
* **Phase 2 – Transformer Path**
  Scope: `dnnct/myDNN.py`, `dnn_predict_common.py`, entrypoints `dnnct_transformer_multi*.py`, `run_dnnct.py`.
  Deliverable: `reports/phase2.md` (MHA numerics tests, path map, SHAP×Concolic integration notes, modularization plan).
* **Final – Integration & Roadmap**
  Deliverable: `reports/final.md` (cross‑phase risks closed, RFCs, timeline).

## 5) Execution Playbook

1. **/init**: Read this handbook and `codex.init.json`; index code; build dependency graph.
2. **/analyze**: Report cycles, hotspots, public API map, global state inventory.
3. **/inspect transformer**: Backtrace from output layer; enumerate attention path and constraints metadata.
4. **/report phase={1|2|all}**: Produce Markdown reports (read‑only).
5. **/refactor --plan-only**: Produce minimal diffs and migration notes; do not apply unless Apply mode is requested for a single file.

## 6) Quality Gates (Global)

* Behavior‑equivalent unless an RFC is approved. No change to logging schema, file outputs, or CLI defaults without deprecation path.
* Static checks: **ruff**/**mypy --strict** must not regress. No new cycles. Public symbols remain backward compatible.
* Determinism: repeated runs with fixed seeds yield identical constraint order modulo stable tie‑breakers.

## 7) Reporting Spec

* **Format**: Markdown (zh‑TW). Place under `reports/`.
* **Common sections**: Summary → Dependency graph → Hotspots → Risks & mitigations → SHAP/Concolic touchpoints → Test plan → Next actions.
* **Pointers**: Reference concrete files/lines when asserting risks or actions.

## 8) File‑by‑File Refactor Guardrails (Strict)

### 0) Modes

* **Plan‑only (default)**: emit patch suggestions & report; **no file writes**.
* **Apply (explicit)**: allow writing changes to **the current file only**; cross‑file edits require a new task after this file is complete.

### 1) Scope & Granularity

* Exactly **one** `.py` file per task. No global renames; no cross‑module API breaks; no data/model path changes.
* Allowed: function/method reordering; private helper extraction; docstrings; typing; naming normalization; dead‑code removal; behavior‑equivalent refactors.

### 2) Function‑level Full Inventory

* Before changes, produce a **Function Inventory** with: `name`, `signature`, `lineno`, `public`, `risk`, `disposition {keep|rename-internal|extract|deprecate|tests-added}`, `notes`.
* At completion, include a **Before/After** mapping (added/removed/renamed).

### 3) Quality Gates (per file)

* **Behavior unchanged** (I/O, side‑effects, exceptions).
* **Cyclomatic ≤ 12** per function post‑refactor or accompanied by a mitigation plan.
* **File size ≤ ~600 LOC** post‑refactor; otherwise submit a split plan.
* No new heavyweight deps; no implicit I/O; no randomness. MP start method & seeds unchanged.

### 4) API Stability

* Public API (imported by other modules) must keep names & parameters. For necessary changes: wrap with backward‑compatible shims, emit `PendingDeprecationWarning`, and record at file top.

### 5) Style & Static Analysis

* PEP8; **docstrings on all new/changed functions**; add explicit type hints. `mypy --strict` and `ruff` clean; inline `# noqa` allowed with rationale recorded in the inventory.

### 6) Tests (per file)

* At least **one unit test per changed/new function**. Cover edge cases (empty/None/zero‑len/extremes).
* Transformer/MHA: verify masks, softmax stability, attention weights sum to 1, tensor shapes preserved.
* SHAP/Concolic: cache key semantics unchanged; if adding locks, include a **concurrency test**.
* Solver: CLI/binary name & timeout side‑effects (SMT dump) unchanged.

### 7) Performance & Memory

* Do not increase peak RSS materially. Avoid promoting large objects to wider scope. If adding caches, provide per‑file cleanup or scope limits; use process‑safe primitives/locks.

### 8) Logging & Observability

* Keep existing fields/levels. New debug logs must be optional and disabled by default. For queue/solver events, prefer structured JSON fields without removing existing ones.

### 9) Workflow (per file)

1. **Inventory** → 2) **Proposal** (≤300 changed lines, ≤5 hunks) → 3) **Checks** (ruff/mypy/tests if allowed) → 4) **Patch/Report** (plan‑only) → 5) **Record** (update inventory with dispositions & links to tests).

### 10) Definition of Done

* Every function accounted for; tests attached where required; static checks clean; public API compatible; behavior equivalence demonstrated.

### 11) Prohibited

* Multi‑file edits; global renames; changing data/model/cache directory semantics; removing/renaming log fields; introducing networking or hidden I/O.

### 12) Escalations & Exceptions

* For cross‑file or breaking changes, submit an **RFC** (impact, migration path, tests) and wait for approval before proceeding. Record exceptions under an `Exceptions` section in the file’s report.

#### Appendix A — Function Inventory Table (template)

```markdown
## Function Inventory — <file.py>
| name | signature | lineno | public | risk | disposition | notes |
|------|-----------|--------|--------|------|------------|-------|
| foo  | (x: np.ndarray, k:int)->np.ndarray | 123 | yes | med | keep | add type hints, docstring |
| bar  | (cfg: Dict)->None | 220 | no  | low | extract | split I/O out |
```

#### Appendix B — Small‑step Diff Rules

* ≤ **300** changed lines; ≤ **5** hunks; ≤ **1** new file (tests or private helper).
* Commit message: `<file>: <concise change>; behavior‑equivalent; add tests`.
* Include: Before/After function inventory mapping; static‑check summary; test summary.

---

## 9) Known Risks & Immediate Actions (from audit)

* Normalize **queue mode** API (string vs bool) and fail fast on unsupported modes.
* Align solver to **cvc5**, add availability checks and `--solver` flag.
* Eliminate **globals across processes** via `ExploreContext`/`SolverContext`.
* Modularize `dnnct/myDNN.py` and `libct/concolic/*`; add focused MHA numeric tests.
* Add file‑locking & memoization for SHAP caches; index by model hash/layer id.

## 10) Environment & Tooling

* Python **3.9**; package tools: conda/pip; external: **cvc5** (required).
* Respect `environment.yml` versions; provide `/check-env` style preflight where possible.

## 11) Glossary

* **Concolic**: combined concrete+symbolic execution to explore paths/constraints.
* **SHAP**: feature contribution measure; here used to **prioritize** constraint solving, not to prove feasibility.
* **MHA**: multi‑head attention; verify masking/softmax numerics and shape invariants.
