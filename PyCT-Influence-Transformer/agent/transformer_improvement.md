# Transformer Attack Improvement Plan

## Goal

Improve transformer attack performance against `cifar10_concolic_transformer.h5` with the correct priority:

1. Fix correctness issues first.
2. Then reduce repeated runtime overhead.
3. Finally improve success-rate policy.

Current diagnosis: the main bottlenecks are **forward/constraint generation** and **SHAP ranking signal quality**, not SMT solving (`exec_mean >> solve_mean` in `insight.md`).

## Model Snapshot

`Input(32,32,3) -> Reshape(1024,3) -> MHA(4,key_dim=16) -> BN -> Flatten -> Dense(128,relu) -> Reshape(16,8) -> MHA(4,key_dim=8) -> BN -> Flatten -> Dense(10,softmax)`

Implication: first attention layer has sequence length `L=1024`; pure Python concolic forward is expensive.

## Attack Chain (Main Path)

1. `start_test.py` CLI entry
2. `start_launch.py` builds task waves/worker execution
3. `run_dnnct.py` runs one concolic task
4. `libct/explore.py` builds SHAP comparator + queue + exploration
5. `libct/solver.py` converts constraints and calls solver

## Priority Backlog

### P0 Correctness

- Softmax/exp symbolic correctness in transformer forward (`dnnct/myDNN.py`)
- MHA position registration alignment in attention max-selection path (`dnnct/myDNN.py`)

### P1 Performance

- SHAP comparator JSON-only fast path when SHAP is pre-calculated (`libct/shapInfl.py`)
- Process-local predictor module cache (`run_dnnct.py`, `libct/utils.py`)
- Reduce process churn in concolic execution (`libct/explore.py`)

### P2 Data/Structure Overhead

- Reduce repeated large dict/background construction (`utils/dataset.py`, `utils/experiment_task_specs.py`)

### P3 Success-Rate Strategy

- Transformer-aware priority scoring (`SHAP + normalized path length + dynamic alpha`)
- Queue diversification (avoid strict top-1 bias)
- Stagnation-triggered ton escalation

## Work Stages

## Stage 1 (Completed)

Focus: correctness + immediate wins.

- [x] MHA position registration alignment fix
- [x] safe exp / softmax symbolic-safe handling
- [x] Baseline for threshold-based early symbolic stop kept available
- [x] Stage-1 validation completed (no critical regression reported)

## Stage 2 (Completed)

Focus: execution architecture and runtime overhead.

- [x] Add SHAP comparator JSON-only path; avoid model load when `shap_value_pre_calculated=True`
- [x] Add process-local predictor module cache in `run_dnnct` path
- [x] Reuse worker-side runner across tasks (reduce init churn)
- [x] Keep worker alive between ton waves
- [x] Make constraint log output switchable (default off)
- [x] Move `con_dict` toward sparse/set-like representation where safe

Acceptance targets:

- Reduce end-to-end `exec_mean` significantly without increasing solver timeout rate
- No success-rate regression compared to Stage 1

## Stage 3 (Planned)

Focus: attack effectiveness.

- [ ] Transformer-aware priority score
- [ ] Queue diversification policy
- [ ] Stagnation-triggered ton progression
- [ ] A/B comparison against Stage 2 on CIFAR10 transformer task set

Acceptance targets:

- Improve success rate while keeping wall time close to Stage 2
- Keep `solve_mean` stable (do not shift bottleneck into solver)

## Measurement Protocol

- Main metrics: `success_rate`, `wall_mean`, `wall_p90`, `exec_mean`, `solve_mean`
- Compare by stage: Stage 1 baseline vs Stage 2 vs Stage 3
- Run with fixed seed and same task slice for fair comparison

## Immediate Next Step

Implement Stage 3 items in this order:

1. Transformer-aware priority score
2. Queue diversification
3. Stagnation-triggered ton progression
4. Stage 2 vs Stage 3 A/B evaluation
