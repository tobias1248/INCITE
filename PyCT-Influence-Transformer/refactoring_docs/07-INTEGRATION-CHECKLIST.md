# ✅ 集成與驗證清單

## Milestone 2 Compatibility Batch

This milestone is accepted as a compatibility-preserving modularization before the final KLEE-style engine rewrite.

- [x] `libct/state/` exposes `ExecutionState`, `StateManager`, and `ConstraintWorkItem`.
- [x] `libct/searcher/` owns stack, queue, priority, and random constraint worklist behavior.
- [x] `libct/executor/` exposes a legacy executor adapter used by `ExplorationEngine`.
- [x] `priority_queue` keeps the existing score formula and pop return shape for the compatibility path.
- [x] Existing CLI entrypoints and primary output paths remain available; solver attempt observability is now additive through bounded `solver_iter1_top3` artifacts and slimmed `stats.json` entries.
- [x] Required validation: full pytest suite and CLI help smoke checks passed on 2026-05-07.
- [ ] Environment-dependent validation: one tiny queue-mode attack and one tiny SHAP-mode attack when local cache/model/artifacts/solver are available.

The final target of `libct/explore.py < 300 行` remains a later engine rewrite acceptance criterion, not a blocker for this compatibility batch.

## Phase 1：單元測試

### Searcher 模塊測試

- [ ] `test_priority_searcher_selects_highest_score`
  - 添加低分和高分狀態，驗證高分先被選出
- [ ] `test_priority_searcher_alpha_linear_schedule`
  - 驗證每隔 N 次迭代 α 正確增加
- [ ] `test_priority_searcher_alpha_capped_at_max`
  - 驗證 α 不超過 alpha_max（默認 0.95）
- [ ] `test_priority_searcher_remove_states`
  - 添加 3 個狀態，移除 1 個，驗證 len() == 2
- [ ] `test_dfs_searcher_lifo_order`
  - 添加 A、B、C，驗證選出 C、B、A
- [ ] `test_bfs_searcher_fifo_order`
  - 添加 A、B、C，驗證選出 A、B、C
- [ ] `test_random_searcher_reproducible_with_seed`
  - 兩個相同 seed 的 RandomSearcher 選擇順序相同
- [ ] `test_searcher_empty_raises_index_error`
  - 對空 Searcher 調用 select_state() 應拋出 IndexError
- [ ] `test_searcher_empty_returns_true_when_no_states`
  - 空 Searcher 的 empty() 返回 True
- [ ] `test_searcher_len_accurate`
  - 添加 N 個狀態後 len() == N

### ExecutionState 測試

- [ ] `test_execution_state_default_values`
  - 驗證 depth=0, status='pending', constraints=[]
- [ ] `test_execution_state_pickle_serializable`
  - `pickle.dumps(state)` 不拋出異常
- [ ] `test_fork_creates_correct_successor`
  - fork() 後 depth+1, parent_id 正確
- [ ] `test_shap_methods_with_none_scores`
  - shap_scores=None 時 get_max_shap_score() 返回 0.0
- [ ] `test_get_top_k_shap_indices_correct_order`
  - 給定 [0.1, 0.5, 0.3, 0.9], k=2 返回 [3, 1]
- [ ] `test_mark_adversarial`
  - mark_adversarial() 後 is_adversarial() 返回 True
- [ ] `test_add_coverage_merges_sets`
  - 兩次調用 add_coverage() 正確合併集合

### StateManager 測試

- [ ] `test_state_manager_add_and_get`
  - 添加後能通過 path_id 找到狀態
- [ ] `test_state_manager_duplicate_raises`
  - 添加重複 path_id 拋出 ValueError
- [ ] `test_state_manager_remove_updates_stats`
  - remove() 後 total_completed += 1
- [ ] `test_state_manager_get_adversarial_states`
  - 3 個狀態中 2 個標記為 adversarial，get_adversarial_states() 返回 2
- [ ] `test_state_manager_thread_safety`
  - 10 個線程各添加 100 個狀態，最終 count() == 1000
- [ ] `test_state_manager_clear`
  - clear() 後 count() == 0

### Executor 測試

- [ ] `test_symbolic_executor_initial_returns_root`
  - execute_initial() 返回 path_id='root', depth=0
- [ ] `test_symbolic_executor_timeout_returns_empty`
  - 超時的 execute() 返回空列表（不崩潰）
- [ ] `test_concrete_executor_collects_coverage`
  - execute_initial() 返回非空 coverage 集合
- [ ] `test_executor_subprocess_crash_handled`
  - 子進程崩潰不影響主進程，返回空列表

---

## Phase 2：集成測試

### 模塊間協作

- [ ] `test_engine_with_bfs_searcher`
  - BFSSearcher + SymbolicExecutor 完整探索 2 層路徑
- [ ] `test_engine_with_priority_searcher`
  - PrioritySearcher 優先探索高 SHAP 分數路徑
- [ ] `test_engine_stop_on_timeout`
  - 設置 timeout=1，驗證引擎在 1-3 秒內終止
- [ ] `test_engine_stop_on_first_adversarial`
  - 設置 stop_on_first_adversarial=True，找到第一個樣本後停止
- [ ] `test_engine_exhausted_termination`
  - 所有路徑探索完後 termination_reason='exhausted'
- [ ] `test_state_manager_stats_after_exploration`
  - 探索完成後 total_created > 0, total_completed > 0
- [ ] `test_searcher_update_called_with_correct_args`
  - 驗證每次迭代都正確調用 searcher.update(current, added, removed)

### 數據流驗證

- [ ] `test_successor_states_added_to_manager`
  - execute() 返回的後繼狀態都被加入 StateManager
- [ ] `test_adversarial_states_marked_correctly`
  - 預測標籤改變的狀態被標記為 adversarial
- [ ] `test_results_adversarial_inputs_not_none`
  - 所有 adversarial 狀態的 input_data 都不為 None

---

## Phase 3：系統測試

### 功能驗證（與舊版輸出對比）

- [ ] `test_cli_shap_mode_backward_compat`
  ```bash
  # 使用舊版本運行並保存輸出
  uv run python -m pyct --attack-mode shap ... > old_output.json
  
  # 使用新版本運行
  uv run python -m pyct --attack-mode shap ... > new_output.json
  
  # 對比結果（允許隨機性差異）
  python scripts/compare_outputs.py old_output.json new_output.json
  ```
  - 驗證對抗樣本數量在允許誤差範圍內（±5%）
  - 驗證統計指標格式相同

- [ ] `test_cli_queue_mode_backward_compat`
  - 同上，攻擊模式改為 queue

- [ ] `test_cli_random_mode_backward_compat`
  - 同上，攻擊模式改為 random

- [ ] `test_multiprocess_behavior_unchanged`
  - 多進程（--num-process 4）與單進程結果一致

### 性能測試

- [ ] `test_exploration_performance_no_regression`
  - 單個 case 的探索時間相比舊版不超過 20% 增加

- [ ] `test_memory_usage_no_regression`
  - 1000 次迭代後內存使用量與舊版相當

- [ ] `test_priority_searcher_1000_states_under_10ms`
  ```python
  import time
  searcher = PrioritySearcher()
  states = [make_state(depth=i, shap=random()) for i in range(1000)]
  searcher.update(None, added=states, removed=[])
  
  start = time.perf_counter()
  for _ in range(1000):
      searcher.select_state()
  elapsed_ms = (time.perf_counter() - start) * 1000
  assert elapsed_ms < 10
  ```

---

## Phase 4：代碼質量檢查

### 靜態分析

- [ ] `mypy --strict libct/searcher/ libct/state/ libct/executor/`
  - 無類型錯誤

- [ ] `ruff check libct/searcher/ libct/state/ libct/executor/ libct/explore.py`
  - 無 lint 警告

- [ ] `ruff format --check libct/`
  - 格式符合標準

### 測試覆蓋率

```bash
# 運行覆蓋率分析
uv run pytest test/unit/ test/integration/ --cov=libct/searcher \
  --cov=libct/state --cov=libct/executor --cov-report=term-missing
```

- [ ] `libct/searcher/` 覆蓋率 > 85%
- [ ] `libct/state/` 覆蓋率 > 90%
- [ ] `libct/executor/` 覆蓋率 > 80%
- [ ] `libct/explore.py` 覆蓋率 > 75%

### 性能基准

```bash
# 運行基准測試
uv run python -m pytest test/benchmarks/ -v
```

- [ ] Searcher select_state() 1000 次 < 10ms
- [ ] StateManager add() 1000 次 < 5ms
- [ ] ExplorationEngine 初始化 < 100ms

### 代碼複雜度

```bash
# 檢查循環複雜度
uv run radon cc libct/explore.py libct/searcher/ libct/state/ libct/executor/ -s
```

- [ ] 所有方法的 McCabe 複雜度 ≤ 10
- [ ] 所有方法不超過 50 行

---

## Phase 5：文檔和發佈

- [ ] 所有公共類和方法有 docstring
- [ ] `CHANGELOG.md` 已更新（新增重構說明）
- [ ] `README.md` 中的架構圖已更新
- [ ] 新模塊的 `__init__.py` 導出正確的公共 API
- [ ] `refactoring_docs/` 所有文檔已審閱

---

## 驗收準則

重構完成的判斷標準（必須全部滿足）：

| 準則 | 驗證方法 |
|------|---------|
| ✅ 代碼行數：每模塊 < 300 行 | `wc -l libct/explore.py libct/searcher/*.py` |
| ✅ 類型安全：無 mypy 錯誤 | `mypy --strict libct/` |
| ✅ 單元測試：覆蓋率 > 80% | `pytest --cov=libct` |
| ✅ 向後兼容：CLI 行為不變 | 系統測試對比 |
| ✅ 無循環依賴 | `pydeps libct --max-bacon 3` |
| ✅ 文檔完整 | 代碼審查 |

---

## 時間線表

| 週次 | Phase | 主要任務 | 里程碑 |
|------|-------|---------|-------|
| 第 1 週 | Phase 1 | 定義 ExecutionState, StateManager | State 模塊完成 |
| 第 2 週 | Phase 1 | 實現 Searcher 接口和 4 個策略 | Searcher 單元測試通過 |
| 第 3 週 | Phase 2 | 提取 SymbolicExecutor | Executor 可獨立運行 |
| 第 4 週 | Phase 2 | 實現 ConcreteExecutor，集成測試 | 集成測試通過 |
| 第 5 週 | Phase 3 | 重構 ExplorationEngine | Engine < 300 行 |
| 第 6 週 | Phase 3-5 | 系統測試，代碼清理，文檔 | 全部驗收準則通過 |

---

## 常見問題

**Q：重構過程中如何保持可工作的代碼？**
A：採用「絞殺者模式」（Strangler Fig Pattern）：
1. 先創建新模塊，不修改舊代碼
2. 在 ExplorationEngine 中逐步引用新模塊
3. 最後刪除舊代碼

```python
# 過渡期：ExplorationEngine 同時支持新舊接口
class ExplorationEngine:
    def __init__(self, searcher=None, ...):
        if searcher is not None:
            self._searcher = searcher  # 新路徑
        else:
            self._legacy_mode = True  # 舊路徑（向後兼容）
```

**Q：如何處理 SHAP 計算的依賴？**
A：SHAP 計算邏輯保留在 `explainability/shap_calculator.py`，通過依賴注入傳入 `PrioritySearcher`：

```python
searcher = PrioritySearcher(
    shap_calculator=ShapValuesComparator(model, background_data)
)
```

**Q：多進程環境下 StateManager 是否需要 multiprocessing.Lock？**
A：不需要。每個 worker 進程有獨立的 StateManager 實例，進程間不共享狀態。`threading.Lock` 足夠保護進程內的多線程訪問（如果有）。

**Q：如何運行特定 Phase 的測試？**
```bash
# Phase 1：單元測試
uv run pytest test/unit/ -v

# Phase 2：集成測試
uv run pytest test/integration/ -v

# Phase 3：系統測試（需要完整環境）
uv run pytest test/system/ -v --timeout=300

# 全部
uv run pytest -v
```
