# 🎛️ ExplorationEngine 模塊指南

## 概述和職責

**ExplorationEngine** 是整個符號執行系統的協調者（Coordinator）。重構後它的職責大幅精簡：

**職責**（只做這些）：
- 協調 Searcher、Executor、StateManager 的互動
- 管理探索循環（主 loop）
- 超時監控
- 對抗樣本檢測
- 進度日誌輸出
- 收集和返回最終結果

**不再做的事**（已委託給子模塊）：
- ~~選擇下一條路徑~~ → Searcher
- ~~執行符號計算~~ → Executor
- ~~管理狀態集合~~ → StateManager
- ~~計算 SHAP 分數排序~~ → PrioritySearcher

---

## ExplorationEngine 完整實現（< 300 行）

```python
# libct/explore.py（重構後）
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from libct.executor.base import Executor, ExecutionError, ExecutionTimeout
from libct.searcher.base import Searcher
from libct.state.execution_state import ExecutionState
from libct.state.manager import StateManager

logger = logging.getLogger("ct.explore")


@dataclass
class ExplorationResults:
    """探索完成後的結果集合。"""

    adversarial_inputs: List[Any] = field(default_factory=list)
    """所有發現的對抗樣本輸入列表。"""

    total_states_explored: int = 0
    """探索的總狀態數。"""

    total_iterations: int = 0
    """主循環的總迭代次數。"""

    elapsed_seconds: float = 0.0
    """總耗時（秒）。"""

    termination_reason: str = "unknown"
    """
    終止原因：
    - 'timeout'：達到全局超時
    - 'exhausted'：所有路徑已探索完畢
    - 'adversarial_found'：找到對抗樣本後立即終止（如設置了 stop_on_first）
    - 'max_iterations'：達到最大迭代次數限制
    """

    state_manager_stats: dict = field(default_factory=dict)
    """StateManager 的最終統計信息。"""

    def found_adversarial(self) -> bool:
        """是否找到了至少一個對抗樣本。"""
        return len(self.adversarial_inputs) > 0

    def __repr__(self) -> str:
        return (
            f"ExplorationResults("
            f"adversarial={len(self.adversarial_inputs)}, "
            f"states={self.total_states_explored}, "
            f"iterations={self.total_iterations}, "
            f"elapsed={self.elapsed_seconds:.1f}s, "
            f"reason={self.termination_reason!r})"
        )


class ExplorationEngine:
    """
    符號執行探索引擎（協調者）。
    
    協調 Searcher、Executor、StateManager 完成完整的路徑探索。
    目標：< 300 行，不包含具體的搜索或執行邏輯。
    """

    def __init__(
        self,
        searcher: Searcher,
        executor: Executor,
        state_manager: StateManager,
        timeout: int = 900,
        max_iterations: Optional[int] = None,
        stop_on_first_adversarial: bool = False,
        verbose: int = 1,
        log_interval: int = 10,
        original_label: Optional[int] = None,
    ) -> None:
        """
        Args:
            searcher: 路徑選擇策略（PrioritySearcher、DFSSearcher 等）
            executor: 執行引擎（SymbolicExecutor 等）
            state_manager: 狀態管理器
            timeout: 全局探索超時（秒），0 表示無限制
            max_iterations: 最大迭代次數限制，None 表示無限制
            stop_on_first_adversarial: 找到第一個對抗樣本後立即停止
            verbose: 日誌級別（0=靜默, 1=基本, 2=詳細）
            log_interval: 每隔多少次迭代輸出一次進度日誌
            original_label: 原始輸入的預測標籤（用於對抗樣本檢測）
        """
        self._searcher = searcher
        self._executor = executor
        self._state_manager = state_manager
        self._timeout = timeout
        self._max_iterations = max_iterations
        self._stop_on_first = stop_on_first_adversarial
        self._verbose = verbose
        self._log_interval = log_interval
        self._original_label = original_label
```

---

### `__init__`, `explore` 主方法

```python
    def explore(self, initial_input: Any) -> ExplorationResults:
        """
        從初始輸入開始探索所有可達路徑。
        
        執行流程：
        1. 初始執行（具體執行，建立根狀態）
        2. 主探索循環（符號執行 + 狀態管理）
        3. 收集結果
        
        Args:
            initial_input: 初始輸入數據（如圖像的 numpy array）
            
        Returns:
            ExplorationResults 包含所有發現的對抗樣本和統計信息
        """
        start_time = time.monotonic()
        results = ExplorationResults()

        # Step 1: 初始執行
        root_state = self._initial_execution(initial_input)
        if root_state is None:
            results.termination_reason = "initial_execution_failed"
            results.elapsed_seconds = time.monotonic() - start_time
            return results

        # Step 2: 主探索循環
        termination_reason = self._main_loop(
            root_state=root_state,
            start_time=start_time,
        )

        # Step 3: 收集結果
        results.adversarial_inputs = [
            s.input_data
            for s in self._state_manager.get_adversarial_states()
            if s.input_data is not None
        ]
        results.total_states_explored = self._state_manager.stats()["total_completed"]
        results.state_manager_stats = self._state_manager.stats()
        results.elapsed_seconds = time.monotonic() - start_time
        results.termination_reason = termination_reason

        self._log_final_results(results)
        return results
```

---

### `_initial_execution`, `_main_loop`

```python
    def _initial_execution(self, initial_input: Any) -> Optional[ExecutionState]:
        """
        執行初始具體運行，建立根狀態並加入 Searcher 和 StateManager。
        
        Returns:
            根 ExecutionState，失敗時返回 None
        """
        if self._verbose >= 1:
            logger.info("[EXPLORE] Starting initial execution")

        try:
            root_state = self._executor.execute_initial(initial_input)
        except (ExecutionError, ExecutionTimeout) as e:
            logger.error("[EXPLORE] Initial execution failed: %s", e)
            return None

        self._state_manager.add(root_state)
        self._searcher.update(None, added=[root_state], removed=[])

        if self._verbose >= 2:
            logger.debug("[EXPLORE] Root state created: %s", root_state)

        return root_state

    def _main_loop(self, root_state: ExecutionState, start_time: float) -> str:
        """
        主探索循環：不斷從 Searcher 選取狀態，執行，更新。
        
        Returns:
            終止原因字符串
        """
        iteration = 0

        while not self._searcher.empty():
            # 超時檢查
            if self._is_timed_out(start_time):
                logger.info("[EXPLORE] Global timeout reached at iteration %d", iteration)
                return "timeout"

            # 最大迭代次數檢查
            if self._max_iterations is not None and iteration >= self._max_iterations:
                logger.info("[EXPLORE] Max iterations (%d) reached", self._max_iterations)
                return "max_iterations"

            # 選擇下一個狀態
            current_state = self._searcher.select_state()
            current_state.status = "exploring"

            if self._verbose >= 2:
                logger.debug(
                    "[EXPLORE] iter=%d state=%s depth=%d",
                    iteration, current_state.path_id, current_state.depth
                )

            # 執行符號計算，生成後繼狀態
            try:
                successors = self._executor.execute(current_state)
            except Exception as e:
                logger.error("[EXPLORE] Executor error at iter=%d: %s", iteration, e)
                successors = []

            # 對抗樣本檢測
            adversarial_found = self._detect_adversarial(successors)

            if adversarial_found and self._stop_on_first:
                self._state_manager.add_batch(successors)
                self._searcher.update(current_state, added=successors, removed=[])
                logger.info("[EXPLORE] Adversarial found, stopping early at iter=%d", iteration)
                return "adversarial_found"

            # 更新狀態管理器
            current_state.status = "done"
            self._state_manager.add_batch(successors)
            self._searcher.update(current_state, added=successors, removed=[current_state])

            # 進度日誌
            iteration += 1
            if self._verbose >= 1 and iteration % self._log_interval == 0:
                self._log_progress(iteration, start_time)

        return "exhausted"
```

---

### 超時檢查、對抗樣本檢測

```python
    def _is_timed_out(self, start_time: float) -> bool:
        """
        檢查是否超過全局超時限制。
        
        Args:
            start_time: 探索開始的單調時間戳
            
        Returns:
            True 表示已超時
        """
        if self._timeout <= 0:
            return False
        return (time.monotonic() - start_time) >= self._timeout

    def _detect_adversarial(self, states: List[ExecutionState]) -> bool:
        """
        檢測後繼狀態中是否包含對抗樣本。
        
        對抗樣本條件：模型對 state.input_data 的預測類別
        與原始輸入的預測類別不同。
        
        Args:
            states: 後繼狀態列表
            
        Returns:
            True 表示發現至少一個對抗樣本
        """
        if self._original_label is None:
            return False

        found = False
        for state in states:
            if state.output is None:
                continue
            
            predicted_label = self._get_predicted_label(state.output)
            if predicted_label != self._original_label:
                state.mark_adversarial()
                if self._verbose >= 1:
                    logger.info(
                        "[EXPLORE] Adversarial found: state=%s orig=%d pred=%d",
                        state.path_id, self._original_label, predicted_label
                    )
                found = True

        return found

    @staticmethod
    def _get_predicted_label(output: Any) -> int:
        """
        從模型輸出提取預測標籤。
        支援 softmax 輸出（取 argmax）和直接整數輸出。
        """
        import numpy as np
        if isinstance(output, int):
            return output
        arr = np.asarray(output)
        return int(np.argmax(arr))
```

---

### 進度日誌、結果收集

```python
    def _log_progress(self, iteration: int, start_time: float) -> None:
        """輸出探索進度日誌。"""
        elapsed = time.monotonic() - start_time
        stats = self._state_manager.stats()
        pending = len(self._searcher)

        logger.info(
            "[EXPLORE] iter=%d elapsed=%.1fs pending=%d "
            "explored=%d adversarial=%d alpha=%.3f",
            iteration,
            elapsed,
            pending,
            stats["total_completed"],
            stats["adversarial_count"],
            getattr(self._searcher, "alpha", float("nan")),
        )

    def _log_final_results(self, results: ExplorationResults) -> None:
        """輸出最終探索結果。"""
        if self._verbose < 1:
            return
        logger.info(
            "[EXPLORE] DONE reason=%s adversarial=%d states=%d elapsed=%.1fs",
            results.termination_reason,
            len(results.adversarial_inputs),
            results.total_states_explored,
            results.elapsed_seconds,
        )
```

---

## ExplorationResults 數據類

```python
@dataclass
class ExplorationResults:
    adversarial_inputs: List[Any] = field(default_factory=list)
    total_states_explored: int = 0
    total_iterations: int = 0
    elapsed_seconds: float = 0.0
    termination_reason: str = "unknown"
    state_manager_stats: dict = field(default_factory=dict)

    def found_adversarial(self) -> bool:
        return len(self.adversarial_inputs) > 0
```

---

## 完整的執行流程註解

```
explore(initial_input)
│
├─ [1] _initial_execution(initial_input)
│   ├─ executor.execute_initial(input)  ← 具體執行，建立根狀態
│   ├─ state_manager.add(root)
│   └─ searcher.update(None, added=[root], removed=[])
│
└─ [2] _main_loop(root, start_time)
    │
    ├─ WHILE searcher.empty() == False:
    │   │
    │   ├─ [超時檢查] time.monotonic() - start >= timeout?  → return "timeout"
    │   ├─ [迭代上限] iteration >= max_iterations?           → return "max_iterations"
    │   │
    │   ├─ current = searcher.select_state()                 ← O(log n)
    │   ├─ successors = executor.execute(current)            ← 子進程符號執行
    │   ├─ [對抗樣本] _detect_adversarial(successors)
    │   │   ├─ 發現且 stop_on_first → return "adversarial_found"
    │   │   └─ 繼續
    │   │
    │   ├─ state_manager.add_batch(successors)
    │   ├─ searcher.update(current, added=successors, removed=[current])
    │   └─ [進度日誌] iteration % log_interval == 0 → _log_progress()
    │
    └─ searcher.empty() → return "exhausted"
```

---

## 測試範例

```python
# test/integration/test_engine_with_searchers.py
from unittest.mock import MagicMock
import numpy as np

from libct.explore import ExplorationEngine
from libct.searcher.bfs import BFSSearcher
from libct.state.execution_state import ExecutionState
from libct.state.manager import StateManager


def make_mock_executor(root_output, successor_outputs=None):
    executor = MagicMock()
    root = ExecutionState(path_id="root", depth=0, output=root_output)
    executor.execute_initial.return_value = root
    
    if successor_outputs:
        successors = [
            ExecutionState(path_id=f"s{i}", depth=1, output=out)
            for i, out in enumerate(successor_outputs)
        ]
        executor.execute.return_value = successors
    else:
        executor.execute.return_value = []
    
    return executor


def test_engine_finds_adversarial():
    executor = make_mock_executor(
        root_output=np.array([0.9, 0.1]),     # label=0
        successor_outputs=[
            np.array([0.4, 0.6]),              # label=1（對抗樣本！）
        ]
    )
    
    engine = ExplorationEngine(
        searcher=BFSSearcher(),
        executor=executor,
        state_manager=StateManager(),
        original_label=0,
        timeout=10,
    )
    
    results = engine.explore(np.zeros((28, 28, 1)))
    assert results.found_adversarial()
    assert len(results.adversarial_inputs) == 1


def test_engine_stops_on_exhaustion():
    executor = make_mock_executor(
        root_output=np.array([0.9, 0.1]),
        successor_outputs=[]  # 沒有後繼狀態
    )
    
    engine = ExplorationEngine(
        searcher=BFSSearcher(),
        executor=executor,
        state_manager=StateManager(),
        original_label=0,
    )
    
    results = engine.explore(np.zeros((28, 28, 1)))
    assert results.termination_reason == "exhausted"
    assert not results.found_adversarial()
```

---

## 驗收標準

- [ ] `ExplorationEngine` 代碼行數 < 300 行（不含注釋）
- [ ] `explore()` 正確返回所有對抗樣本
- [ ] 超時機制在規定時間內終止（誤差 < 2 秒）
- [ ] `stop_on_first_adversarial=True` 時在發現第一個樣本後立即停止
- [ ] `ExplorationResults` 可被 `pickle` 序列化
- [ ] 集成測試覆蓋 4 種終止原因
- [ ] 日誌輸出格式與現有輸出兼容
