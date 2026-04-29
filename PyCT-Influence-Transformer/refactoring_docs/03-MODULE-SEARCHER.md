# 🔍 Searcher 模塊指南

## 概述和職責

**Searcher** 負責決定探索哪條執行路徑，是 KLEE 架構中最核心的設計之一。

**職責**：
- 維護待探索狀態的集合
- 根據策略選擇下一個要探索的狀態
- 根據執行結果更新狀態優先級

**不做的事**：
- 不執行任何符號計算
- 不修改狀態內容（只讀取用於排序）
- 不管理狀態的生命週期（由 `StateManager` 負責）

---

## 抽象基類設計

```python
# libct/searcher/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from libct.state.execution_state import ExecutionState


class Searcher(ABC):
    """
    路徑搜索策略的抽象基類，仿照 KLEE 的 Searcher 接口設計。
    
    參考：klee/lib/Core/Searcher.h
    """

    @abstractmethod
    def select_state(self) -> ExecutionState:
        """
        選擇下一個要探索的狀態。
        
        Returns:
            下一個要執行的 ExecutionState
            
        Raises:
            IndexError: 當沒有待探索的狀態時
        """
        ...

    @abstractmethod
    def update(
        self,
        current: Optional[ExecutionState],
        added: List[ExecutionState],
        removed: List[ExecutionState],
    ) -> None:
        """
        根據本次執行結果更新 Searcher 的內部狀態。
        
        Args:
            current: 剛剛執行完的狀態（可為 None，表示初始化）
            added: 本次執行後新增的後繼狀態
            removed: 本次執行後需要移除的狀態（如超時或已完成）
        """
        ...

    @abstractmethod
    def empty(self) -> bool:
        """
        返回是否還有待探索的狀態。
        
        Returns:
            True 表示探索完成，False 表示還有狀態待探索
        """
        ...

    def __len__(self) -> int:
        """返回當前待探索狀態數量（可選覆寫）。"""
        raise NotImplementedError
```

---

## 4 個具體實現

### 實現一：PrioritySearcher（SHAP-guided）

**策略**：使用 SHAP 分數與路徑深度的混合加權進行優先級排序，動態調整 α 權重。

```python
# libct/searcher/priority.py
from __future__ import annotations

import heapq
import itertools
import logging
from typing import List, Optional, Tuple

import numpy as np

from libct.searcher.base import Searcher
from libct.state.execution_state import ExecutionState

logger = logging.getLogger("ct.searcher.priority")


class PrioritySearcher(Searcher):
    """
    SHAP 導向的優先級搜索。
    
    分數計算：score = α * shap_score + (1 - α) * depth_score
    α 隨迭代動態增加（更依賴 SHAP 啟發）。
    
    參考：現有 libct/explore.py 中的 heapq 實現
    """

    def __init__(
        self,
        alpha: float = 0.5,
        alpha_schedule: str = "linear",
        alpha_increment: float = 0.05,
        alpha_max: float = 0.95,
        alpha_update_interval: int = 10,
    ) -> None:
        """
        Args:
            alpha: 初始 SHAP 權重（0.0 = 純深度優先，1.0 = 純 SHAP 優先）
            alpha_schedule: 調度策略（"linear" 或 "fixed"）
            alpha_increment: 每次更新的 α 增量
            alpha_max: α 的最大值
            alpha_update_interval: 每隔多少次更新調整一次 α
        """
        self._alpha = alpha
        self._alpha_schedule = alpha_schedule
        self._alpha_increment = alpha_increment
        self._alpha_max = alpha_max
        self._alpha_update_interval = alpha_update_interval
        self._iteration = 0
        self._counter = itertools.count()  # 打破分數相等時的平局
        self._heap: List[Tuple[float, int, ExecutionState]] = []

    @property
    def alpha(self) -> float:
        """當前有效的 α 值（考慮動態調度後）。"""
        if self._alpha_schedule == "linear":
            increments = self._iteration // self._alpha_update_interval
            return min(self._alpha_max, self._alpha + self._alpha_increment * increments)
        return self._alpha

    def _compute_score(self, state: ExecutionState) -> float:
        """計算狀態的優先級分數（越高越優先）。"""
        alpha = self.alpha

        # SHAP 分數：取最大絕對值的像素 SHAP 分數
        shap_component = 0.0
        if state.shap_scores is not None and len(state.shap_scores) > 0:
            shap_component = float(np.max(np.abs(state.shap_scores)))

        # 深度分數：路徑越深，後期越值得探索
        depth_component = state.depth / max(state.depth, 1)

        return alpha * shap_component + (1 - alpha) * depth_component

    def select_state(self) -> ExecutionState:
        if not self._heap:
            raise IndexError("PrioritySearcher: no states to select")
        _, _, state = heapq.heappop(self._heap)
        logger.debug("selected state depth=%d alpha=%.3f", state.depth, self.alpha)
        return state

    def update(
        self,
        current: Optional[ExecutionState],
        added: List[ExecutionState],
        removed: List[ExecutionState],
    ) -> None:
        # 移除已完成或超時的狀態（標記為無效）
        removed_ids = {id(s) for s in removed}
        self._heap = [
            entry for entry in self._heap if id(entry[2]) not in removed_ids
        ]
        heapq.heapify(self._heap)

        # 加入新的後繼狀態
        for state in added:
            score = self._compute_score(state)
            heapq.heappush(self._heap, (-score, next(self._counter), state))

        self._iteration += 1

    def empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)
```

---

### 實現二：DFSSearcher

**策略**：深度優先搜索，優先探索最新加入的狀態。

```python
# libct/searcher/dfs.py
from __future__ import annotations

from typing import List, Optional

from libct.searcher.base import Searcher
from libct.state.execution_state import ExecutionState


class DFSSearcher(Searcher):
    """
    深度優先搜索。後進先出（stack）。
    
    參考：KLEE lib/Core/Searcher.cpp DFSSearcher
    """

    def __init__(self) -> None:
        self._stack: List[ExecutionState] = []

    def select_state(self) -> ExecutionState:
        if not self._stack:
            raise IndexError("DFSSearcher: no states to select")
        return self._stack.pop()

    def update(
        self,
        current: Optional[ExecutionState],
        added: List[ExecutionState],
        removed: List[ExecutionState],
    ) -> None:
        removed_ids = {id(s) for s in removed}
        self._stack = [s for s in self._stack if id(s) not in removed_ids]
        self._stack.extend(added)

    def empty(self) -> bool:
        return len(self._stack) == 0

    def __len__(self) -> int:
        return len(self._stack)
```

---

### 實現三：BFSSearcher

**策略**：廣度優先搜索，先進先出（queue）。

```python
# libct/searcher/bfs.py
from __future__ import annotations

from collections import deque
from typing import List, Optional

from libct.searcher.base import Searcher
from libct.state.execution_state import ExecutionState


class BFSSearcher(Searcher):
    """
    廣度優先搜索。先進先出（deque）。
    對應現有代碼中的 collect_constraints_with="queue" 模式。
    
    參考：KLEE lib/Core/Searcher.cpp BFSSearcher
    """

    def __init__(self) -> None:
        self._queue: deque[ExecutionState] = deque()

    def select_state(self) -> ExecutionState:
        if not self._queue:
            raise IndexError("BFSSearcher: no states to select")
        return self._queue.popleft()

    def update(
        self,
        current: Optional[ExecutionState],
        added: List[ExecutionState],
        removed: List[ExecutionState],
    ) -> None:
        removed_ids = {id(s) for s in removed}
        filtered = deque(s for s in self._queue if id(s) not in removed_ids)
        self._queue = filtered
        self._queue.extend(added)

    def empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)
```

---

### 實現四：RandomSearcher

**策略**：隨機選擇待探索狀態，均勻覆蓋路徑空間。

```python
# libct/searcher/random_searcher.py
from __future__ import annotations

import random
from typing import List, Optional

from libct.searcher.base import Searcher
from libct.state.execution_state import ExecutionState


class RandomSearcher(Searcher):
    """
    隨機路徑搜索。提供均勻的路徑覆蓋。
    
    參考：KLEE lib/Core/Searcher.cpp RandomSearcher
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._states: List[ExecutionState] = []
        self._rng = random.Random(seed)

    def select_state(self) -> ExecutionState:
        if not self._states:
            raise IndexError("RandomSearcher: no states to select")
        idx = self._rng.randrange(len(self._states))
        # swap-and-pop：O(1) 刪除
        self._states[idx], self._states[-1] = self._states[-1], self._states[idx]
        return self._states.pop()

    def update(
        self,
        current: Optional[ExecutionState],
        added: List[ExecutionState],
        removed: List[ExecutionState],
    ) -> None:
        removed_ids = {id(s) for s in removed}
        self._states = [s for s in self._states if id(s) not in removed_ids]
        self._states.extend(added)

    def empty(self) -> bool:
        return len(self._states) == 0

    def __len__(self) -> int:
        return len(self._states)
```

---

## 驗收標準

### 功能測試

```python
# test/unit/test_searcher_priority.py
import numpy as np
from libct.searcher.priority import PrioritySearcher
from libct.state.execution_state import ExecutionState


def make_state(depth: int, shap: float) -> ExecutionState:
    state = ExecutionState(path_id=f"path_{depth}", depth=depth)
    state.shap_scores = np.array([shap])
    return state


def test_priority_searcher_selects_highest_score():
    searcher = PrioritySearcher(alpha=1.0)  # 純 SHAP 模式
    low = make_state(depth=1, shap=0.1)
    high = make_state(depth=1, shap=0.9)
    searcher.update(None, added=[low, high], removed=[])
    assert searcher.select_state() is high


def test_alpha_increases_linearly():
    searcher = PrioritySearcher(alpha=0.5, alpha_schedule="linear",
                                alpha_increment=0.05, alpha_update_interval=1)
    assert searcher.alpha == 0.5
    searcher.update(None, added=[], removed=[])  # iteration = 1
    assert abs(searcher.alpha - 0.55) < 1e-9


def test_empty_raises_on_select():
    searcher = PrioritySearcher()
    assert searcher.empty()
    try:
        searcher.select_state()
        assert False, "Should raise IndexError"
    except IndexError:
        pass
```

### 性能測試

- `select_state()` 1000 次調用應在 10ms 以內（1000 個狀態）
- `update()` 批量添加 100 個狀態應在 1ms 以內

---

## 相關文件連結

| 文件 | 說明 |
|------|------|
| `libct/explore.py` | 現有的 `ExplorationEngine`（重構來源） |
| `libct/state/execution_state.py` | `ExecutionState` 定義 |
| [02-ARCHITECTURE.md](./02-ARCHITECTURE.md) | 架構總覽 |
| [08-KLEE-REFERENCE.md](./08-KLEE-REFERENCE.md) | KLEE Searcher 原始接口 |
