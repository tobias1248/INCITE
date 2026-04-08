# 📊 State 管理模塊指南

## 概述和職責

**State 模塊**包含兩個核心組件：

1. **`ExecutionState`**：單個執行路徑的完整數據模型（`@dataclass`）
2. **`StateManager`**：管理所有活躍狀態的集合（線程安全）

**職責**：
- 結構化存儲每條執行路徑的所有相關信息
- 提供線程安全的狀態增刪查操作
- 管理狀態的完整生命週期（創建 → 活躍 → 完成 / 超時）

**不做的事**：
- 不選擇要執行哪個狀態（由 Searcher 負責）
- 不執行符號計算（由 Executor 負責）
- 不決定探索終止條件（由 Engine 負責）

---

## ExecutionState @dataclass 定義

### 必填字段和可選字段

```python
# libct/state/execution_state.py
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from libct.constraint import Constraint
from libct.path import PathToConstraint


@dataclass
class ExecutionState:
    """
    單個符號執行路徑的完整狀態。
    
    參考：KLEE lib/Core/ExecutionState.h
    
    設計原則：
    - 所有字段都有明確的類型提示
    - 可被 pickle 序列化（多進程需要）
    - 不包含業務邏輯，只存儲數據
    """

    # ─── 必填字段 ──────────────────────────────────────────
    path_id: str = field(
        default_factory=lambda: str(uuid.uuid4())[:8]
    )
    """唯一路徑標識符，格式：根節點為 'root'，後繼節點為 '{parent_id}_{branch_idx}'。"""

    depth: int = 0
    """路徑深度（從根節點起的分支次數）。"""

    # ─── 可選字段 ──────────────────────────────────────────
    input_data: Optional[Any] = None
    """當前路徑的輸入數據（如 numpy array，對應特定像素值）。"""

    output: Optional[Any] = None
    """執行函數的輸出（如模型預測結果）。"""

    parent_id: Optional[str] = None
    """父狀態的 path_id，根節點為 None。"""

    created_at: float = field(default_factory=time.monotonic)
    """狀態創建的單調時間戳（秒）。"""

    # ─── 約束管理字段 ───────────────────────────────────────
    constraints: List[Constraint] = field(default_factory=list)
    """從根節點到此狀態的完整路徑約束列表（有序）。"""

    path_constraint: Optional[PathToConstraint] = None
    """路徑約束的結構化表示（用於求解器）。"""

    # ─── SHAP 相關字段 ──────────────────────────────────────
    shap_scores: Optional[np.ndarray] = None
    """每個輸入特徵（像素）的 SHAP 值，形狀與 input_data 相同。"""

    shap_computed: bool = False
    """標記 SHAP 分數是否已計算（避免重複計算）。"""

    # ─── 覆蓋信息字段 ──────────────────────────────────────
    coverage: Optional[Set[str]] = None
    """此路徑覆蓋的代碼位置集合，格式：'{filename}:{line_number}'。"""

    # ─── 元數據字段 ────────────────────────────────────────
    metadata: Dict[str, Any] = field(default_factory=dict)
    """自由格式的元數據字典，用於存儲額外信息（如執行統計）。"""

    status: str = "pending"
    """
    狀態生命週期標記：
    - 'pending'：等待探索
    - 'exploring'：正在探索
    - 'done'：探索完成
    - 'timeout'：超時終止
    - 'error'：執行出錯
    - 'adversarial'：發現對抗樣本
    """
```

---

### 約束管理方法

```python
# libct/state/execution_state.py（續）

    def add_constraint(self, constraint: Constraint) -> None:
        """追加一個新的路徑約束。"""
        self.constraints.append(constraint)

    def get_constraint_count(self) -> int:
        """返回當前約束數量（等於路徑深度）。"""
        return len(self.constraints)

    def has_unsatisfied_constraints(self) -> bool:
        """
        檢查是否存在尚未求解的約束。
        用於判斷此狀態是否需要調用求解器。
        """
        return any(not c.is_solved for c in self.constraints)

    def fork(self, branch_idx: int, negated_constraint: Constraint) -> "ExecutionState":
        """
        從此狀態分叉，創建一個對第 branch_idx 個約束取反的後繼狀態。
        
        Args:
            branch_idx: 要取反的約束索引
            negated_constraint: 取反後的約束對象
            
        Returns:
            新的後繼 ExecutionState（深度 + 1）
        """
        new_constraints = self.constraints[:branch_idx] + [negated_constraint]
        return ExecutionState(
            path_id=f"{self.path_id}_{branch_idx}",
            depth=self.depth + 1,
            parent_id=self.path_id,
            constraints=new_constraints,
        )
```

---

### 元數據管理方法

```python
    def set_metadata(self, key: str, value: Any) -> None:
        """設置元數據鍵值對。"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """獲取元數據，鍵不存在時返回默認值。"""
        return self.metadata.get(key, default)

    def record_execution_time(self, elapsed_seconds: float) -> None:
        """記錄執行耗時。"""
        self.set_metadata("execution_time_s", elapsed_seconds)

    def get_execution_time(self) -> Optional[float]:
        """獲取執行耗時（秒），未記錄時返回 None。"""
        return self.get_metadata("execution_time_s")
```

---

### SHAP 相關方法

```python
    def set_shap_scores(self, scores: np.ndarray) -> None:
        """設置 SHAP 分數並標記已計算。"""
        self.shap_scores = scores
        self.shap_computed = True

    def get_max_shap_score(self) -> float:
        """
        獲取最大絕對 SHAP 分數。
        未計算時返回 0.0。
        """
        if self.shap_scores is None or len(self.shap_scores) == 0:
            return 0.0
        return float(np.max(np.abs(self.shap_scores)))

    def get_top_k_shap_indices(self, k: int) -> List[int]:
        """
        返回 SHAP 分數最高的前 k 個特徵索引（按絕對值降序）。
        
        Args:
            k: 要返回的特徵數量
            
        Returns:
            特徵索引列表（降序），長度 min(k, len(shap_scores))
        """
        if self.shap_scores is None:
            return []
        flat_scores = np.abs(self.shap_scores).flatten()
        top_k = min(k, len(flat_scores))
        return list(np.argsort(flat_scores)[::-1][:top_k])
```

---

### 覆蓋信息方法

```python
    def add_coverage(self, lines: Set[str]) -> None:
        """添加覆蓋的代碼行（合併到現有集合）。"""
        if self.coverage is None:
            self.coverage = set()
        self.coverage.update(lines)

    def coverage_count(self) -> int:
        """返回覆蓋的代碼行數。"""
        return len(self.coverage) if self.coverage else 0

    def is_adversarial(self) -> bool:
        """
        判斷此狀態的輸入是否為對抗樣本。
        
        對抗樣本條件：輸出的預測類別與初始輸入不同。
        具體判斷邏輯由 Engine 在執行後設置 status='adversarial'。
        """
        return self.status == "adversarial"

    def mark_adversarial(self) -> None:
        """標記此狀態為對抗樣本。"""
        self.status = "adversarial"

    def __repr__(self) -> str:
        return (
            f"ExecutionState(id={self.path_id!r}, depth={self.depth}, "
            f"status={self.status!r}, constraints={len(self.constraints)})"
        )
```

---

## StateManager 實現

### 線程安全的狀態管理

```python
# libct/state/manager.py
from __future__ import annotations

import logging
import threading
from typing import Dict, Iterator, List, Optional

from libct.state.execution_state import ExecutionState

logger = logging.getLogger("ct.state.manager")


class StateManager:
    """
    管理所有活躍執行狀態的集合。
    
    提供線程安全的增刪查操作，支持按狀態篩選查詢。
    
    設計說明：
    - 使用 threading.Lock 保護內部狀態（進程內多線程安全）
    - 不需要跨進程鎖（每個 worker 進程有獨立的 StateManager）
    - 通過 path_id 快速查找（O(1) dict 查找）
    """

    def __init__(self) -> None:
        self._states: Dict[str, ExecutionState] = {}
        self._lock = threading.Lock()
        self._total_created = 0
        self._total_completed = 0

    def add(self, state: ExecutionState) -> None:
        """
        添加一個新狀態。
        
        Args:
            state: 要添加的 ExecutionState
            
        Raises:
            ValueError: 如果 path_id 已存在
        """
        with self._lock:
            if state.path_id in self._states:
                raise ValueError(
                    f"StateManager: path_id '{state.path_id}' already exists"
                )
            self._states[state.path_id] = state
            self._total_created += 1
            logger.debug("Added state %s (total: %d)", state.path_id, len(self._states))

    def add_batch(self, states: List[ExecutionState]) -> None:
        """批量添加狀態（單鎖操作，更高效）。"""
        with self._lock:
            for state in states:
                if state.path_id not in self._states:
                    self._states[state.path_id] = state
                    self._total_created += 1

    def get(self, path_id: str) -> Optional[ExecutionState]:
        """根據 path_id 獲取狀態，不存在時返回 None。"""
        with self._lock:
            return self._states.get(path_id)

    def remove(self, path_id: str) -> Optional[ExecutionState]:
        """
        移除並返回指定狀態。
        
        Returns:
            被移除的狀態，不存在時返回 None
        """
        with self._lock:
            state = self._states.pop(path_id, None)
            if state is not None:
                self._total_completed += 1
            return state

    def remove_batch(self, path_ids: List[str]) -> int:
        """批量移除狀態，返回實際移除的數量。"""
        count = 0
        with self._lock:
            for pid in path_ids:
                if self._states.pop(pid, None) is not None:
                    count += 1
                    self._total_completed += 1
        return count
```

---

### 生命週期管理

```python
    def get_by_status(self, status: str) -> List[ExecutionState]:
        """獲取所有指定狀態的 ExecutionState 列表（快照）。"""
        with self._lock:
            return [s for s in self._states.values() if s.status == status]

    def get_adversarial_states(self) -> List[ExecutionState]:
        """獲取所有對抗樣本狀態。"""
        return self.get_by_status("adversarial")

    def get_all(self) -> List[ExecutionState]:
        """獲取所有當前活躍狀態的快照。"""
        with self._lock:
            return list(self._states.values())

    def count(self) -> int:
        """返回當前活躍狀態數量。"""
        with self._lock:
            return len(self._states)

    def clear(self) -> None:
        """清除所有狀態（用於重置或清理）。"""
        with self._lock:
            self._states.clear()
            logger.info("StateManager cleared")

    def stats(self) -> dict:
        """
        返回當前統計信息。
        
        Returns:
            包含 total_created, total_completed, active_count,
            adversarial_count 的字典
        """
        with self._lock:
            adversarial = sum(
                1 for s in self._states.values() if s.status == "adversarial"
            )
            return {
                "total_created": self._total_created,
                "total_completed": self._total_completed,
                "active_count": len(self._states),
                "adversarial_count": adversarial,
            }

    def __iter__(self) -> Iterator[ExecutionState]:
        """迭代當前所有活躍狀態的快照（非實時）。"""
        return iter(self.get_all())

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"StateManager(active={stats['active_count']}, "
            f"created={stats['total_created']}, "
            f"adversarial={stats['adversarial_count']})"
        )
```

---

## 測試範例

```python
# test/unit/test_state_manager.py
import threading
from libct.state.execution_state import ExecutionState
from libct.state.manager import StateManager


def test_add_and_get():
    mgr = StateManager()
    state = ExecutionState(path_id="root", depth=0)
    mgr.add(state)
    
    retrieved = mgr.get("root")
    assert retrieved is state
    assert mgr.count() == 1


def test_remove_returns_state():
    mgr = StateManager()
    state = ExecutionState(path_id="s1", depth=1)
    mgr.add(state)
    
    removed = mgr.remove("s1")
    assert removed is state
    assert mgr.count() == 0
    assert mgr.stats()["total_completed"] == 1


def test_duplicate_path_id_raises():
    mgr = StateManager()
    state = ExecutionState(path_id="dup", depth=0)
    mgr.add(state)
    
    try:
        mgr.add(ExecutionState(path_id="dup", depth=1))
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_thread_safety():
    """並發添加 1000 個狀態，不應丟失或重複。"""
    mgr = StateManager()
    threads = []
    
    def add_states(offset: int):
        for i in range(100):
            mgr.add(ExecutionState(path_id=f"state_{offset}_{i}", depth=0))
    
    for i in range(10):
        t = threading.Thread(target=add_states, args=(i * 100,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    assert mgr.count() == 1000


def test_shap_methods():
    import numpy as np
    state = ExecutionState(path_id="s", depth=0)
    state.set_shap_scores(np.array([0.1, 0.5, 0.3, 0.9, 0.2]))
    
    assert state.shap_computed
    assert abs(state.get_max_shap_score() - 0.9) < 1e-9
    assert state.get_top_k_shap_indices(2) == [3, 1]  # 0.9, 0.5
```

---

## 驗收標準

- [ ] `ExecutionState` 可被 `pickle.dumps()` 序列化
- [ ] `StateManager` 在 1000 個並發添加操作中無數據競爭
- [ ] `get_top_k_shap_indices()` 返回正確的降序索引
- [ ] `fork()` 正確繼承約束並增加深度
- [ ] `stats()` 正確跟蹤 total_created 和 total_completed
- [ ] 所有字段有文檔字符串
- [ ] 單元測試覆蓋率 > 90%
