# ⚙️ Executor 模塊指南

## 概述和職責

**Executor** 負責執行目標函數，生成新的執行狀態。它是符號執行引擎的"執行臂"。

**職責**：
- 執行初始（具體）運行，建立根狀態
- 對給定狀態進行符號執行，生成後繼狀態
- 處理超時和異常，確保主進程穩定性
- 收集覆蓋率信息

**不做的事**：
- 不決定執行順序（由 Searcher 負責）
- 不存儲歷史執行結果（由 StateManager 負責）
- 不修改探索策略

---

## 抽象基類設計

```python
# libct/executor/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from libct.state.execution_state import ExecutionState


class Executor(ABC):
    """
    執行邏輯的抽象基類。
    
    每個 Executor 實現封裝一種執行方式（符號執行或具體執行），
    並在獨立子進程中運行以保證隔離性。
    """

    @abstractmethod
    def execute_initial(self, input_data: Any) -> ExecutionState:
        """
        執行初始具體運行，建立根狀態。
        
        初始運行使用實際輸入值（非符號值），目的是：
        1. 驗證輸入的合法性
        2. 建立初始覆蓋率基線
        3. 收集 SHAP 分數（如果適用）
        
        Args:
            input_data: 初始輸入數據（如圖像的 numpy array）
            
        Returns:
            根 ExecutionState，包含初始路徑約束和 SHAP 分數
            
        Raises:
            ExecutionError: 執行失敗時
            ExecutionTimeout: 超時時
        """
        ...

    @abstractmethod
    def execute(self, state: ExecutionState) -> List[ExecutionState]:
        """
        從給定狀態進行符號執行，生成後繼狀態。
        
        符號執行會：
        1. 重放 state 的路徑約束到達當前分支點
        2. 對當前分支條件取反，生成新約束
        3. 調用 SMT 求解器求解新約束
        4. 返回所有可達的後繼狀態
        
        Args:
            state: 當前要探索的 ExecutionState
            
        Returns:
            後繼狀態列表（可能為空，表示路徑終止）
            
        Raises:
            ExecutionError: 執行失敗時
            ExecutionTimeout: 超時時
        """
        ...


class ExecutionError(Exception):
    """符號執行過程中發生的錯誤。"""
    pass


class ExecutionTimeout(Exception):
    """符號執行超時。"""
    pass
```

---

## 2 個具體實現

### 實現一：SymbolicExecutor（符號執行）

**職責**：包裝現有的 `libct.explore.ExplorationEngine` 的符號執行邏輯，在獨立子進程中執行。

```python
# libct/executor/symbolic.py
from __future__ import annotations

import logging
import multiprocessing
import pickle
import time
from typing import Any, Callable, List, Optional

from libct.executor.base import Executor, ExecutionError, ExecutionTimeout
from libct.path import PathToConstraint
from libct.solver import Solver
from libct.state.execution_state import ExecutionState
from libct.utils import ConcolicObject

logger = logging.getLogger("ct.executor.symbolic")


class SymbolicExecutor(Executor):
    """
    符號執行器。在獨立子進程中執行符號計算，避免內存洩漏影響主進程。
    
    內部使用 libct.concolic 的符號值類型和 cvc5/z3 求解器。
    """

    def __init__(
        self,
        module: Any,
        execute_fn: Callable,
        solver: Solver,
        timeout: int = 20,
        constraint_build_timeout: bool = True,
        constraint_build_timeout_seconds: int = 30,
        solver_run_timeout: Optional[int] = 60,
        norm_01: bool = True,
    ) -> None:
        self._module = module
        self._execute_fn = execute_fn
        self._solver = solver
        self._timeout = timeout
        self._constraint_build_timeout = constraint_build_timeout
        self._constraint_build_timeout_seconds = constraint_build_timeout_seconds
        self._solver_run_timeout = solver_run_timeout
        self._norm_01 = norm_01

    def execute_initial(self, input_data: Any) -> ExecutionState:
        """
        執行初始具體運行（不使用符號值）。
        
        建立初始路徑約束，收集 SHAP 分數。
        對應現有代碼的 _initial_execution 邏輯。
        """
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=self._initial_worker,
            args=(input_data, result_queue),
        )
        p.start()
        p.join(timeout=self._timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            raise ExecutionTimeout(f"Initial execution timed out after {self._timeout}s")

        if result_queue.empty():
            raise ExecutionError("Initial execution produced no result")

        result = result_queue.get_nowait()
        if isinstance(result, Exception):
            raise ExecutionError(f"Initial execution failed: {result}") from result

        return result

    def execute(self, state: ExecutionState) -> List[ExecutionState]:
        """
        符號執行：對給定狀態取反約束，求解後繼狀態。
        
        對應現有代碼的 _one_iteration 或 explore_one_path 邏輯。
        """
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=self._symbolic_worker,
            args=(state, result_queue),
        )
        p.start()

        timeout = self._constraint_build_timeout_seconds if self._constraint_build_timeout else self._timeout
        p.join(timeout=timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            logger.warning("Symbolic execution timed out for state %s", state.path_id)
            return []  # 超時視為無後繼

        if result_queue.empty():
            return []

        result = result_queue.get_nowait()
        if isinstance(result, Exception):
            logger.error("Symbolic execution error for state %s: %s", state.path_id, result)
            return []

        return result

    def _initial_worker(self, input_data: Any, result_queue: multiprocessing.Queue) -> None:
        """子進程：執行初始具體運行。"""
        try:
            path = PathToConstraint(lambda c: None)
            # 具體執行，收集路徑約束
            output = self._execute_fn(self._module, input_data)
            
            state = ExecutionState(
                path_id="root",
                depth=0,
                input_data=input_data,
                output=output,
                path_constraint=path,
            )
            result_queue.put(state)
        except Exception as e:
            result_queue.put(e)

    def _symbolic_worker(
        self, state: ExecutionState, result_queue: multiprocessing.Queue
    ) -> None:
        """子進程：執行符號計算，生成後繼狀態。"""
        try:
            successors: List[ExecutionState] = []
            
            # 對每個約束取反，嘗試求解
            for i, constraint in enumerate(state.constraints):
                negated = constraint.negate()
                new_constraints = state.constraints[:i] + [negated]
                
                # 調用求解器
                solution = self._solver.solve(new_constraints, timeout=self._solver_run_timeout)
                if solution is None:
                    continue  # 不可滿足，跳過

                # 創建後繼狀態
                successor = ExecutionState(
                    path_id=f"{state.path_id}_{i}",
                    depth=state.depth + 1,
                    input_data=solution,
                    constraints=new_constraints,
                    parent_id=state.path_id,
                )
                successors.append(successor)
            
            result_queue.put(successors)
        except Exception as e:
            result_queue.put(e)
```

---

### 實現二：ConcreteExecutor（覆蓋收集）

**職責**：使用具體輸入執行函數，收集代碼覆蓋率，用於評估測試品質。

```python
# libct/executor/concrete.py
from __future__ import annotations

import logging
from typing import Any, List, Optional

import coverage

from libct.executor.base import Executor, ExecutionError
from libct.state.execution_state import ExecutionState

logger = logging.getLogger("ct.executor.concrete")


class ConcreteExecutor(Executor):
    """
    具體執行器。使用真實輸入值執行函數，收集覆蓋率信息。
    
    主要用於：
    1. 驗證對抗樣本是否真正改變了模型預測
    2. 收集代碼覆蓋率指標
    3. 生成執行報告
    """

    def __init__(
        self,
        module: Any,
        execute_fn: Any,
        collect_coverage: bool = True,
        coverage_source: Optional[str] = None,
    ) -> None:
        self._module = module
        self._execute_fn = execute_fn
        self._collect_coverage = collect_coverage
        self._coverage_source = coverage_source

    def execute_initial(self, input_data: Any) -> ExecutionState:
        """執行初始具體運行，同時收集基線覆蓋率。"""
        return self._run_with_coverage(input_data, path_id="root", depth=0)

    def execute(self, state: ExecutionState) -> List[ExecutionState]:
        """
        具體執行給定狀態的輸入數據。
        
        注意：ConcreteExecutor 不進行路徑分叉，只返回單個後繼狀態
        （如果執行成功）或空列表（如果失敗）。
        """
        if state.input_data is None:
            return []
        
        try:
            result_state = self._run_with_coverage(
                input_data=state.input_data,
                path_id=f"{state.path_id}_concrete",
                depth=state.depth + 1,
            )
            return [result_state]
        except ExecutionError as e:
            logger.error("ConcreteExecutor failed for state %s: %s", state.path_id, e)
            return []

    def _run_with_coverage(
        self, input_data: Any, path_id: str, depth: int
    ) -> ExecutionState:
        """執行並收集覆蓋率。"""
        cov = coverage.Coverage(source=[self._coverage_source]) if self._collect_coverage else None
        
        if cov:
            cov.start()
        
        try:
            output = self._execute_fn(self._module, input_data)
        except Exception as e:
            raise ExecutionError(f"Concrete execution failed: {e}") from e
        finally:
            if cov:
                cov.stop()

        covered_lines = set()
        if cov:
            try:
                covered_lines = self._extract_covered_lines(cov)
            except Exception:
                logger.warning("Failed to extract coverage data")

        return ExecutionState(
            path_id=path_id,
            depth=depth,
            input_data=input_data,
            output=output,
            coverage=covered_lines,
        )

    @staticmethod
    def _extract_covered_lines(cov: coverage.Coverage) -> set:
        """從 coverage 對象提取覆蓋的行號集合。"""
        data = cov.get_data()
        covered = set()
        for filename in data.measured_files():
            for line in data.lines(filename) or []:
                covered.add(f"{filename}:{line}")
        return covered
```

---

## 單進程隔離的設計

每次 `execute()` 調用都在獨立子進程中運行：

```
主進程                          子進程
   │                               │
   │  multiprocessing.Process()    │
   │ ─────────────────────────→   │
   │                               │  符號執行
   │                               │  （可能崩潰/洩漏）
   │  result_queue.put(result)     │
   │ ←─────────────────────────   │
   │                               │  子進程退出
   │                               ×
   │  繼續（主進程安全）
```

**好處**：
- 符號執行的內存洩漏不影響主進程
- 子進程崩潰不終止整個探索循環
- 可以精確控制超時（`p.join(timeout=N)`）

---

## 異常和超時處理

```python
# 在 ExplorationEngine 中的使用方式
try:
    successors = executor.execute(current_state)
except ExecutionTimeout:
    logger.warning("State %s timed out, skipping", current_state.path_id)
    successors = []
except ExecutionError as e:
    logger.error("State %s execution error: %s", current_state.path_id, e)
    successors = []
```

---

## 測試範例

```python
# test/unit/test_executor_symbolic.py
from unittest.mock import MagicMock
from libct.executor.symbolic import SymbolicExecutor
from libct.state.execution_state import ExecutionState
import numpy as np


def test_execute_initial_returns_root_state():
    module = MagicMock()
    execute_fn = MagicMock(return_value=np.array([0.9, 0.1]))
    solver = MagicMock()

    executor = SymbolicExecutor(module=module, execute_fn=execute_fn, solver=solver, timeout=5)
    input_data = np.zeros((28, 28, 1))

    state = executor.execute_initial(input_data)
    assert state.path_id == "root"
    assert state.depth == 0
    assert state.input_data is not None


def test_execute_returns_empty_on_timeout():
    module = MagicMock()
    execute_fn = MagicMock(side_effect=lambda *_: __import__("time").sleep(100))
    solver = MagicMock()

    executor = SymbolicExecutor(module=module, execute_fn=execute_fn, solver=solver,
                                 timeout=1, constraint_build_timeout_seconds=1)
    root = ExecutionState(path_id="root", depth=0)

    result = executor.execute(root)
    assert result == []
```

---

## 驗收標準

- [ ] `SymbolicExecutor.execute_initial()` 返回合法的根狀態
- [ ] `SymbolicExecutor.execute()` 在超時時返回空列表（不崩潰）
- [ ] `ConcreteExecutor` 正確收集覆蓋率數據
- [ ] 子進程崩潰不影響主進程
- [ ] 所有方法有類型提示
- [ ] 單元測試覆蓋率 > 80%
