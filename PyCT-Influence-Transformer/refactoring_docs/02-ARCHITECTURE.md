# 🏗️ 目標架構設計

## 完整層次結構圖

```
PyCT-Influence-Transformer/
├── pyct/                          # CLI 入口層（不變）
│   ├── main.py
│   ├── args.py
│   └── config.py
│
├── orchestration/                 # 多進程協調層（不變）
│   ├── launcher.py
│   ├── runners.py
│   └── progress.py
│
├── engine/                        # 執行器適配層（微調）
│   └── executor.py                # 包裝 libct 模塊
│
└── libct/                         # 核心符號執行層（本次重構重點）
    ├── searcher/                  # 🔍 路徑選擇策略（新增）
    │   ├── __init__.py
    │   ├── base.py                # Searcher 抽象基類
    │   ├── priority.py            # PrioritySearcher（SHAP-guided）
    │   ├── dfs.py                 # DFSSearcher
    │   ├── bfs.py                 # BFSSearcher
    │   └── random_searcher.py     # RandomSearcher
    │
    ├── state/                     # 📊 狀態管理（新增）
    │   ├── __init__.py
    │   ├── execution_state.py     # ExecutionState dataclass
    │   └── manager.py             # StateManager
    │
    ├── executor/                  # ⚙️ 執行邏輯（重構）
    │   ├── __init__.py
    │   ├── base.py                # Executor 抽象基類
    │   ├── symbolic.py            # SymbolicExecutor
    │   └── concrete.py            # ConcreteExecutor
    │
    ├── explore.py                 # 🎛️ ExplorationEngine（精簡至 < 300 行）
    ├── solver.py                  # 不變
    ├── constraint.py              # 不變
    ├── path.py                    # 不變
    └── concolic/                  # 不變
```

---

## 模塊間關係（單向依賴流）

```
┌─────────────────────────────────────────────────────────┐
│                    ExplorationEngine                    │
│                  (libct/explore.py)                     │
└──────┬────────────────┬──────────────┬──────────────────┘
       │                │              │
       ▼                ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌────────────────┐
│   Searcher   │ │   Executor   │ │  StateManager  │
│  (抽象基類)   │ │  (抽象基類)   │ │                │
└──────┬───────┘ └──────┬───────┘ └───────┬────────┘
       │                │                 │
       ▼                ▼                 ▼
  具體策略:         具體實現:        ┌─────────────────┐
  - Priority       - Symbolic       │  ExecutionState │
  - DFS            - Concrete       │   (dataclass)   │
  - BFS                             └─────────────────┘
  - Random

禁止方向：↑（下層不得依賴上層）
```

---

## 接口定義

### Searcher 接口

```python
# libct/searcher/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
from libct.state.execution_state import ExecutionState

class Searcher(ABC):
    @abstractmethod
    def select_state(self) -> ExecutionState:
        """選擇下一個要探索的狀態。隊列為空時拋出 IndexError。"""
        ...

    @abstractmethod
    def update(self,
               current: Optional[ExecutionState],
               added: List[ExecutionState],
               removed: List[ExecutionState]) -> None:
        """根據執行結果更新策略狀態。"""
        ...

    @abstractmethod
    def empty(self) -> bool:
        """是否還有待探索的狀態。"""
        ...
```

### Executor 接口

```python
# libct/executor/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
from libct.state.execution_state import ExecutionState

class Executor(ABC):
    @abstractmethod
    def execute_initial(self, input_data: Any) -> ExecutionState:
        """執行初始（具體）運行，返回根狀態。"""
        ...

    @abstractmethod
    def execute(self, state: ExecutionState) -> List[ExecutionState]:
        """從給定狀態進行符號執行，返回後繼狀態列表。"""
        ...
```

---

## 主要模塊職責表

| 模塊 | 職責 | 不做什麼 |
|------|------|---------|
| `ExplorationEngine` | 協調探索循環、超時監控、結果收集 | 不選擇路徑、不執行符號運算 |
| `Searcher` | 決定探索哪條路徑 | 不執行任何計算、不修改狀態 |
| `Executor` | 執行函數（符號/具體） | 不決定探索順序、不存儲歷史 |
| `ExecutionState` | 存儲單個執行路徑的所有信息 | 不包含業務邏輯 |
| `StateManager` | 管理所有活躍狀態的集合 | 不選擇狀態、不執行狀態 |

---

## 文件目錄結構

重構後的 `libct/` 目錄結構：

```
libct/
├── __init__.py
├── explore.py          # ExplorationEngine（精簡版）
├── solver.py           # 不變
├── constraint.py       # 不變
├── path.py             # 不變
├── record.py           # 不變
├── utils.py            # 不變
├── wrapper.py          # 不變
├── position.py         # 不變
├── predicate.py        # 不變
├── random_assign_attack.py  # 不變
├── concolic/           # 不變
│   ├── __init__.py
│   ├── bool.py
│   ├── float.py
│   ├── int.py
│   ├── range.py
│   └── str.py
├── searcher/           # 新增
│   ├── __init__.py
│   ├── base.py
│   ├── priority.py
│   ├── dfs.py
│   ├── bfs.py
│   └── random_searcher.py
├── state/              # 新增
│   ├── __init__.py
│   ├── execution_state.py
│   └── manager.py
└── executor/           # 新增
    ├── __init__.py
    ├── base.py
    ├── symbolic.py
    └── concrete.py
```

---

## Current Compatibility Implementation Snapshot（2026-06-02）

The repository is currently in a compatibility modularization stage. The target
architecture above remains the KLEE-inspired destination, but the current code
intentionally keeps `libct.explore.ExplorationEngine` as the stable entrypoint
while delegating selected legacy responsibilities to smaller modules.

Current `libct/` runtime shape:

```
libct/
├── explore.py                         # 810-line compatibility coordinator
├── searcher/
│   ├── base.py                        # constraint worklist interface today
│   ├── constraint_worklist.py          # stack / queue / priority / random
│   └── constraint_scheduler.py         # legacy push/pop facade
├── executor/
│   ├── base.py
│   ├── legacy.py                      # adapter around existing engine methods
│   ├── child_protocol.py              # child envelope and transfer protocol
│   ├── concolic.py                    # concolic subprocess runner
│   ├── primitive.py                   # primitive/coverage subprocess runner
│   ├── execution_pair.py              # candidate validation + execution pair
│   └── arguments.py                   # concolic argument builder
└── state/
    ├── execution_state.py
    ├── manager.py
    └── work_item.py                   # constraint-level work item
```

Important compatibility notes:

- Search scheduling is still **constraint-level**, through `ConstraintWorkItem`,
  not yet full path-level `ExecutionState` scheduling.
- `libct/explore.py` keeps wrapper methods such as `push_constraint()`,
  `pop_constraint()`, `_one_execution()`, and `_get_concolic_arguments()` so
  existing call sites and tests retain their patch points.
- `ChildProtocol`, `ConcolicExecutionRunner`, `PrimitiveExecutionRunner`,
  `CandidateExecutionRunner`, `ConcolicArgumentBuilder`, and
  `ConstraintScheduler` still depend on compatibility accessors on
  `ExplorationEngine`.
- The final `< 300 行` `ExplorationEngine` remains a later acceptance target.

Next target boundary:

- Extract `explore()` setup/teardown into a runtime session or environment
  helper before moving `_execution_loop()`.
- Preserve CLI and artifact output contracts while this compatibility shell is
  still the public runtime entrypoint.

---

## 初始化流程代碼

```python
# engine/executor.py 中 ExplorerConfig 實例化 ExplorationEngine 的方式
def create_exploration_engine(config: ExplorerConfig) -> ExplorationEngine:
    # 1. 選擇 Searcher 策略
    if config.attack_mode == "shap":
        searcher = PrioritySearcher(
            alpha=config.shap_score_alpha or 0.5,
            alpha_schedule="linear"
        )
    elif config.attack_mode == "queue":
        searcher = BFSSearcher()
    elif config.attack_mode == "random":
        searcher = RandomSearcher(seed=config.base_seed)
    else:
        searcher = DFSSearcher()

    # 2. 創建 Executor
    symbolic_executor = SymbolicExecutor(
        module=config.module,
        execute=config.execute,
        solver=Solver(config.solver),
        timeout=config.timeout,
        constraint_build_timeout=config.constraint_build_timeout,
    )

    # 3. 創建 StateManager
    state_manager = StateManager()

    # 4. 創建 ExplorationEngine
    return ExplorationEngine(
        searcher=searcher,
        executor=symbolic_executor,
        state_manager=state_manager,
        timeout=config.timeout,
        verbose=config.verbose,
    )
```

---

## 3 個關鍵設計決策

### 決策一：`ExecutionState` 使用 `@dataclass` 而非繼承 `dict`

```python
# ✅ 選擇：dataclass
@dataclass
class ExecutionState:
    path_id: str
    depth: int
    constraints: List[Constraint]
    shap_scores: Optional[np.ndarray] = None
    coverage: Optional[Set[str]] = None

# ❌ 棄選：dict 子類
class ExecutionState(dict):
    # 無類型安全，IDE 無自動補全，容易拼錯鍵名
```

**理由**：dataclass 提供類型提示、默認值、`__repr__`、`__eq__`，且可被 `pickle` 序列化（多進程需要）。

### 決策二：`Searcher.update()` 批量處理而非單個處理

```python
# ✅ 選擇：批量更新
def update(self, current, added: List[ExecutionState], removed: List[ExecutionState]):
    ...

# ❌ 棄選：逐一調用
def add_state(self, state: ExecutionState): ...
def remove_state(self, state: ExecutionState): ...
```

**理由**：符合 KLEE 接口設計，允許 Searcher 在一次更新中做全局優化（如重新計算所有優先級）。

### 決策三：`Executor` 每次執行都在獨立子進程中運行

```python
# ✅ 選擇：進程隔離
class SymbolicExecutor(Executor):
    def execute(self, state: ExecutionState) -> List[ExecutionState]:
        # 在 multiprocessing.Process 中執行，隔離內存
        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=self._run, args=(state, result_queue))
        p.start()
        p.join(timeout=self.timeout)
        ...
```

**理由**：符號執行可能有內存洩漏或崩潰，進程隔離保證主進程穩定性。

---

## 測試架構

```
test/
├── unit/
│   ├── test_searcher_priority.py
│   ├── test_searcher_dfs.py
│   ├── test_searcher_bfs.py
│   ├── test_searcher_random.py
│   ├── test_execution_state.py
│   ├── test_state_manager.py
│   ├── test_executor_symbolic.py
│   └── test_executor_concrete.py
│
├── integration/
│   ├── test_engine_with_searchers.py
│   ├── test_engine_with_executors.py
│   └── test_full_exploration_cycle.py
│
└── system/
    ├── test_cli_backward_compat.py
    └── test_output_equivalence.py  # 與舊版輸出對比
```

---

## 性能特性表

| 操作 | 時間複雜度 | 說明 |
|------|-----------|------|
| `PrioritySearcher.select_state()` | O(log n) | heapq 實現 |
| `DFSSearcher.select_state()` | O(1) | list.pop() |
| `BFSSearcher.select_state()` | O(1) | deque.popleft() |
| `RandomSearcher.select_state()` | O(1) | random.choice() |
| `StateManager.add()` | O(1) | dict 插入 |
| `StateManager.get()` | O(1) | dict 查找 |
| `SymbolicExecutor.execute()` | O(路徑深度) | 取決於約束複雜度 |
