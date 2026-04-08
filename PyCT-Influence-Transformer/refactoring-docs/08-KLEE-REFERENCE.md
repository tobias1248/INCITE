# 📚 KLEE 原始代碼參考

## 概述

KLEE 是學術界和工業界最廣泛使用的符號執行引擎之一，由 LLVM 位元組碼的符號執行實現。本文檔整理了 KLEE 中與 PyCT 重構最相關的代碼片段，以及 PyCT 的對應實現。

**KLEE 版本**: [GitHub v3.1](https://github.com/klee/klee)  
**相關文件**: `lib/Core/Searcher.h`, `lib/Core/Searcher.cpp`, `lib/Core/ExecutionState.h`, `lib/Core/Executor.h`

---

## KLEE 的 Searcher 接口（原始 C++ 代碼 + PyCT 對應物）

### 原始 C++ 代碼

```cpp
// KLEE: lib/Core/Searcher.h
namespace klee {

class Searcher {
public:
  // Explicit name for this searcher, used for logging and debugging.
  const std::string name;

  explicit Searcher(const std::string &name) : name(name) {}
  virtual ~Searcher();

  // 選擇下一個要執行的狀態
  virtual ExecutionState &selectState() = 0;

  // 根據執行結果更新 Searcher 的內部狀態
  // current: 當前執行完的狀態（可為 nullptr 表示初始化）
  // addedStates: 新添加的後繼狀態
  // removedStates: 需要移除的狀態
  virtual void update(ExecutionState *current,
                      const std::vector<ExecutionState *> &addedStates,
                      const std::vector<ExecutionState *> &removedStates) = 0;

  // 是否還有狀態待探索
  virtual bool empty() = 0;

  // 輸出 Searcher 狀態（用於調試）
  virtual void printName(llvm::raw_ostream &os) {
    os << "<unnamed searcher>\n";
  }
};

} // namespace klee
```

### PyCT 對應物

```python
# libct/searcher/base.py（PyCT 等價實現）
from abc import ABC, abstractmethod
from typing import List, Optional
from libct.state.execution_state import ExecutionState

class Searcher(ABC):
    """等價於 KLEE 的 klee::Searcher"""

    @abstractmethod
    def select_state(self) -> ExecutionState:
        """等價於 KLEE 的 selectState()"""
        ...

    @abstractmethod
    def update(self,
               current: Optional[ExecutionState],
               added: List[ExecutionState],
               removed: List[ExecutionState]) -> None:
        """等價於 KLEE 的 update()"""
        ...

    @abstractmethod
    def empty(self) -> bool:
        """等價於 KLEE 的 empty()"""
        ...
```

**主要差異**：
- KLEE 返回引用（`&`），PyCT 返回對象引用（Python 默認）
- KLEE 使用 `nullptr`，PyCT 使用 `Optional[...]`
- KLEE 有 `printName` 調試方法，PyCT 使用 `__repr__`

---

## KLEE 的 ExecutionState（字段映射）

### KLEE 的關鍵字段

```cpp
// KLEE: lib/Core/ExecutionState.h（精選）
namespace klee {

class ExecutionState {
public:
  // 路徑深度
  unsigned depth;
  
  // 路徑約束
  ConstraintSet constraints;
  
  // 地址空間（內存快照）
  AddressSpace addressSpace;
  
  // 程序計數器
  KInstIterator pc, prevPC;
  
  // 調用棧
  CallStack stack;
  
  // 內存使用量
  mutable double queryCost;
  
  // 是否已執行完畢
  bool isFinished;
  
  // 分叉計數
  unsigned forkDisabled;
  
  // 路徑 ID（用於調試）
  unsigned id;
  
  // 路徑哈希（用於去重）
  uint64_t pathHash;
};

} // namespace klee
```

### PyCT ExecutionState 字段映射表

| KLEE 字段 | PyCT 字段 | 類型 | 說明 |
|----------|----------|------|------|
| `depth` | `depth` | `int` | 路徑深度 |
| `constraints` | `constraints` | `List[Constraint]` | 路徑約束列表 |
| `id` | `path_id` | `str` | 路徑唯一標識 |
| `isFinished` | `status == 'done'` | `str` | 執行狀態 |
| `queryCost` | `metadata['execution_time_s']` | `float` | 執行成本 |
| N/A | `shap_scores` | `np.ndarray` | PyCT 特有：SHAP 分數 |
| N/A | `input_data` | `Any` | PyCT 特有：輸入數據 |
| N/A | `output` | `Any` | PyCT 特有：模型輸出 |
| `addressSpace` | N/A | - | KLEE 特有：內存快照 |
| `stack` | N/A | - | KLEE 特有：調用棧 |

---

## KLEE 的搜索策略實現（DFS, Random 等）

### DFSSearcher（原始 C++）

```cpp
// KLEE: lib/Core/Searcher.cpp
ExecutionState &DFSSearcher::selectState() {
  // 從棧頂取狀態（後進先出）
  return *states.back();
}

void DFSSearcher::update(
    ExecutionState *current,
    const std::vector<ExecutionState *> &addedStates,
    const std::vector<ExecutionState *> &removedStates) {
  // 先添加新狀態
  states.insert(states.end(),
                addedStates.begin(),
                addedStates.end());
  
  // 再移除已完成的狀態
  for (const auto *es : removedStates) {
    auto it = std::find(states.begin(), states.end(), es);
    assert(it != states.end());
    states.erase(it);
  }
}
```

### PyCT 等價實現

```python
# libct/searcher/dfs.py（PyCT 等價）
class DFSSearcher(Searcher):
    def select_state(self) -> ExecutionState:
        return self._stack.pop()  # 後進先出，等價於 states.back()

    def update(self, current, added, removed):
        removed_ids = {id(s) for s in removed}
        self._stack = [s for s in self._stack if id(s) not in removed_ids]
        self._stack.extend(added)  # 等價於 states.insert(end, addedStates)
```

### RandomSearcher（原始 C++）

```cpp
// KLEE: lib/Core/Searcher.cpp
ExecutionState &RandomSearcher::selectState() {
  // 隨機均勻選擇
  return *states[theRNG.getInt32() % states.size()];
}
```

### PyCT 等價實現

```python
class RandomSearcher(Searcher):
    def select_state(self) -> ExecutionState:
        idx = self._rng.randrange(len(self._states))
        self._states[idx], self._states[-1] = self._states[-1], self._states[idx]
        return self._states.pop()  # swap-and-pop，O(1)
```

---

## KLEE 的 Executor (Interpreter)

### 核心執行循環（概念）

```cpp
// KLEE: lib/Core/Executor.cpp（概念性摘錄）
void Executor::run(ExecutionState &initialState) {
  // 注入初始狀態
  searcher->update(0, {&initialState}, {});
  
  // 主執行循環
  while (!searcher->empty()) {
    // 選擇下一個狀態
    ExecutionState &state = searcher->selectState();
    
    // 執行一條指令
    KInstruction *ki = state.pc;
    stepInstruction(state);
    executeInstruction(state, ki);
    
    // 更新統計
    ++stats::instructions;
    
    // 周期性檢查終止條件
    if (--timeBudget <= 0) {
      haltExecution = true;
    }
  }
}
```

### PyCT 等價實現

```python
# libct/explore.py（重構後，ExplorationEngine._main_loop）
def _main_loop(self, root_state, start_time):
    # 等價於 KLEE 的 while (!searcher->empty())
    while not self._searcher.empty():
        # 等價於 searcher->selectState()
        current = self._searcher.select_state()
        
        # 等價於 executeInstruction()，但在子進程中
        successors = self._executor.execute(current)
        
        # 等價於 KLEE 的超時檢查
        if self._is_timed_out(start_time):
            return "timeout"
        
        # 等價於 searcher->update()
        self._searcher.update(current, added=successors, removed=[current])
    
    return "exhausted"
```

**主要差異**：
- KLEE 執行 LLVM 指令（位元組碼級），PyCT 執行 Python 函數（抽象語義）
- KLEE 的狀態分叉發生在指令執行中，PyCT 在 `execute()` 完成後
- KLEE 的子進程模型基於 OS fork，PyCT 使用 `multiprocessing.Process`

---

## KLEE 的 ConstraintManager

### 原始 C++ 代碼

```cpp
// KLEE: lib/Core/ConstraintManager.h
class ConstraintManager {
public:
  using constraints_ty = std::vector<ref<Expr>>;
  
  ConstraintManager() = default;
  explicit ConstraintManager(const constraints_ty &_constraints);
  
  // 添加約束並簡化
  bool addConstraint(ref<Expr> e);
  
  // 迭代訪問
  constraints_ty::const_iterator begin() const;
  constraints_ty::const_iterator end() const;
  size_t size() const;
  
private:
  constraints_ty constraints;
  
  // 約束化簡（消除重複，常量折疊）
  ref<Expr> simplifyExpr(const ConstraintSet &constraints, const ref<Expr> &e) const;
};
```

### PyCT 對應物

PyCT 的約束管理分散在 `libct/path.py` 和 `libct/constraint.py` 中：

```python
# libct/path.py（對應 KLEE ConstraintManager）
class PathToConstraint:
    def __init__(self, add_constraint):
        self._constraints = []
        self._add_constraint = add_constraint
    
    def which(self, cond):
        # 等價於 KLEE 的 addConstraint()
        # 記錄路徑分支條件
        ...

# libct/constraint.py
class Constraint:
    # 等價於 KLEE 的 ref<Expr>
    # 存儲 SMT 約束表達式
    ...
```

---

## 架構對比（KLEE vs PyCT）

```
KLEE 架構：                        PyCT 架構（重構後）：

klee::Executor                     ExplorationEngine
    │                                   │
    ├── Searcher (接口)             ├── Searcher (接口)
    │   ├── DFSSearcher             │   ├── DFSSearcher
    │   ├── BFSSearcher             │   ├── BFSSearcher
    │   ├── RandomSearcher          │   ├── RandomSearcher
    │   └── WeightedRandomSearcher  │   └── PrioritySearcher ← SHAP 特有
    │                               │
    ├── ExecutionState              ├── ExecutionState
    │   ├── constraints             │   ├── constraints
    │   ├── addressSpace            │   ├── shap_scores ← 特有
    │   └── callStack               │   └── output ← 特有
    │                               │
    ├── Solver                      ├── Executor (包裝 Solver)
    │   ├── MetaSolver              │   ├── SymbolicExecutor
    │   ├── Z3Solver                │   └── ConcreteExecutor
    │   └── CVC5Solver              │
    │                               └── StateManager ← 新增
    └── ConstraintManager
        └── PathToConstraint
```

**關鍵相同點**：
1. Searcher 接口完全對應
2. ExecutionState 的核心字段（depth, constraints, id）
3. 主執行循環結構（select → execute → update）

**關鍵差異**：
1. PyCT 有 SHAP 分數（KLEE 無此概念）
2. PyCT 的 Executor 在子進程中運行（KLEE 使用 OS fork）
3. PyCT 有 StateManager（KLEE 的狀態集合在 Executor 內部）
4. KLEE 有完整的 AddressSpace（PyCT 不需要，Python 有 GC）

---

## 學習要點

### 1. 為什麼 KLEE 的 Searcher 設計如此優雅？

KLEE 的 Searcher 接口只有 3 個方法（`selectState`, `update`, `empty`），但足以支持所有搜索策略。

**關鍵洞察**：狀態選擇的「排序策略」與「執行邏輯」完全解耦。Executor 不知道 Searcher 的存在，Searcher 不知道如何執行狀態。

### 2. 批量更新（update with vectors）的重要性

KLEE 的 `update()` 接受向量而非單個狀態，允許 Searcher 在一次調用中做全局優化：

```cpp
// KLEE 允許 WeightedRandomSearcher 在每次 update 後重新計算所有權重
void WeightedRandomSearcher::update(current, addedStates, removedStates) {
  // 添加新狀態到加權分佈
  for (auto es : addedStates) {
    double w = getWeight(es);
    states->insert(es, w);
  }
  // 移除舊狀態
  for (auto es : removedStates) {
    states->remove(es);
  }
}
```

### 3. ExecutionState 的輕量化原則

KLEE 的 ExecutionState 只存儲「恢復執行所需的最小信息」。PyCT 也應遵循此原則：ExecutionState 不應包含模型、求解器等重量級對象。

---

## 參考資源

| 資源 | URL | 相關性 |
|------|-----|--------|
| KLEE 論文（OSDI 2008） | https://www.usenix.org/conference/osdi-08/ | ⭐⭐⭐ |
| KLEE 源碼 Searcher.h | https://github.com/klee/klee/blob/main/lib/Core/Searcher.h | ⭐⭐⭐ |
| KLEE 源碼 ExecutionState.h | https://github.com/klee/klee/blob/main/lib/Core/ExecutionState.h | ⭐⭐⭐ |
| KLEE 源碼 Executor.cpp | https://github.com/klee/klee/blob/main/lib/Core/Executor.cpp | ⭐⭐ |
| KLEE 教程 | https://klee-se.org/docs/tutorials/ | ⭐⭐ |

---

## 快速查詢表

| 我想了解... | KLEE 文件 | PyCT 對應文件 |
|------------|----------|--------------|
| Searcher 接口 | `lib/Core/Searcher.h` | `libct/searcher/base.py` |
| DFS 實現 | `lib/Core/Searcher.cpp` | `libct/searcher/dfs.py` |
| 狀態字段 | `lib/Core/ExecutionState.h` | `libct/state/execution_state.py` |
| 主執行循環 | `lib/Core/Executor.cpp: run()` | `libct/explore.py: _main_loop()` |
| 約束管理 | `lib/Core/ConstraintManager.h` | `libct/path.py` + `libct/constraint.py` |
| 求解器接口 | `lib/Solver/Solver.h` | `libct/solver.py` |
