# 🎓 設計哲學 & KLEE 原則

## 核心三大支柱詳解

### 支柱一：Separation of Concerns（關注點分離）

**原則**：每個類只有一個改變的理由。

在重構前，`ExplorationEngine` 同時負責：
- 管理執行狀態的生命週期
- 選擇下一條要探索的路徑
- 執行符號運算
- 計算 SHAP 分數並排序
- 超時監控
- 結果收集

這導致任何一個方面的修改都可能影響其他方面。重構後：

| 職責 | 負責類 | 改變原因 |
|------|--------|---------|
| 路徑選擇策略 | `Searcher` | 要更換排程算法 |
| 符號執行 | `Executor` | 要換求解器或執行方式 |
| 狀態數據 | `ExecutionState` | 要增加狀態字段 |
| 狀態生命週期 | `StateManager` | 要修改狀態存儲方式 |
| 探索協調 | `ExplorationEngine` | 要修改整體流程 |

---

### 支柱二：Pluggable Strategies（可插拔策略）

**原則**：對擴展開放，對修改封閉（Open/Closed Principle）。

借鑑 KLEE 的 `Searcher` 接口設計，定義統一抽象：

```python
class Searcher(ABC):
    @abstractmethod
    def select_state(self) -> ExecutionState:
        """選擇下一個要探索的狀態"""
        ...

    @abstractmethod
    def update(self,
               current: Optional[ExecutionState],
               added: List[ExecutionState],
               removed: List[ExecutionState]) -> None:
        """根據執行結果更新策略"""
        ...

    @abstractmethod
    def empty(self) -> bool:
        """是否還有待探索的狀態"""
        ...
```

**好處**：
- 新增 `GreedyCoverageSearcher` 只需新建一個文件
- 不需要修改 `ExplorationEngine`
- 不需要修改現有 Searcher 實現
- 可以在運行時動態切換策略（未來擴展）

---

### 支柱三：Clear Data Flow（清晰數據流）

**原則**：依賴只沿一個方向流動，避免循環依賴。

```
ExplorationEngine
    │
    ├──> Searcher (選擇狀態)
    │       │
    │       └──> ExecutionState (只讀)
    │
    ├──> Executor (執行函數)
    │       │
    │       └──> ExecutionState (生成新狀態)
    │
    └──> StateManager (管理狀態集合)
            │
            └──> ExecutionState (存儲和查詢)
```

**禁止**：
- `Searcher` 不得調用 `Executor`
- `ExecutionState` 不得引用 `Searcher` 或 `Executor`
- `StateManager` 不得調用 `ExplorationEngine`

---

## PyCT 的獨特挑戰

PyCT 相比純符號執行工具（如 KLEE）有以下特殊需求，設計時必須考慮：

### 挑戰一：SHAP 集成

SHAP 值計算需要運行模型推理，成本較高。設計考量：

```python
# ❌ 錯誤：在每次路徑選擇時重新計算 SHAP
class BadSearcher(Searcher):
    def select_state(self) -> ExecutionState:
        for state in self.states:
            state.shap_score = self.recalculate_shap(state)  # 太慢！
        return max(self.states, key=lambda s: s.shap_score)

# ✅ 正確：SHAP 值在初始執行後計算一次，存入狀態
class PrioritySearcher(Searcher):
    def select_state(self) -> ExecutionState:
        _, state = heapq.heappop(self.heap)  # O(log n)，SHAP 已預計算
        return state
    
    def update(self, current, added, removed):
        for state in added:
            # SHAP 分數已在 Executor 執行後計算並存入 state
            score = self._compute_combined_score(state)
            heapq.heappush(self.heap, (-score, state))
```

### 挑戰二：動態 α 調度

α 是 path_length 與 SHAP 分數的混合權重，隨探索進度動態調整：

```
score(state) = α * shap_score(state) + (1 - α) * path_depth_score(state)
```

隨著探索深入，α 逐漸增大（更依賴 SHAP 啟發），以避免在深層路徑中浪費資源。

**設計方案**：α 調度邏輯封裝在 `PrioritySearcher` 中：

```python
class PrioritySearcher(Searcher):
    def __init__(self, initial_alpha: float = 0.5,
                 alpha_schedule: str = "linear"):
        self.alpha = initial_alpha
        self.iteration = 0
        self.alpha_schedule = alpha_schedule

    def _current_alpha(self) -> float:
        if self.alpha_schedule == "linear":
            # 每 10 次迭代增加 0.05，上限 0.95
            return min(0.95, self.alpha + 0.05 * (self.iteration // 10))
        elif self.alpha_schedule == "fixed":
            return self.alpha
        return self.alpha
```

### 挑戰三：多進程協調

PyCT 的多進程在不同 input case 間並行（非路徑探索間並行），因此：

- `Searcher`、`Executor`、`StateManager` 都是**進程內單例**
- 不需要跨進程鎖（每個 worker 有獨立的探索引擎）
- `orchestration/` 層負責跨進程協調（不在本次重構範圍）

---

## 設計決策表

| 決策 | 選擇 | 理由 | 棄選項 |
|------|------|------|--------|
| Searcher 接口設計 | 仿 KLEE C++ 接口 | 經過大規模生產驗證 | Manticore 的 Plugin 系統（過於複雜） |
| 狀態表示 | Python `dataclass` | 清晰、可序列化、支援類型提示 | 繼承 dict（無類型安全） |
| 優先隊列實現 | `heapq` | 標準庫、高性能 | `sortedcontainers`（額外依賴） |
| α 調度位置 | `PrioritySearcher` 內部 | 調度策略與搜索策略綁定 | 獨立 Scheduler 類（過度設計） |
| 線程安全 | `threading.Lock` | 進程內多線程保護 | `multiprocessing.Lock`（不需要跨進程） |

---

## 指標定義

| 指標 | 計算方式 | 良好標準 |
|------|---------|---------|
| **模塊耦合度** | 跨模塊的直接調用數 | < 5 個公共方法依賴 |
| **扇入（Fan-In）** | 有多少模塊依賴此模塊 | Searcher/Executor: ≤ 1（只有 Engine） |
| **扇出（Fan-Out）** | 此模塊依賴多少模塊 | Engine: ≤ 4 |
| **代碼行數** | 每個文件的實際代碼行 | < 300 行（不含注釋） |
| **圈複雜度** | McCabe 複雜度 | < 10（每個方法） |

---

## 反面教材（避免的做法）

### ❌ God Class（上帝類）

```python
# 避免：一個類知道所有事情
class ExplorationEngine:
    def __init__(self):
        self.priority_queue = []
        self.visited_states = {}
        self.shap_calculator = ShapValuesComparator(...)
        self.solver = Solver(...)
        self.coverage_tracker = CoverageTracker()
        # ... 20 多個實例變量
    
    def explore(self):
        # ... 300 行代碼，做所有事情
```

### ❌ Feature Envy（功能嫉妒）

```python
# 避免：一個方法大量使用另一個對象的數據
class ExplorationEngine:
    def _select_next_state(self):
        # 這個方法主要操作 state 的數據 → 應該是 Searcher 的職責
        best_score = -1
        best_state = None
        for state in self.states:
            score = state.shap_score * self.alpha + state.depth * (1 - self.alpha)
            if score > best_score:
                best_score = score
                best_state = state
        return best_state
```

### ❌ 隱式狀態（Implicit State）

```python
# 避免：狀態散落在多個地方
class ExplorationEngine:
    self.path_to_constraint = {}  # 路徑約束在這裡
    self.visited = set()          # 訪問記錄在這裡
    self.shap_scores = {}         # SHAP 分數在這裡（與狀態分離）
```

---

## 推薦閱讀

| 資源 | 相關性 | 重點章節 |
|------|--------|---------|
| [KLEE 論文](https://www.usenix.org/conference/osdi-08/) | ⭐⭐⭐ | Searcher 設計 |
| [Architecture Patterns with Python](https://www.cosmicpython.com/) | ⭐⭐⭐ | Domain Model, Repository Pattern |
| [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) | ⭐⭐ | Dependency Rule |
| KLEE 源碼 `lib/Core/Searcher.h` | ⭐⭐⭐ | 接口設計 |
| KLEE 源碼 `lib/Core/ExecutionState.h` | ⭐⭐ | 狀態字段設計 |
