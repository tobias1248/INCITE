# 📋 PyCT 重構指南總覽

## 快速導航

| 文檔 | 內容 | 狀態 |
|-----|------|------|
| [00-OVERVIEW.md](./00-OVERVIEW.md) | 總覽和快速開始（本文件） | ✅ |
| [01-DESIGN-PHILOSOPHY.md](./01-DESIGN-PHILOSOPHY.md) | 設計哲學 & KLEE 原則 | ✅ |
| [02-ARCHITECTURE.md](./02-ARCHITECTURE.md) | 目標架構設計 | ✅ |
| [03-MODULE-SEARCHER.md](./03-MODULE-SEARCHER.md) | Searcher 模塊指南 | ✅ |
| [04-MODULE-EXECUTOR.md](./04-MODULE-EXECUTOR.md) | Executor 模塊指南 | ✅ |
| [05-MODULE-STATE.md](./05-MODULE-STATE.md) | State 管理模塊指南 | ✅ |
| [06-MODULE-ENGINE.md](./06-MODULE-ENGINE.md) | ExplorationEngine 模塊指南 | ✅ |
| [07-INTEGRATION-CHECKLIST.md](./07-INTEGRATION-CHECKLIST.md) | 集成與驗證清單 | ✅ |
| [08-KLEE-REFERENCE.md](./08-KLEE-REFERENCE.md) | KLEE 原始代碼參考 | ✅ |

---

## 🎯 重構目標（一句話版本）

> **將 `libct/explore.py` 中單體的 `ExplorationEngine` 拆分為四個職責清晰的模塊：Searcher（路徑排程）、Executor（符號執行）、State（狀態管理）、Engine（探索協調），以 KLEE 架構為藍本，保留 SHAP 驅動的動態 α 調度優勢。**

---

## 📊 現狀 vs 目標對比

| 維度 | 現狀 | 目標 |
|------|------|------|
| **核心文件** | `libct/explore.py`（~800 行單體） | 4 個模塊，各 < 300 行 |
| **搜索策略** | 硬編碼於 `ExplorationEngine` | `Searcher` 抽象基類 + 可插拔策略 |
| **執行邏輯** | 混在探索循環中 | 獨立 `Executor` 類 |
| **狀態管理** | 散落在多個方法中 | `ExecutionState` dataclass + `StateManager` |
| **α 調度** | 內嵌在優先隊列邏輯 | `PrioritySearcher` 封裝 |
| **可測試性** | 需要完整模型才能測試 | 每個模塊可獨立單元測試 |
| **擴展性** | 新增策略需修改核心代碼 | 新增策略只需繼承 `Searcher` |

---

## 🗺️ 3 個 Phase 的路線圖

### Phase 1：提取 State 與 Searcher（週 1-2）
- 定義 `ExecutionState` dataclass（從現有代碼提取字段）
- 實現 `StateManager`（線程安全）
- 實現 `Searcher` 抽象基類
- 實現 `PrioritySearcher`、`DFSSearcher`、`BFSSearcher`、`RandomSearcher`
- **驗收**：Searcher 單元測試全通過

### Phase 2：提取 Executor（週 3-4）
- 定義 `Executor` 抽象基類
- 實現 `SymbolicExecutor`（包裝現有符號執行邏輯）
- 實現 `ConcreteExecutor`（覆蓋收集）
- **驗收**：Executor 可獨立運行，不依賴 `ExplorationEngine`

### Phase 3：重構 Engine + 集成（週 5-6）
- 重構 `ExplorationEngine`（< 300 行協調者）
- 集成所有模塊
- 端對端測試（與舊版輸出對比）
- **驗收**：所有現有 CLI 命令行為不變

---

## 📂 文檔查詢表

| 我想了解... | 看這裡 |
|------------|--------|
| 為什麼要這樣設計？ | [01-DESIGN-PHILOSOPHY.md](./01-DESIGN-PHILOSOPHY.md) |
| 各模塊如何分工？ | [02-ARCHITECTURE.md](./02-ARCHITECTURE.md) |
| 如何實現搜索策略？ | [03-MODULE-SEARCHER.md](./03-MODULE-SEARCHER.md) |
| 如何處理符號執行？ | [04-MODULE-EXECUTOR.md](./04-MODULE-EXECUTOR.md) |
| 如何管理執行狀態？ | [05-MODULE-STATE.md](./05-MODULE-STATE.md) |
| Engine 如何協調？ | [06-MODULE-ENGINE.md](./06-MODULE-ENGINE.md) |
| 測試清單是什麼？ | [07-INTEGRATION-CHECKLIST.md](./07-INTEGRATION-CHECKLIST.md) |
| KLEE 如何啟發設計？ | [08-KLEE-REFERENCE.md](./08-KLEE-REFERENCE.md) |

---

## 🏛️ 核心三大支柱概述

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. Separation of Concerns（關注點分離）              │
│     ─ 每個模塊只做一件事                              │
│     ─ Searcher 選路徑，Executor 執行，State 存狀態   │
│                                                     │
│  2. Pluggable Strategies（可插拔策略）                │
│     ─ 借鑑 KLEE Searcher 接口                        │
│     ─ 新增策略不需修改 Engine                         │
│                                                     │
│  3. Clear Data Flow（清晰數據流）                     │
│     ─ 單向依賴：Engine → Searcher/Executor/State     │
│     ─ 無循環依賴                                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## ❓ 常見問題解答

**Q：重構後 SHAP 驅動的 α 調度還在嗎？**
A：在，被封裝進 `PrioritySearcher`，邏輯不變，只是位置更清晰。

**Q：現有 CLI 接口會改變嗎？**
A：不會。`pyct/` 目錄下的 CLI 接口保持完全向後兼容。

**Q：重構後多進程架構有變化嗎？**
A：`orchestration/` 的多進程架構不在本次重構範圍內。本次只重構 `libct/explore.py`。

**Q：為什麼選 KLEE 而不是 Manticore？**
A：詳見 [01-DESIGN-PHILOSOPHY.md](./01-DESIGN-PHILOSOPHY.md#設計決策表)。

**Q：重構的目標代碼行數？**
A：現有 ~800 行單體 → 目標 4 個模塊各 < 300 行，總量可能增加但可讀性大幅提升。

---

## ✅ 成功指標

| 指標 | 當前值 | 目標值 |
|------|--------|--------|
| 核心文件行數 | ~800 行 | 各模塊 < 300 行 |
| 單元測試覆蓋率 | < 30% | > 80% |
| 新增搜索策略所需修改文件數 | 3+ 文件 | 1 文件（繼承 Searcher） |
| 模塊間循環依賴數 | 多個 | 0 |
| 端對端行為相容性 | 基準 | 100% 兼容 |
