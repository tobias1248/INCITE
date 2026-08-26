# PyCT Influence Transformer 技術總覽

## 1. 文件目的

本文件提供 `PyCT-Influence-Transformer` 的技術導覽，說明其系統定位、核心架構、主要演算法設計與實驗工作流。這份文件的目標是幫助讀者快速理解 repo 的設計脈絡與主要模組分工，而不是取代原始碼作為唯一事實來源。

文件內容偏向 implementation-oriented overview；若後續程式行為與文字描述有落差，應以目前程式碼與測試為準。

若要快速掌握此專案，建議把它理解成一個面向影像分類模型的測試系統，其目標是在有限的像素變更空間內，透過符號執行與解釋性訊號，系統化地搜尋 adversarial input。

## 2. 系統要解決的問題

傳統隨機式 adversarial search 的問題在於：

- 搜尋空間過大，像素組合數量爆炸。
- 單靠 solver 容易遇到 constraint queue 過長、路徑太深、求解效率不穩定。
- 對 transformer 或較新架構，既有 concolic execution 支援通常不完整。
- 實驗往往缺少可恢復性、可追蹤性與標準化輸出。

本專案採用的策略是：

- 用 SHAP 預先估計重要像素或 token 區域。
- 將 SHAP 訊號導入 constraint scheduling，優先處理更有希望的 constraint。
- 用多階段 `ton` 設定逐步擴大允許修改的像素數。
- 為每個 case 建立獨立 artifacts 與統計，支援中斷後續跑與批次分析。

## 3. 系統定位與核心概念

### 3.1 Concolic testing 在本專案中的角色

本 repo 延伸自 PyCT 思路，將模型推論流程中的部分輸入維度設為可符號化變數，然後：

1. 先以原始輸入執行一次 forward pass，收集可翻轉的條件。
2. 逐步取出 constraint 並交由 SMT solver 求解。
3. 將 solver 找到的新輸入重新送入模型執行。
4. 若分類標籤改變，則記錄為 adversarial example；否則持續探索。

### 3.2 `ton` 的意義

`ton` 可以理解成單一測試階段中允許被修改的像素數量或座標數量。專案支援像 `1,2,4,8,16,32` 這樣的 multi-stage 設定，其目標是：

- 先以最小修改量嘗試攻擊。
- 若該階段 timeout 或 exhaust，才擴張到下一個 `ton`。
- 避免一開始就進入高維度、難求解的問題。

### 3.3 SHAP 的角色

SHAP 在此不是最終目標，而是搜尋指引：

- 在 preprocessing 階段，先以 Keras 對原始樣本的預測類別作為固定 target，
  再計算該 target class 的 SHAP value map。
- 在 task generation 階段選出 top-k 像素、patch 或 token group。
- 在 concolic engine 中，進一步用 SHAP 值與 path length 組合成 constraint priority。

### 3.4 Predicate、Constraint 與 Path 的基本觀念

本專案中的 concolic execution 不是直接把整個神經網路轉成單一大公式，而是沿著一次 concrete execution 實際走到的控制路徑，逐步收集 branch predicate，並把它們串成一條 constraint path。

可以把它拆成三層：

- `Predicate`：單一 branch 條件，例如 `x < 0`、`score_i > score_j`。
- `Constraint`：從 root 到某個 branch node 的 predicate 鏈。
- `PathToConstraint`：負責在執行過程中維護目前走到哪個 constraint node，以及哪些分支的反向條件還沒有被探索。

這個設計的核心不是「一次生成所有可能 constraint」，而是：

1. concrete run 先決定目前真實走到哪一條 path。
2. 每遇到一個 branch，就把「當前走到的方向」設成 current child。
3. 同時把「相反方向」建成 sibling constraint，推進待解 queue。

因此 queue 中保存的其實是「未來可探索的 path 分支」。

## 4. 系統架構

### 4.1 高層資料流

```text
python -m pyct.shap
    -> 產生 SHAP cache JSON

python -m pyct
    -> pyct/args.py 解析 CLI
    -> orchestration/launcher.py 建立 tasks / workers / ton stages
    -> orchestration/runners.py 執行單個 payload
    -> engine/executor.py 建立 ExplorationEngine
    -> libct/explore.py 執行 concolic exploration
    -> libct/record.py 寫出 stats / inputs / images
    -> python -m pyct.stats 做跨 case 聚合分析
```

### 4.2 主要模組職責

| 模組 | 角色 |
| --- | --- |
| `pyct/main.py` | 最薄的 attack 入口，串起 parse 與 launcher |
| `pyct/args.py` | 定義實驗參數、驗證 attack mode 與 selector 限制 |
| `orchestration/launcher.py` | 建立多進程 worker、安排 `ton` stage、處理 resume/skip |
| `tasks/builders/*` | 生成 dataset-specific payload、決定 output path |
| `pyct/shap.py` | 預先產生 SHAP cache 的 CLI |
| `engine/executor.py` | 載入模型、組裝 explorer、傳遞 metadata |
| `engine/predictor_runtime.py` | 載入 Keras model 並轉接到自訂 `NNModel` 執行路徑 |
| `dnnct/myDNN.py` | 純 Python forward path，補齊 transformer 相關 layer 支援 |
| `libct/explore.py` | concolic engine 主體，收集/排序/彈出 constraints |
| `libct/record.py` | 保存每次攻擊的統計、輸入、輸出與中間資料 |
| `reporting/experiment_stats.py` | 聚合 `stats.json`，產生跨實驗摘要 |

其中有一個實作細節值得特別說明：

- `libct/explore.py` 在 execution 期間會暫時切換工作目錄到 runtime module 所在位置。
- 為了避免輸出跟著 `cwd` 漂移，正式輸出路徑不再依賴相對路徑。
- `exp/`、`popped_constraint_position/` 等輸出都透過 `tasks.paths` 解析為 **repo root 下的絕對路徑**。

## 5. 執行流程

### 5.1 SHAP 預處理流程

SHAP 預處理由 `python -m pyct.shap` 對應的 `pyct/shap.py` 負責。它根據 dataset 與 model 選擇對應 handler，並將背景樣本設定寫入環境變數：

- `PYCT_BG_PER_CLASS`
- `PYCT_BG_SEED`

這樣的設計讓 SHAP 背景集具備兩個特性：

- class-balanced：每個類別抽固定數量樣本。
- reproducible：固定 random seed，降低不同 run 間的漂移。

輸出結果會以 JSON 形式保存在
`shap_target_class/<model_name>/shap_value_<idx>.json`。cache metadata 會記錄
target class，並驗證 model、input、background、case index 與 explainer identity；
舊的 class-averaged cache 不會被靜默重用。

### 5.2 Task generation

Task generation 的責任是把原始資料集樣本轉成 concolic engine 能執行的 payload。每個 payload 至少會包含：

- `model_name`
- `idx`
- `in_dict`
- `con_dict`
- `input_for_shap`
- `background_dataset_for_shap`
- `ton_plans`

其中：

- `in_dict` 是輸入張量攤平成 `v_i_j_k` 形式後的字典。
- `con_dict` 表示哪些輸入維度目前允許被符號化。
- `ton_plans` 則保存不同 stage 下的約束計畫。

對於 CIFAR10 的 SHAP 攻擊，`JsonShapPixelProvider` 會先從 SHAP cache 中取出 top pixels，再交由 `Cifar10Dataset.get_cifar10_test_data_and_set_condict()` 建立對應的 `con_dict`。

### 5.3 Launcher 與多進程調度

`orchestration/launcher.py` 會：

1. 依據 CLI 選擇 dataset-specific task generator。
2. 根據 attack mode 決定 runner 類型。
3. 啟動多個 persistent worker process。
4. 以 stage 為單位，分批提交相同 `ton` 的任務。
5. 在每一 stage 後讀取 `stats.json`，決定哪些 case 需要進入下一個 `ton`。

這個設計有兩個明顯優點：

- worker 可重複使用，不必每個 case 都重建整套執行環境。
- `ton` stage 的推進由實際執行結果控制，而不是盲目跑完整個序列。

### 5.4 Exploration 與 SMT solving

`engine/executor.py` 會把單個 payload 組裝成 `ExplorationEngine` 所需參數，並補上：

- `score_alpha`
- `symbolic_path_threshold`
- `ton`
- `ton_next`
- output path metadata

`ExplorationEngine` 的執行循環大致是：

1. 對原始輸入執行一次 forward，建立初始 constraint queue。
2. 從 queue 中取出下一個 constraint。
3. 呼叫 solver 求 model。
4. 若 SAT，生成新輸入再 forward 一次。
5. 若找到 label flip，記錄為成功。
6. 若 queue 空了，記錄為 exhausted。
7. 若超過 timeout，記錄為 timeout。

### 5.5 Predicate 是怎麼被找到的

這是 concolic engine 的核心細節。

#### (1) 輸入先被包成 ConcolicObject

在執行 target function 前，`ExplorationEngine` 會根據 `con_dict`，把允許符號化的輸入維度包成 `ConcolicObject`。例如原始輸入中的：

```text
v_12_7_0 = 0.413
```

會被轉成對應型別的 concolic 物件，例如 `ConcolicFloat(value=0.413, expr='v_12_7_0_VAR')`。

之後若這個值參與算術運算，相關運算子會建立 expression tree。例如：

- `x + y` 可能變成 `['+', x, y]`
- `x * w` 可能變成 `['*', x, w]`
- `x < y` 可能變成 `['<', x, y]`

也就是說，symbolic expression 不是字串拼接，而是巢狀 list / concolic node 的結構化表示。

#### (2) 真正產生 branch 的時機是 `__bool__`

本系統中，branch 並不是在比較運算當下就立刻加入 queue，而是在比較結果被 Python 當成布林值使用時才被截獲。也就是說，像這種情況：

```python
if score > 0:
    ...
```

會先得到一個 `ConcolicBool(expr=['>', score, 0])`，然後在進入 `if` 判斷時，觸發 `ConcolicBool.__bool__()`。這時 engine 才會：

1. 讀出目前 branch 的 symbolic expression。
2. 取得 concrete truth value。
3. 把這一個 branch 包成 `Predicate(expr, value)`。
4. 交給 `PathToConstraint.add_branch()` 更新 path tree。

#### (3) `add_branch()` 同時建立正向與反向分支

`PathToConstraint.add_branch()` 的邏輯很關鍵。對於目前 concrete execution 真正走到的條件 `p`：

- 建立 `p` 這個 child，並把它設成新的 `current_constraint`
- 同時建立 `not p` 的 sibling child
- 把 `not p` 對應的 constraint 推進待解 queue

換句話說，系統一邊沿著 concrete path 往前跑，一邊把「只差一個 branch 翻轉的替代路徑」存起來留給 solver。

### 5.6 Constraint 是怎麼生成的

當 queue 中某個 constraint 被取出後，solver 不會只看最後一個 predicate，而是會把整條 path 上的 predicate 全部展開。

具體流程是：

1. `Constraint.get_all_asserts()` 由目前 node 往 parent 回溯，取出完整 predicate chain。
2. 每個 `Predicate` 透過 `get_formula()` 轉成 SMT assertion。
3. `Solver._build_formulas_from_constraint()` 組成完整 SMT-LIB2 程式。

其產生出的公式大致包含四個部分：

- `declare-const`：宣告所有可符號輸入變數，如 `v_12_7_0_VAR`
- `queries`：path 上所有 branch predicate 的 assert
- `norm_queries`：若有啟用 `norm` 或 range limit，則加入輸入範圍限制
- `get-value`：要求 solver 回傳各符號輸入的新值

簡化後，公式會長得像：

```text
(set-logic ALL)
(declare-const v_12_7_0_VAR Real)
(declare-const v_12_8_0_VAR Real)
(assert (> (+ (* 0.31 v_12_7_0_VAR) (* -0.18 v_12_8_0_VAR) 0.07) 0))
(assert (<= v_12_7_0_VAR 1))
(assert (>= v_12_7_0_VAR 0))
(check-sat)
(get-value (v_12_7_0_VAR))
(get-value (v_12_8_0_VAR))
```

實際公式通常會更長，因為 branch expression 可能來自多層神經網路計算後的結果。

### 5.7 Branch position 怎麼跟 SHAP 對齊

本專案不是只收集 constraint，還會嘗試知道「這個 constraint 對應到網路哪一層、哪個位置」。

做法是：

- `NNModel.forward()` 在每層執行前註冊目前的 Keras layer 編號。
- 各 layer 在計算神經元、token 或 attention query 時，呼叫 `register_current_indices(...)` 記錄當前位置。
- 當 branch 真的被觸發時，`PathToConstraint.add_branch()` 會一併抓取這個 `(layer_number, indices)`。
- 之後 `push_constraint()` 就可以用這個 position 去查 SHAP influence，進一步決定 queue priority。

因此 queue 中的元素其實不只是 path constraint，還帶有「這個 branch 是在哪個 layer / neuron / token 上發生」的定位資訊。

## 6. 核心技術亮點

### 6.1 SHAP-guided priority queue

這是本專案最重要的演算法設計之一。當 queue mode 為 `priority_queue` 時，constraint 並不是單純依加入順序處理，而是依下列 score 排序：

```text
score = (1 - alpha) * log10(|shap_value| + eps)
      - alpha * log10(path_len + 1)
```

直觀上，這個公式同時平衡兩種偏好：

- 偏好 SHAP 值較大的位置，因為這些位置更可能影響模型預測。
- 懲罰 path length 較長的 constraint，因為它們通常代表更深、更難解的路徑。

這個設計比單純依 SHAP 排序更穩健，也比純 FIFO/LIFO 更有目標性。

### 6.1.1 為什麼 SHAP 可以綁到 constraint 上

要做到這件事，前提是 branch 需要能映射回 layer position。這正是本 repo 比一般 solver queue 更完整的地方：

- branch 不是匿名的布林條件，而是帶有 `position=(layer, indices)`。
- `ShapValuesComparator` 可依 `(layer, indices)` 查詢對應 SHAP value。
- 單一 position 使用 `abs(target-class influence)`。
- 若位置是一組 indices，例如 attention query 對整個 feature vector 的定位，
  則使用 `abs(mean(signed target-class influences))`。正負方向互相抵消時，
  代表這組 feature 沒有一致的 target-class influence，因此不會被
  `mean(abs(...))` 人為放大。
- 不直接排序的 output layer 使用有限分數 `0.0`，避免非有限 sentinel 經過
  `abs()` 後意外成為最高優先級。

因此，constraint priority 並不是「某個 path 的抽象分數」，而是「某條 path 中最新 branch 對應到的神經位置重要性」。

### 6.2 Multi-stage `ton` orchestration

每個 case 都可以包含多個 `ton` stage。系統不是將其視為互不相關的獨立實驗，而是當成可恢復、可決策的連續攻擊流程：

- 成功找到 adversarial input：停止後續 stage。
- solver 把當前 constraint 解完但未成功：進入下一個 `ton`。
- timeout：也可以選擇繼續下一個 `ton`。
- incomplete：保留重跑機會。

stage 狀態會回寫到 `stats.json` 的 `meta.ton_progress`，並額外記錄到 `stats_history.jsonl`，因此整個探索過程是可追蹤的。

### 6.3 結構感知的 SHAP selector

本專案不只支援 `pixel-shap`，也支援：

- `patch-shap`
- `token-shap`

其關鍵不是多一個 flag，而是 `explainability/pixel_provider.py` 會分析 model config，推斷 tokenizer 類型：

- `patch_2d`：例如 Conv2D patch embedding + Reshape
- `sequence_pool_1d`：例如 token sequence + AveragePooling1D

基於此，系統可以：

- 將 patch 內所有像素的 SHAP 分數聚合，選出最重要 patch。
- 將 token group 內所有像素的 SHAP 分數聚合，選出最重要 token group。

這使得 SHAP selection 不再綁死在 pixel 級別，而能與模型的輸入分塊方式對齊。

### 6.4 Transformer symbolic execution support

為了讓 concolic testing 能跑在 transformer 類模型上，repo 補了兩層能力：

#### (1) Keras 模型相容載入

`engine/predictor_runtime.py` 為下列自訂 layer 提供 compatibility objects：

- `AddClsToken`
- `AddPositionEmbedding`
- `ExtractClsToken`
- `DropPath`
- `SequencePooling`

此外，它還會將 Keras graph 轉成自訂 `NNModel` 所需的 layer 與 inbound 依賴圖。

這兩個 model runtime 的責任不同：

- 原始 Keras model 是 label 的 authoritative reference。
- 自訂 `NNModel` 只負責 concolic forward、symbolic path 與 constraint 生成。

`original_label` 與 `attack_label` 都只會由同一個 Keras model 的
`Model.predict(..., verbose=0)` 產生。即使 `NNModel.forward()` 得到不同
class，只要 Keras prediction 沒有改變，該 candidate 就不算
adversarial。

#### (2) Symbolic-safe forward path

`dnnct/myDNN.py` 針對 transformer 補了多個 layer 的純 Python 推論實作，例如：

- `LayerNormLayer`
- `AddPositionEmbeddingLayer`
- `AddClsTokenLayer`
- `SequencePoolingLayer`
- `MultiHeadAttentionLayer`
- `AveragePooling1DLayer`

其中幾個關鍵技巧包括：

- 使用 concrete-guided piecewise linear `exp` approximation，減少 symbolic expression 膨脹。
- 將 softmax 保持為 concrete 值，避免注意力權重導致約束不可控地爆炸。
- 對 GELU 採用 piecewise linear approximation，提高 symbolic stability。

這些技巧對於讓 transformer 進入 concolic workflow 是必要的工程補強。

### 6.4.1 Transformer constraint 大概長什麼樣

若從 solver 角度看，transformer constraint 仍然是輸入變數上的代數式與比較式，只是 expression tree 會更深。

#### (1) 經過線性層或 patch projection 後

最常見的 constraint 形式仍然是 affine expression：

```text
(> (+ (* w1 v_0_0_0_VAR) (* w2 v_0_0_1_VAR) ... b) 0)
```

這類約束常來自：

- Dense / Conv / patch embedding
- Add / residual connection
- Position embedding 相加

#### (2) 經過 LayerNorm 後

本 repo 的 `LayerNormLayer` 採 concrete mean / variance，但保留 centered term 的 symbolic 性質，因此 constraint 會近似為：

```text
(> (+ (* gamma_i (* inv_std (- x_i concrete_mean))) beta_i) 0)
```

這代表：

- mean 與 variance 不再是符號化約束的一部分
- 但每個 normalized channel 對輸入變數的依賴仍然保留

這是一種精度與可解性之間的折衷。

#### (3) 經過 Multi-Head Attention 後

MHA 中最危險的部分是 softmax。若完全符號化 softmax，公式大小與非線性都會快速失控，因此本 repo 採取：

- Q/K/V 線性投影保留 symbolic 計算
- score matrix 的 softmax 改用 concrete 值
- 後續 context vector 視為「以 concrete attention weight 加權的 symbolic value sum」

因此 attention 後的 constraint 通常不是完整 softmax 公式，而更像：

```text
(> (+ (* 0.41 qproj_expr_1) (* 0.27 qproj_expr_2) (* 0.32 qproj_expr_3)) c)
```

也就是說，softmax 權重已經 concretize，剩下的是一組 concrete coefficient 對 symbolic token features 的線性組合。

#### (4) 經過 GELU / sigmoid / tanh 後

這些 activation 若完整保留原始非線性，solver 壓力會非常大，所以 repo 使用 symbolic-safe approximation：

- GELU：piecewise linear
- sigmoid / tanh：飽和式或近似式
- exp：concrete-guided piecewise linear approximation

因此 transformer 路徑上的 constraint 整體呈現為：

- 以 affine expression 為主
- 穿插少量 piecewise / ite 結構
- 避免直接把 softmax 與複雜浮點非線性完整編入 SMT

這也是本 repo 能把 transformer 放進 concolic 流程的關鍵原因。

### 6.4.2 不是所有運算都會保留 symbolic 表達式

這個 repo 的策略是「能安全保留 symbolic 的保留，風險太高的就 concrete 化」。因此：

- 某些不易映射到 SMT 的運算，可能直接退回 primitive value
- 某些統計量，如 LayerNorm 的均值與方差，會用 concrete 值
- attention softmax 權重也會用 concrete 值

這不是缺陷，而是系統設計上的明確取捨。否則 expression tree 會迅速膨脹到 solver 難以處理的程度。

### 6.4.3 SAT candidate 的 label verification

每個 case 的執行順序固定如下：

1. Keras reference model 對原始輸入做 prediction，保存
   `original_label`。
2. `NNModel` 執行第一次 concolic forward，建立 constraint queue。
3. solver 產生未嘗試過的 SAT candidate。
4. Keras reference model 對 candidate 做 prediction。
5. 若 candidate label 與 `original_label` 不同，保存
   `attack_label` 與 `adv_input.npy`，並停止該 case。
6. 若 label 相同，才讓 `NNModel` 對 candidate 做下一次 concolic
   forward，繼續生成 constraints。

這個順序同時適用於 ternary 與 non-ternary search。search runtime
不會把自己的輸出重用成 verification label。`random-assign` 也使用
同一個 Keras prediction 契約比較修改前後 label。

Keras reference model 會依 model path 在每個 worker process 內快取；
不同 ternary threshold 的 `NNModel` search runtime 共用同一份 reference
model。若 reference model 載入或 prediction 失敗、輸入包含
NaN/Inf、輸出包含 NaN/Inf，case 會 fail closed，並在 `stats.json`
記錄：

- `status: "error"`
- `error_type: "reference_prediction_failure"`
- `error_phase: "reference_model_load"`、`"original_reference"` 或
  `"candidate_reference"`

成功 case 的 metadata 會包含
`label_source: "keras_model_predict"`；reference prediction 次數、phase
與 wall time 則記在 `summary`。舊實驗結果不會因此自動獲得新的 label
語意保證，需要使用 adversarial replay 重新檢查。

### 6.6 `symbolic-path-threshold` 的作用

這個參數是 repo 中非常重要、但容易被忽略的控制閥。

#### 作用時機

每當一個 `ConcolicBool` 被轉成布林值時，engine 會先檢查目前 path 長度，也就是目前 `current_constraint.height`。若：

```text
current_height >= symbolic_path_threshold
```

則 engine 會把 `symbolic_enabled` 設成 `False`。

#### 關閉 symbolic 後會發生什麼

一旦 symbolic 被關閉：

- `PathToConstraint.add_branch()` 不再記錄新的 branch
- 後續不再生成新的 predicate / constraint
- `ConcolicObject(...)` 會直接回傳 primitive value，而不是新的 concolic wrapper
- 後續運算繼續執行，但只剩 concrete execution，不再擴張 expression tree

也就是說，threshold 不是停止整個 execution，而是停止更深層的 symbolic tracking。

#### 為什麼它重要

對深層網路尤其是 transformer 而言，若讓 symbolic expression 無限制往後傳播，會出現幾個問題：

- branch 數量快速增加，queue 爆炸
- expression tree 深度變大，constraint build time 變長
- SMT formula 大小與 assert 長度持續上升
- 即使 solver timeout 設很短，牆鐘時間仍可能被 formula build 吃掉

`symbolic-path-threshold` 的效果就是強制系統在「夠深」之後轉回 concrete mode，避免：

- 新 queue 項目繼續增加
- 更深的非線性與 attention 路徑被完整符號化
- 單個 constraint 變得過大、過深、不可解

#### 實際上的取捨

它帶來的是標準的 completeness / tractability trade-off：

- threshold 太高：
  - 探索較完整
  - 但 queue、build time、solver cost 都可能失控
- threshold 太低：
  - 執行更穩定
  - 但可能過早失去深層 branch 的探索機會

因此這個參數不是單純的 performance tweak，而是直接決定 symbolic exploration 深度上限的系統級控制參數。

### 6.5 完整的 per-case observability

每個 case 都會寫出完整 artifacts，而不是只有成功或失敗：

- `stats.json`
- `stats_history.jsonl`
- `ori_input.npy`
- `ori_input.jpg`
- `adv_input.npy`
- `sat_inputs.npy`
- `adv_<label_from>_to_<label_to>.jpg`

`stats.json` 至少包含：

- `meta`: 狀態、標籤、timeout、stage metadata
- `summary`: wall time、cpu time、iteration totals
- `solver`: sat/unsat/unknown 統計
- `constraints`: generated/solved/queue max
- `iters_summary`: 每輪分佈摘要

這使得 repo 不只適合「跑出結果」，也適合做 post-hoc analysis。

## 7. 實驗輸出與命名設計

實驗目錄採參數化命名，且固定建立在 **repo root** 下：

```text
<repo_root>/exp/<model>_<attack_mode>_<timeout>_<build_timeout>_<alpha_tag>_<symbolic_threshold>/case_<idx>/
```

這個命名策略的價值在於：

- 不同實驗設定天然分流，不易覆蓋。
- 不必打開檔案就能辨識 timeout / alpha / threshold。
- 有利於批量比較與自動統計。

對研究型 repo 來說，這比單一輸出資料夾更容易維護。

除 `exp/` 外，constraint pop log 也會固定寫到：

```text
<repo_root>/popped_constraint_position/<model_name>/<attack_mode>_<ton>/
```

這樣即使 runtime 在執行中切換 `cwd`，輸出仍會穩定落在 repo root，不會漂移到 `engine/exp/` 或其他子目錄。

## 8. 結果分析流程

`reporting/experiment_stats.py` 會遞迴掃描 `stats.json`，並聚合：

- success / timeout / exhausted / incomplete 數量
- 多種 timing metric
- iter count 分佈
- constraint complexity 統計

它也支援 `--split-by-status`，可把不同 outcome 分開分析。這對比單純平均值更有意義，因為 success case 與 timeout case 的時間分佈通常差異很大。

## 9. 可重現性設計

本專案已經有數個有助於實驗重現的設計：

- SHAP background sampling 使用固定 seed。
- random baseline 的座標生成可設定 `--random-seed`。
- 實驗目錄名包含核心超參數。
- `force_refresh` 與 skip-existing 邏輯可避免無意義重跑。
- 每個 case 都保存原始輸入與 SAT 輸入。

若後續要強化可重現性，可以再補：

- 統一版本紀錄，例如 TensorFlow、cvc5 與 Python 版本寫入 `stats.json`。
- 在 experiment root 額外保存完整 CLI command。
- 將 git commit hash 納入 metadata。

## 10. 已知限制

這是一個實驗框架，不是泛化到所有模型結構的通用測試平台。目前至少有以下限制：

- `patch-shap` 與 `token-shap` 目前只支援 CIFAR10，且僅支援 `--pixel-search 1`。
- SHAP selector 依賴 model config heuristic；若模型 tokenizer 結構過於特殊，可能需要手動補 override。
- solver 主要預設為 `cvc5`，其他 solver 的支援與穩定性不是此 repo 的主線。
- transformer symbolic path 的數值近似是工程上的折衷，重點是可探索性與穩定性，而不是完全精確重建原始浮點運算。
- `symbolic-path-threshold` 會在深路徑處停止新增 symbolic branch，因此結果應理解為「受控深度下的 concolic exploration」，不是無界完整探索。
- 目前重點資料集是 `fashion_mnist`、`mnist`、`cifar10`；擴展到新資料集仍需自行補 task generator 與 dataset adapter。

## 11. 建議的閱讀順序

如果讀者想快速理解程式碼，建議依下列順序閱讀：

1. `README.md`
2. `pyct/main.py`
3. `pyct/args.py`
4. `orchestration/launcher.py`
5. `tasks/builders/`
6. `engine/executor.py`
7. `libct/explore.py`
8. `libct/record.py`
9. `engine/predictor_runtime.py`
10. `dnnct/myDNN.py`
11. `reporting/experiment_stats.py`

若只想理解 SHAP selector 與 transformer 支援，可優先讀：

1. `explainability/pixel_provider.py`
2. `pyct/shap.py`
3. `engine/predictor_runtime.py`
4. `dnnct/myDNN.py`

## 12. 總結

`PyCT-Influence-Transformer` 的技術價值不只在於「把 SHAP 接到 concolic testing 上」，而在於它把多個難以直接兼容的部分系統化整合起來：

- SHAP-guided search
- SMT-based concolic exploration
- transformer symbolic forward support
- multi-stage attack scheduling
- experiment artifact management
- post-run statistical analysis

因此，這個 repo 最適合作為「研究型測試框架」來介紹，而不是單一攻擊演算法實作。若要對外說明其亮點，建議聚焦在「結構感知的 SHAP-guided concolic testing pipeline」這個主軸。
