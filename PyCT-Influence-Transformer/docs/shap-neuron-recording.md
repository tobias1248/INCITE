# 各層 Neuron SHAP Value 計算技術說明

本文件依據
[`explainability/shap_calculator.py`](../explainability/shap_calculator.py)
說明專案如何為「每一層每一個 neuron 位置」產生可查詢的影響分數，
並落地成 JSON cache。實作上有兩條主要路徑：

- `Sequential` 模型：逐層建立可解釋子模型，主要使用 `shap.GradientExplainer`。
- `Functional/DAG/Residual` 模型：保留完整圖結構，輸入層仍算 SHAP，中間層改算 branch influence。

目前實務上最重要的是：

- `3.2 Sequential` 下 `GradientExplainer` 的運作方式。
- `4. DAG/Residual` 下的 branch influence 計算方式。

---

## 1. 設計目標與輸出格式

`ShapValuesCalculator` 的目標不是只產生輸入像素的 SHAP，而是要把「模型內部任一層某個位置」的影響值記錄下來，供後續 `ShapValuesComparator` 與 priority queue 使用。

輸出位置：

```text
shap_target_class/<model_name>/shap_value_<idx>.json
```

輸出內容是 key-value map。key 由 `get_position_key(layer_number, indices)` 組成，例如：

- `-1_12_7_0`：輸入空間某個位置。
- `3_5_9`：第 3 個 tracked layer 某個 neuron 位置。

JSON 外層的 `__meta__` 會記錄 attribution contract 與 provenance，例如：

```json
{
  "__meta__": {
    "schema_version": 2,
    "attribution_target": "original_prediction",
    "target_class": 7,
    "class_count": 10,
    "background_per_class": 5,
    "background_seed": 1234,
    "case_index": 0
  },
  "values": {
    "-1_12_7_0": 0.031,
    "3_5_9": -0.114
  }
}
```

---

## 2. 計算入口

核心類別是 `explainability/shap_calculator.py` 中的
`ShapValuesCalculator`。

初始化時它會：

1. 載入模型與 custom objects。
2. 保留 `background_dataset` 與 `input_data`。
3. 判斷模型是否為 `Sequential`。
4. 建立 `self._tracked_layers`，排除 `InputLayer`、`Dropout`、`Embedding`。

核心分流發生在 `_compute_shap_values()`：

- 若 `self._layerwise_enabled == True`，走 `Sequential` 逐層 SHAP。
- 否則走 `Functional/DAG` 路徑：
  1. 先算輸入層 SHAP。
  2. 再補各中間層 branch influence。

這個分流是整份文件最重要的背景，因為兩條路徑對「各層 neuron 分數」的定義並不相同。

---

## 3. Sequential 模型：逐層 SHAP

### 3.1 整體流程

當模型是 `Sequential` 時，程式會認定可以安全地把模型一層一層往前裁切。主循環如下：

1. 用目前的 `trimmed_model` 對目前表示 `transformed_input` 算一次 layer SHAP。
2. 用 `apply_one_layer()` 把單筆輸入 forward 一層。
3. 用 `apply_one_layer_to_dataset()` 把背景資料也 forward 一層。
4. 若還沒到最後一層，就用 `without_first_layer()` 把模型最前面那層切掉，進入下一輪。

因此第 `k` 次迭代裡：

- `trimmed_model` 代表「從第 `k` 層開始到輸出的子模型」。
- `transformed_input` 代表原始輸入經過前 `k` 層之後的中間表示。
- `transformed_background` 代表背景資料在同一層表示空間下的 baseline。

這個設計成立的前提是：Sequential 模型只有單一路徑，不存在 residual merge 或共享張量。

### 3.2 `GradientExplainer` 如何動作

這是目前主要使用的計算方式。對應實作在 `_calculate_layer_shap_values(..., explainer_type="gradient")`。

每一層的流程可精確拆成以下步驟。

#### 3.2.1 準備 explainer 的輸入

程式會先把背景資料包成 list：

```python
if isinstance(background_dataset, list):
    bg_for_shap = background_dataset
else:
    bg_for_shap = [background_dataset]
```

然後建立：

```python
explainer = shap.GradientExplainer(model, bg_for_shap)
```

這裡的 `model` 並不一定是原始完整模型，而是當前迭代中的 `trimmed_model`。因此在不同 layer iteration 中，`GradientExplainer` 實際看到的是不同的子模型：

- 第 0 輪：原始輸入 -> 最終輸出。
- 第 1 輪：第 1 層表示 -> 最終輸出。
- 第 2 輪：第 2 層表示 -> 最終輸出。

也就是說，這裡不是只對輸入做一次 SHAP，而是在每一層表示空間上各自做一次 SHAP。

#### 3.2.2 `shap_values(input_data)` 回傳的是什麼

接著會呼叫：

```python
gradients = explainer.shap_values(input_data)
```

這個 `input_data` 是當前層的表示，可能是：

- 原始輸入張量。
- 經過前幾層後的 activation tensor。

`GradientExplainer` 的輸出 shape 依 SHAP 版本與模型輸出格式而異，實作因此
不直接假設固定 class axis，而是交給 `select_target_class_values()` 驗證並
選出固定 target class。

#### 3.2.3 如何選出固定 target class

calculator 載入模型後，先對未修改的 case 執行一次 Keras prediction：

```python
predictions = model.predict(original_input)
target_class = np.argmax(predictions[0])
```

這個 `target_class` 在同一個 case 的整次 attribution 計算中保持固定。後續
`GradientExplainer` 回傳多個 class outputs 時，
`select_target_class_values()` 只取出這個 class：

```python
target_values = select_target_class_values(
    gradients,
    target_class=target_class,
    batched_input_shape=input_data.shape,
)
```

這裡沒有跨 classes 平均。batch size 必須對應目前的一筆 case，選出的
`target_values` 與當前層表示的 feature shape 一致，每個位置保存一個
對原始預測類別有方向的 scalar：

- 正值表示支持原始預測類別。
- 負值表示抑制原始預測類別。
- magnitude 表示影響強度。

#### 3.2.4 如何把 attribution 寫進 cache

選出 target class 後，程式用 `np.ndenumerate(target_values)` 逐點寫入：

```python
shap_values[self.get_position_key(layer_number - 1, indices)] = float(value)
```

這裡有一個非常重要的規則：寫入時用的是 `layer_number - 1`。

因此：

- 第 0 輪寫出的 key 會是 `-1_*`，代表輸入空間。
- 第 1 輪寫出的 key 會是 `0_*`，代表原模型的第 0 個隱藏表示。
- 第 2 輪寫出的 key 會是 `1_*`，依此類推。

這個 offset 是後續 comparator 需要做 fallback lookup 的原因之一。

#### 3.2.5 為什麼這樣能得到「各層 neuron SHAP」

直觀上，這條路徑在每一輪都在回答同一件事：

「若我把目前這一層表示中的某個 neuron 位置往背景 baseline 的方向移動或偏離，它對最終輸出有多大影響？」

因為每次都對「當前層表示 -> 最終輸出」這個子模型建立 `GradientExplainer`，所以結果可以被解讀為：

- 在固定後半段網路不變的前提下，
- 目前層表示中的每個位置，
- 對最終輸出的 attribution。

對 Sequential 模型來說，這種分層切法是合理的，因為各層之間沒有旁路與回流。

### 3.3 與 Kernel explainer 的關係

`kernel` 分支仍然保留，但目前不是主要路徑。它的做法是：

1. 若輸入 rank 大於 2，先 flatten。
2. 在模型前加一層 `Reshape` 還原原形狀。
3. 使用 `shap.KernelExplainer`。
4. 再把 flat index 映回原本的多維座標。

相較之下，`GradientExplainer` 不需要 flatten，對目前常見的神經網路模型更直接，因此是主要使用模式。

---

## 4. Functional / DAG / Residual：輸入 SHAP + Branch Influence

### 4.1 為什麼不能直接沿用 Sequential 的逐層切法

對 `Functional` 或帶 residual/skip connection 的模型，`_compute_shap_values()` 明確註解：

> Functional/DAG models are not safe for "drop first layer" slicing.

原因不是單純「程式不好寫」，而是語意上不成立。對 DAG 模型而言，中間層 activation 常常：

- 被多條路徑共同使用。
- 在之後與 skip branch 匯合。
- 依賴完整圖結構才能定義它對輸出的作用。

若硬把第一層切掉、再拿剩下子模型去做逐層 SHAP，可能會出現兩種問題：

1. 子模型的輸入不再是圖中真正的合法節點語意。
2. 原本需要其他分支共同決定的貢獻，被錯誤地歸到單一路徑上。

因此目前實作對 DAG 採用雙步驟：

1. 輸入層仍使用標準 SHAP。
2. 中間層改用 branch influence，也就是 `gradient * (activation - baseline)`。

### 4.2 Step 1：先保留輸入層 SHAP

即使模型不是 `Sequential`，程式仍然會先執行：

```python
self._calculate_layer_shap_values(
    shap_values,
    self._model,
    self.background_dataset,
    self.input,
    0,
)
```

因此輸入層仍然有一份標準 SHAP cache，寫入 key 為 `-1_*`。這部分與 Sequential 第 0 輪的意義一致。

### 4.3 Step 2：各中間層的 branch influence

接著進入 `_calculate_functional_layer_branch_influence()`。這是 DAG/Residual 路徑目前最主要的中間層分數來源。

#### 4.3.1 收集可追蹤的 layer output

初始化時已建立：

```python
self._tracked_layers = [
    layer for layer in self._model.layers
    if type(layer).__name__ not in ("InputLayer", "Dropout", "Embedding")
]
```

真正計算時還會再過濾一次，只保留：

- 有 `layer.output`
- 且 `layer.output` 不是 list/tuple

的 layer。結果存在 `output_specs`，其元素是：

```python
(layer_index, layer_output)
```

這裡的 `layer_index` 是 `tracked_layers` 內的索引，不是原始 Keras graph 的全域 layer 編號。

#### 4.3.2 建立同時吐出各層 activation 與最終 logits 的模型

程式把所有 tracked layer output 加上模型最終輸出，一起包成：

```python
feature_model = Model(
    inputs=self._model.inputs,
    outputs=[layer_outputs..., model.output],
)
```

這樣只要 forward 一次，就能同時拿到：

- 每一層的 activation
- 最終輸出 logits

這是後續同時計算 baseline 與 gradient 的基礎。

#### 4.3.3 為輸入樣本與背景資料各跑一次 forward

先對背景資料做 forward：

```python
background_out = feature_model(background, training=False)
background_acts = list(background_out[:-1])
```

再在 `GradientTape` 中對目前樣本做 forward：

```python
input_out = feature_model(x, training=False)
input_acts = list(input_out[:-1])
logits = input_out[-1]
```

此時：

- `background_acts[i]` 是第 `i` 個 tracked layer 在背景資料上的 activation batch。
- `input_acts[i]` 是當前待解釋樣本在同一層的 activation。

#### 4.3.4 target 是怎麼選的

`GradientTape` 不是直接對整個輸出向量求梯度，而是使用 calculator
初始化時由原始 Keras prediction 決定的固定 target class：

```python
target = tf.reduce_mean(
    tf.gather(logits, self._target_class, axis=1)
)
```

即使某個中間計算或候選輸入的即時 argmax 不同，target 也不會跟著改變。
branch influence 因此一致表示「對原始預測類別」的支撐或抑制，而不是
所有類別平均，也不是每一層重新選一次 argmax。

#### 4.3.5 為什麼要 `tape.watch(input_acts)`

對每個 activation tensor，程式執行：

```python
for act in input_acts:
    tape.watch(act)
```

原因是這些 activation 不是 trainable variable，而是 forward 過程中的中間 tensor。若不明確 watch，`GradientTape` 不會保證回傳 `target` 對它們的梯度。

接著：

```python
grads = tape.gradient(target, list(input_acts))
```

就能得到每一層 activation 對 target 的局部敏感度。

這裡的 `GradientTape` 可以把它理解成 TensorFlow 的「自動微分錄影機」：

1. `with tf.GradientTape() as tape:` 內部發生的 forward 計算會被記錄下來。
2. `tape.watch(act)` 告訴 TensorFlow 要追蹤哪些中間 activation tensor。
3. `tape.gradient(target, act)` 會回傳：

```text
∂ target / ∂ act
```

也就是「若這個中間 neuron 稍微改變，最終目標分數會變多少」。

因此在 DAG 路徑裡，`GradientTape` 的工作不是計算 SHAP，而是量測每個中間 activation 對目標輸出的局部斜率。後面的 branch influence 則再把這個斜率與 activation 相對背景的偏移量結合起來。

#### 4.3.6 branch influence 的正式定義

對每個 tracked layer，程式計算：

```python
baseline = tf.reduce_mean(act_background, axis=0)
influence = grad[0] * (act_input[0] - baseline)
```

可寫成：

```text
influence_l(i) = d target / d a_l(i) * (a_l(i) - E_bg[a_l(i)])
```

其中：

- `a_l(i)`：當前樣本在第 `l` 層第 `i` 個位置的 activation。
- `E_bg[a_l(i)]`：背景資料在同一位置的平均 activation。
- `d target / d a_l(i)`：該位置對目標輸出的局部梯度。

這個式子本質上是 gradient × activation-deviation。它不是標準 SHAP 公理下的 layer attribution，而是專門為 DAG/Residual 模型設計的可解釋替代量。

若從近似觀點來看，它接近一階泰勒展開：

```text
f(a) ≈ f(a0) + ∇f(a0) · (a - a0)
```

其中：

- `a0` 可視為背景 activation 的平均。
- `a - a0` 是目前樣本相對 baseline 的偏移。
- `∇f(a0)` 或其局部近似對應到這裡用梯度量到的敏感度。

因此這個分數更接近「局部線性近似下的貢獻量」，而不是經典 Shapley value。

#### 4.3.7 這個量的直觀意義

這個分數同時結合兩件事：

1. 這個 neuron 在當前樣本上是否偏離背景平均很多。
2. 這個 neuron 若發生微小變動，是否會明顯推動目標 logit。

因此：

- 大正值：此 neuron 的當前活化相對背景更強，且方向上支持目前預測。
- 大負值：此 neuron 的當前活化方向上抑制目前預測。
- 接近 0：要嘛偏離背景很小，要嘛梯度很小，要嘛兩者互相抵消。

#### 4.3.8 寫入 cache 的方式

計算完 `influence` 後，程式會對 tensor 逐點展開：

```python
for indices, value in np.ndenumerate(influence_np):
    shap_values[self.get_position_key(layer_index, indices)] = float(value)
```

與 Sequential 路徑不同，這裡不再使用 `layer_number - 1`，而是直接用 `tracked_layers` 的 `layer_index`。因此在 DAG 模型裡常見：

- `-1_*`：輸入層 SHAP。
- `0_*`, `1_*`, `2_*`, ...：tracked layer 的 branch influence。

### 4.4 為什麼這種方法適合 residual / multi-branch 結構

這個方法的好處在於它完全不需要破壞原始圖結構：

- skip connection 仍然存在。
- add/concat merge 仍然照原圖計算。
- 每一層的 influence 都是在完整 forward graph 上量測。

因此它比「把 DAG 強行改寫成逐層 Sequential 子模型」更穩定，也更符合 residual 網路裡 branch contribution 的語意。

### 4.5 與 Sequential Gradient SHAP 的核心差異

兩條路徑都會輸出「每層每個位置一個 scalar」，但其意義不同：

- Sequential `GradientExplainer`
  - attribution 對象是「目前層表示作為子模型輸入」。
  - 每一層都重新建一個 `GradientExplainer`。
  - 比較接近 layerwise SHAP。

- DAG branch influence
  - attribution 對象是「完整圖中某中間 activation 對 target 的局部貢獻」。
  - 只在完整圖上做一次 activation/gradient 分解。
  - 比較接近 gradient × deviation。

所以在閱讀 cache 時，不能把 DAG 中間層的值完全視為經典 SHAP；它們是為了 residual/functional 網路而引入的 SHAP-like neuron influence。

### 4.6 DAG 方法與真正 SHAP 的差異

這一段很重要。雖然文件中習慣把這些值統稱為 neuron SHAP value，但在 `Functional/DAG/Residual` 路徑裡，中間層實際上不是由 `shap` 套件算出的 Shapley value。

真正的 SHAP 在問的是：

```text
若把某個 feature/neuron 視為參與者，
它在各種 feature 子集合中加入時，平均帶來多少邊際貢獻？
```

因此標準 SHAP 具有以下特徵：

- 它來自 coalition / subset 的平均邊際貢獻。
- 它依賴對「缺失特徵」的分佈或 baseline 定義。
- 在適當條件下，各 attribution 的總和會對齊輸出差異。
- 它滿足 Shapley value 相關公理，而不只是局部敏感度。

相對地，本專案 DAG 中間層的方法是：

```text
influence = gradient × (activation - background_mean)
```

它與真正 SHAP 的差異在於：

1. 它不枚舉 feature/neuron coalition。
2. 它只看目前輸入附近的局部梯度。
3. 它用的是單一 activation baseline，而不是完整的缺失特徵條件期望。
4. 它不保證滿足 SHAP 的 additivity / symmetry / consistency 等理論性質。
5. 它的品質會受梯度飽和、局部非線性與 baseline 選擇影響。

因此更精確地說，DAG 中間層的值不是「真正的 SHAP」，而是「在保留 DAG 結構前提下，基於梯度的 SHAP-like neuron influence 近似量」。

### 4.7 為什麼這個 proxy 仍然值得使用

雖然它不是經典 SHAP，但在本專案的工程目標下仍然合理，原因有三個：

1. 它不需要破壞 residual / skip / merge 的完整圖結構。
2. 它能低成本地為每一層每個 neuron 產生可排序的 scalar 分數。
3. 對後續 concolic priority queue 來說，重點是得到穩定、有方向性的 importance signal，而不是嚴格驗證 attribution 公理。

因此這個方法回答的不是：

```text
這個 neuron 的 Shapley value 精確是多少？
```

而是：

```text
這個 neuron 目前偏離背景多少，
且它對目標輸出有多敏感，
所以它此刻大約推動了多少預測分數？
```

對於 DAG/Residual 模型中的 branch prioritization，這種近似通常已經足夠實用。

---

## 5. 後續查詢與使用方式

`ShapValuesComparator` 會載入這些 JSON，後續供 constraint priority 使用。
雖然這不是本文件主題，但有五點要記住：

1. cache 可能同時含有 `-1_*` 與非負 layer index。
2. Sequential 路徑有 layer index offset，因此 comparator 需要做 fallback lookup。
3. pixel selector 依 `abs(target-class influence)` 排序，但保留 cache 中的正負號。
4. 一個 constraint 若對應多個 positions，使用
   `abs(mean(signed target-class influences))`；正負值會互相抵消，而不是先
   各自取絕對值。
5. output layer 使用有限的 `0.0` importance，不使用會在 scheduler 中變成
   `+inf` 的非有限 sentinel。

換句話說，整個系統真正依賴的是「位置 -> scalar influence」這個統一介面，而不是各路徑在數學上完全一致。

---

## 6. 總結

本專案目前的各層 neuron 影響值計算可總結為：

1. `Sequential` 模型使用逐層 `GradientExplainer`。每輪把目前層表示當成
   子模型輸入，只選取原始 Keras prediction class 的 attribution，再把每個
   neuron 的 signed value 寫入 cache。
2. `Functional/DAG/Residual` 模型不做危險的 layer slicing。輸入層仍保留標準 SHAP，中間層則用 `gradient * (activation - background_mean)` 計算 branch influence。
3. 兩條路徑最後都統一輸出成相同的 key-value JSON 介面，讓 concolic engine 可以不區分來源地直接查詢 neuron influence。
