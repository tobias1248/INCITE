# GlobalReal ACES-like Study Guide

這份文件是給「想看懂 GlobalReal ACES-like 路徑，但不熟悉色彩科學、PWL 或 symbolic execution」的讀者。
目標不是讓你背下所有函式，而是讓你能回答：

1. 這次改動為什麼需要 ACES-like transform？
2. 一個 shared `X` 如何同時控制整張 RGB image？
3. concrete image transform 和 solver 看到的 symbolic transform 有什麼不同？
4. 結果中的 gamut mapping、clipping、PWL error 是什麼意思？

本 guide 對應目前 `feat/global-appearance-shift` branch 上的完整實作，核心變更起點與後續修正包括：

```text
a3c1dfa feat(global-real): add ACES-like appearance shifts
41e7f62 fix(global-real): validate ACES-like PWL metadata and error bounds
11198b6 fix(global-real): serialize ACES PWL numbers as SMT literals
14ca4cc test(global-real): cover negative ACES PWL SMT literals
```

工作目錄是 `feat/global-appearance-shift`。目前 branch 上的
目前 branch 在 `14ca4cc` 為止包含 metadata validation、runtime PWL fail-closed、dense-error
regression 與負數 SMT literal regression。讀這份文件時，應以目前 branch 的程式碼為準，
而不是只以最初的 feature commit 為準。

## 0. 先建立整體心智模型

原本的 GlobalReal 是把一個 solver real 變數 `X` 套到每個 pixel：

```text
pixel_new = pixel_original + coefficient[pixel] * X
```

例如：

```text
brightness: coefficient[h,w,c] = 1
contrast:   coefficient[h,w,c] = pixel[h,w,c] - channel_mean[c]
shap-sign:  coefficient[h,w,c] = -1, 0, or +1
```

這種做法的優點是每個 pixel 對 `X` 都是 affine，solver 很容易理解。
缺點是它直接在 RGB channel 上操作。當 R、G、B 到達 0 或 1 的時間不同，
即使原本三個 channel 屬於同一個 pixel，顏色也可能改變。

這次新增的 ACES-like 路徑改成：

```text
one shared X
    -> 整張 image 的 concrete ACES-like transform
    -> 用少量線段近似這個 transform
    -> 每個 RGB channel 都使用同一個 X 的 PWL expression
    -> solver 只需要搜尋一個 real variable
```

最重要的一句話是：

> ACES-like 是 reference transform；PWL 是讓這個 reference transform 能被目前 symbolic engine 使用的近似層。

目前 ACES-like symbolic model 的另一個重要不變量是：整張 image 只有一個
symbolic degree of freedom，也就是 `__pyct_global_x_VAR`。OKLab → OKLCh、local
MINDE gamut mapping 和 chroma bisection 都在 concrete PWL 建構階段執行，不會以
`hypot`、`atan2`、`sin`、`cos` 或 bisection constraint 的形式直接進入 SMT。

這不是完整的官方 ACES output transform，而是以 OKLCh、tone curve 和 gamut mapping 組成的工程近似。

## 1. 建議閱讀順序

不要一開始從 `aces_like.py` 第一行讀到最後一行。建議按照下面順序：

### 第一階段：先看入口

```text
pyct/args.py
orchestration/launcher.py
```

要回答：

- CLI 怎麼選到 `aces-brightness` 或 `aces-contrast`？
- `--aces-pwl-max-segments` 和 `--aces-pwl-error-tolerance` 做什麼？
- launcher 何時把參數傳給 builder？

### 第二階段：看每個 case 怎麼建立

```text
tasks/builders/global_real.py
```

要回答：

- ACES-like case 的 payload 裡有哪些欄位？
- 它為什麼不需要 SHAP cache？
- `global_real_config` 如何記錄 PWL knots 與版本資訊？

### 第三階段：看 concrete reference transform

```text
libct/aces_like.py
```

只先看這幾個 public entry points：

```text
apply_aces_like_transform()
build_adaptive_pwl_approximation()
PiecewiseLinearApproximation
```

不要先糾結四個矩陣的每個係數；先理解資料流。

### 第四階段：看 symbolic bridge

```text
libct/global_real.py
```

重點函式：

```text
validate_global_real_config()
build_concolic_global_real_kwargs()
materialize_global_real_details()
```

### 第五階段：看輸出與測試

```text
libct/record.py
test/test_aces_like.py
test/test_global_real.py
```

這一階段才檢查 diagnostics 是否真的被保存，以及測試是否覆蓋你認為重要的契約。

## 2. ACES-like transform 到底做了什麼

### 2.1 輸入不是任意 RGB

`apply_aces_like_transform()` 要求輸入是：

```text
shape = (..., 3)
finite
每個 channel 都在 [0, 1]
```

這裡的輸入被視為 normalized sRGB encoded values，不是 linear light。

### 2.2 sRGB 轉 linear RGB

程式先使用 sRGB transfer curve 解碼。概念上：

```text
if sRGB <= 0.04045:
    linear = sRGB / 12.92
else:
    linear = ((sRGB + 0.055) / 1.055) ^ 2.4
```

原因是：在 encoded sRGB 上直接做加法，不等於在光線強度上做等量調整。
亮度與色彩外觀相關的操作，至少應先把 transfer function 明確化。

對應程式：

```text
libct/aces_like.py::srgb_to_linear()
libct/aces_like.py::linear_to_srgb()
```

### 2.3 linear RGB 到 OKLab，再到 OKLCh

程式使用固定矩陣將 linear RGB 轉成 OKLab：

```text
OKLab = (L, a, b)
```

再轉成極座標形式 OKLCh：

```text
OKLCh = (L, C, h)

L = lightness
C = chroma / colourfulness
h = hue angle
```

這一步的目的，是把「明暗」、「色彩強度」、「色相」分開。brightness 和
contrast 主要改 `L`，而不是分別對 R、G、B 做不同的加法。

對應程式：

```text
linear_rgb_to_oklab()
oklab_to_oklch()
oklch_to_oklab()
```

## 3. brightness 與 contrast 的實際定義

這裡的 `X` 不是直接加到 RGB，而是控制 Lightness tone curve。

### 3.1 brightness

程式使用 bounded logit curve。對 `0 < L < 1`，概念公式是：

```text
L_new = sigmoid(logit(L) + 2 * X)
```

其中：

```text
logit(L)  = log(L / (1 - L))
sigmoid(z) = 1 / (1 + exp(-z))
```

因此：

```text
X > 0  -> L 增加
X < 0  -> L 降低
X = 0  -> L 不變
```

`L=0` 與 `L=1` 被明確固定在端點，避免 logit 對端點產生無限大。

### 3.2 contrast

contrast 改變 logit curve 的 slope：

```text
slope = exp(X * log(2))
L_new = sigmoid(slope * logit(L))
```

因此：

```text
X > 0  -> 遠離中間值，暗部更暗、亮部更亮
X < 0  -> 靠近中間值，對比降低
```

這個 contrast 定義不是傳統的：

```text
pixel_new = pixel + (pixel - mean) * X
```

它是在 OKLCh 的 `L` 上改變 tone curve slope，所以同一個 pixel 的 RGB
relationship 不會被獨立 channel affine operation 直接拉開。

對應程式：

```text
libct/aces_like.py::_transformed_lightness()
libct/aces_like.py::_logit_tone_curve()
```

### 3.3 色彩為什麼不會完全不變

這個 pipeline 會盡量保持 hue 和 chroma，但不是保證 RGB 完全不變：

1. 改變 `L` 本身就會改變 RGB。
2. tone curve 後的 OKLCh 顏色可能轉回 linear RGB 時超出 sRGB gamut。
3. 超出 gamut 時，程式會降低 chroma。
4. 降低 chroma 會讓顏色變得比較不飽和。

所以正確期待是「色相盡量穩定、明暗改變、必要時降低飽和度」，不是「每個 channel 保持比例」。

## 4. gamut mapping 與 clipping 的差別

### 4.1 gamut mapping

tone adjustment 後，程式把 OKLCh 轉回 linear RGB，檢查每個 channel 是否在 `[0, 1]`。

如果超出範圍，使用固定 `L` 和 `h`，透過二分搜尋降低 `C`：

```text
保留 L
保留 h
降低 C
直到轉回 linear RGB 後落在 [0, 1]
```

程式最多執行 24 次 bisection。這是「先犧牲 saturation，盡量保留 lightness 和 hue」的策略。

對應程式：

```text
libct/aces_like.py::_constant_lightness_hue_gamut_map()
```

### 4.2 hard clipping

二分搜尋後，仍可能因為浮點數誤差有極小 overshoot。程式最後還是會：

```text
result = clip(linear_to_srgb(mapped_linear), 0, 1)
```

這個最後的 clip 是數值保護，不是主要的色彩處理策略。

需要分清楚：

```text
gamut_mapped_pixel_count
    有多少 pixel 需要降低 chroma 才回到 gamut

hard_clipped_channel_count
    最後仍有多少 channel 超出範圍而需要硬截斷
```

前者可能很多，但不代表發生了嚴重的 channel clipping；後者才更接近「仍需硬截斷的 channel 數量」。

### 4.3 diagnostics

`AcesLikeDiagnostics` 會記錄：

```text
mean_hue_drift_degrees
max_hue_drift_degrees
mean_chroma_delta
max_abs_chroma_delta
mean_luminance_delta
max_abs_luminance_delta
gamut_mapped_pixel_count
hard_clipped_channel_count
```

注意：這裡的 hue/chroma/luminance diagnostics 是用 exact reference transform 的結果計算，
不是只看 PWL approximation。

## 5. PWL approximation：為什麼需要它

完整的 ACES-like transform 包含：

```text
transfer curve
matrix conversion
OKLCh conversion
tone curve
gamut bisection
```

這些步驟不能直接用目前 GlobalReal 的簡單 affine expression 表示。
因此每個 case 會先固定一張 input image，對 shared `X` 建立一張 RGB output table，
再用分段線性函數近似它。

### 5.1 knots

PWL builder 一開始一定放入：

```text
X_min
0
X_max
```

把 `X=0` 固定成 knot 很重要，因為它讓原圖 identity 是精確的，不是「經過幾次浮點轉換後大約相同」。

### 5.2 如何切分區段

每個目前區間會測試 31 個固定位置：

```text
1/32, 2/32, 3/32, ..., 30/32, 31/32
```

這是 deterministic dense validation grid；它提供 sampled maximum error，
不是數學上的 formal supremum。

在每個位置比較 exact reference 與同一 interval 的線性 interpolation：

```text
exact = apply_aces_like_transform(image, X)
approximation = lower_rgb + fraction * (upper_rgb - lower_rgb)
error = max(abs(exact.rgb - approximation))
```

如果最大誤差超過 tolerance，就把誤差最大的 interval 拆開。直到：

```text
max_error <= error_tolerance
```

或達到 `max_segments`。達到上限仍不符合誤差，就 fail closed，丟出錯誤，而不是悄悄使用不合格的近似。

對應程式：

```text
_pwl_interval_error()
build_adaptive_pwl_approximation()
PiecewiseLinearApproximation.affine_for_segment()
```

### 5.3 PWL 的限制

PWL 是「固定一張 image 對 X 的 approximation」，不是一個對所有圖片通用的 tone curve。
因此每個 case 都可能有不同的：

```text
pwl_knots
pwl_max_abs_error
segment_count
```

這也是為什麼 builder 在每個 case 建立自己的 `global_real_config`。

## 6. builder 如何建立一個 ACES-like case

入口是：

```text
tasks/builders/global_real.py::cifar10_global_real()
```

這個檔案保留了舊的 affine builder，並用 `_legacy_cifar10_global_real` 保存舊版本；
新的同名函式先檢查：

```text
if shift_kind not in ACES_LIKE_SHIFT_KINDS:
    return _legacy_cifar10_global_real(...)
```

所以：

```text
shap-sign / brightness / contrast
    -> 舊 affine path

aces-brightness / aces-contrast
    -> 新 ACES-like PWL path
```

### 6.1 ACES payload 的重要欄位

builder 會建立：

```text
in_dict
con_dict = {"__pyct_global_x": 1}
global_real_config
save_exp
```

`global_real_config` 會記錄：

```text
transform_mode = "aces-like-pwl"
global_shift_kind
pwl_knots
pwl_max_segments
pwl_error_tolerance
pwl_max_abs_error
aces_like_color_space = "OKLCh-sRGB"
aces_like_curve_version = "oklch-logit-v1"
aces_like_gamut_mapper = "css-color-4-local-minde-v1"
pwl_error_metric = "sampled-max-abs-rgb"
pwl_validator_version = "adaptive-31-point-v1"
pwl_segment_count = len(pwl_knots) - 1
```

這些 metadata 的目的，是讓之後讀 `stats.json` 的人知道結果是由哪一版 reference transform 產生的。

### 6.2 為什麼 ACES-like 不需要 SHAP

SHAP-sign 需要 attribution 來決定每個 pixel 的正負方向；ACES-like 的方向是由整張 image 的
colour transform 決定，所以 builder 不載入 SHAP provider，也不建立 SHAP background dataset。

它仍然使用一個 shared `X`，但所有 pixel 的 coefficient 不再由 `coefficient_by_input` 明確列出，
而是由 PWL table 在 runtime 生成。

## 7. symbolic execution 如何使用 PWL

### 7.1 全部 pixel 共用一個 symbolic X

`build_concolic_global_real_kwargs()` 會建立：

```text
shared_x = ConcolicObject(
    concrete_value,
    "__pyct_global_x_VAR",
    engine,
)
```

這代表 solver 只搜尋一個變數：

```text
__pyct_global_x_VAR
```

不是每個 pixel 一個 X，也不是每個 RGB channel 一個 X。

因此目前 ACES-like CT 保留的 symbolic degrees of freedom 是 1：
`__pyct_global_x_VAR`。每個 RGB output expression 很多，但它們都依賴同一個
shared X；PWL segment 的 nested `ite` 是 branch selection，不是額外的 symbolic variable。

### 7.2 每個 channel 變成 piecewise affine expression

假設某個 channel 在某個 PWL segment 的 slope/intercept 是 `m` 和 `b`，它的 expression 是：

```text
m * shared_x + b
```

多個 segment 會包成 nested `ite`：

```text
ite(shared_x <= knot_1,
    m_0 * shared_x + b_0,
    ite(shared_x <= knot_2,
        m_1 * shared_x + b_1,
        m_2 * shared_x + b_2))
```

因此 solver 看到的不是 OKLab、矩陣或 bisection，而是一個以同一個 `X` 為輸入的分段線性 RGB expression。

對應程式：

```text
libct/global_real.py::_pwl_symbolic_expression()
libct/global_real.py::build_concolic_global_real_kwargs()
```

### 7.3 concrete value 與 symbolic expression 的分工

每次 forward 都同時需要：

```text
concrete value
    Python / model 執行當下真正使用的 float

symbolic expression
    solver 未來改變 X 時要追蹤的 expression
```

ACES-like path 會先呼叫 PWL approximation 的 `evaluate(shift_value)` 取得 concrete RGB，
再用同一張 approximation 的 affine segment 建立 symbolic expression。

這是 reviewer 應特別檢查的 invariant：

> concrete path 和 symbolic path 必須使用同一組 knots 與同一組 segment coefficients。

## 8. materialization 與輸出記錄

當 solver 找到一個 X，`materialize_global_real_details()` 會重新建立該 image 的 deterministic PWL table，
並同時計算 exact reference 與 PWL candidate：

```text
exact = apply_aces_like_transform(rgb, X)
approx = PWL.evaluate(X)
pwl_error_at_x = max(abs(exact.rgb - approx.rgb))
```

如果 candidate 的 exact/PWL RGB error 超過 recorded tolerance，materialization 會 fail closed，
不會把不合格的 candidate 寫入正常結果。通過 error guard 後，實際交給 model 的 input RGB 是 `approx`；
exact transform 則提供 gamut、hue、chroma、luminance diagnostics。

要特別分清楚：SAT candidate 不等於 attack success。SAT 後，
`libct/executor/execution_pair.py::validate_sat_candidate()` 會用 exact Keras/reference
prediction 驗證 candidate label。只有：

```text
original_label != candidate_reference_label
```

才會記錄 `attack_label`，發出 `[RESULT_CHANGE]`，並算作 success。SAT candidate 若 exact label
仍與 original label 相同，就會被 fail-closed 拒絕；後續結果可能是 timeout、exhausted 或 incomplete，
但不能算成功。

`libct/record.py` 會把資訊保存到 metadata 與 arrays，例如：

```text
sat_global_x.npy
sat_global_clipped_count.npy
sat_global_gamut_mapped_pixel_count.npy
sat_global_pwl_error.npy
```

並在 `stats.json` 的 global-real metadata 中保存：

```text
global_real_*_pwl_segment_index
global_real_*_pwl_error_at_x
global_real_*_gamut_mapped_pixel_count
global_real_*_mean_hue_drift_degrees
global_real_*_max_hue_drift_degrees
global_real_*_mean_chroma_delta
global_real_*_max_abs_chroma_delta
global_real_*_mean_luminance_delta
global_real_*_max_abs_luminance_delta
```

## 9. CLI 參數與 output path

`pyct/args.py` 新增：

```text
--global-shift-kind aces-brightness
--global-shift-kind aces-contrast
--aces-pwl-max-segments N（default: 32）
--aces-pwl-error-tolerance T（default: 1/255）
```

ACES-like 目前只接受 `--global-x-bounds-mode clip`；CLI 會拒絕 strict mode。
CLI `--timeout` 的 default 是 3600 秒；它是每個 case 的 exploration total timeout，會同時傳給 total/single/stage timeout。
它不是只限制單次 solver invocation；solver invocation 另由 `--solver-run-timeout` 控制。
目前 source 沒有讀取 `CONCOLIC_TOTAL_TIMEOUT`，unset 這個環境變數不會停用 `--timeout`。
若要增加搜尋深度，`--symbolic-path-threshold` 也很重要；CLI 預設是 8000，過低的值會提早關閉 symbolic tracking。

例如：

```bash
.venv/bin/python -m pyct \
  --dataset cifar10 \
  --model-name resnet18_cifar10_clean \
  --attack-mode global-real \
  --global-shift-kind aces-brightness \
  --global-x-min -0.01 \
  --global-x-max 0.01 \
  --aces-pwl-max-segments 8 \
  --aces-pwl-error-tolerance 0.0039215686 \
  --first-n 1 \
  --num-process 1 \
  --score-alpha 0.8 \
  --timeout 30 \
  --solver-run-timeout 5 \
  --force-refresh
```

Output path 的 attack mode 會包含：

```text
aces-brightness
clip
x range
pwlsegN
pwlerrT
```

這是為了避免不同 PWL 設定共用同一個 experiment directory。

## 10. 建議的 hands-on study steps

### Step 1：先看 commit scope

```bash
git log --oneline --decorate -12
git show --stat a3c1dfa
```

先確認這次改動集中在：

```text
CLI -> builder -> global_real symbolic bridge -> recorder
```

### Step 2：只跑 ACES-like unit tests

```bash
.venv/bin/python -m pytest -q \
  test/test_aces_like.py \
  test/test_global_real.py \
  test/test_pyct_args.py
```

### Step 3：觀察一張小 image 的 transform

你可以在 Python shell 中測試：

```python
import numpy as np
from libct.aces_like import apply_aces_like_transform

rgb = np.asarray([[[0.1, 0.2, 0.8]]], dtype=np.float64)

for kind in ("aces-brightness", "aces-contrast"):
    result = apply_aces_like_transform(rgb, 0.05, kind=kind)
    print(kind)
    print("rgb:", result.rgb)
    print("diagnostics:", result.diagnostics)
```

應觀察：

- output 仍在 `[0, 1]`。
- brightness 正向 X 通常增加 lightness。
- contrast 正向 X 通常使暗部更暗、亮部更亮。
- 若 gamut mapping 發生，`gamut_mapped_pixel_count` 可能大於零。

### Step 4：觀察 PWL 是否真的 pin 住 X=0

```python
import numpy as np
from libct.aces_like import build_adaptive_pwl_approximation

rgb = np.asarray([[[0.1, 0.2, 0.8]]], dtype=np.float64)
table = build_adaptive_pwl_approximation(
    rgb,
    kind="aces-brightness",
    x_min=-0.1,
    x_max=0.1,
    max_segments=8,
    error_tolerance=1.0 / 255.0,
)

print("knots:", table.knots)
print("segments:", table.segment_count)
print("max error:", table.max_abs_error)
print("X=0:", table.evaluate(0.0))
print("original:", rgb)
```

應能回答：

```text
X=0 是否出現在 knots？
X=0 的 output 是否等於原圖？
max_abs_error 是否不超過 tolerance？
```

### Step 5：讀 symbolic expression

對照：

```text
test/test_global_real.py::test_aces_like_symbolic_arguments_use_one_shared_x_and_piecewise_ites
libct/global_real.py::_pwl_symbolic_expression()
```

你應能指出：

- expression 裡只有一個 shared X。
- 每個 RGB channel 只是在使用不同的 slope/intercept。
- segment 邊界由 nested `ite` 決定。
- expression 本身沒有重新執行 OKLab 或 gamut bisection。

## 11. 實驗結果如何判讀

一個 solver `sat` 只表示 symbolic constraint 找到可滿足的 model，不代表 exact model 已經改變 label。
目前的 success contract 是：

```text
SAT candidate
    -> materialize PWL candidate
    -> exact/PWL error guard
    -> exact Keras/reference candidate prediction
    -> original_label != attack_label
    -> [RESULT_CHANGE] / success
```

因此讀 `stats.json` 時應分開看：

- `success`：有 `attack_label`，且 exact reference label 與 original label 不同。
- `timeout`：在 total exploration deadline 到期前沒有 success。
- `exhausted`：constraint queue 已耗盡，仍沒有 success。
- `incomplete`：尚未完成正常 exploration lifecycle。
- solver counters 裡的 `sat` / `unsat` 是 constraint attempts，不等於 result status。

如果有大量 SAT、但 `attack_label=None` 且沒有 `[RESULT_CHANGE]`，正確解讀是
SAT-but-concrete-fail，而不是「solver 全部 UNSAT」。

## 12. 讀完後應該能回答的問題

### 基本理解

1. 為什麼 ACES-like 不直接使用 `pixel + coefficient * X`？
2. 為什麼要先轉 linear RGB？
3. OKLCh 裡的 `L`、`C`、`h` 各自代表什麼？
4. brightness 和 contrast 分別改變哪一個部分？

### Symbolic execution

5. solver 搜尋幾個 X 變數？
6. 為什麼每個 channel 的 expression 不需要自己的 symbolic variable？
7. PWL knot 和 nested `ite` 如何對應？
8. 為什麼 `X=0` 必須是 knot？

### 色彩與輸出

9. gamut mapping 和 hard clipping 的數量差別是什麼？
10. `pwl_error_at_x` 是 exact transform 和哪一個結果的差距？
11. output image 是 exact reference 還是 PWL approximation？
12. 如果 PWL 在 segment cap 內達不到 tolerance，程式應該怎麼做？

### Review 判斷

13. metadata 中的 curve version 和 gamut mapper version 是否真的跟程式一致？
14. builder 建立的 RGB mapping 是否涵蓋所有 `v_h_w_c`？
15. concrete execution 和 solver expression 是否使用相同 knots？
16. `clip`、gamut mapping、hard clipping 是否在報告中被混為一談？

## 13. 目前實作的邊界與 reviewer 應注意的地方

### 13.1 這不是官方 ACES

程式名稱是 ACES-like，實際使用的是：

```text
OKLCh-sRGB
oklch-logit-v1
css-color-4-local-minde-v1
```

它模仿「在 appearance space 做 tone，再做 gamut compression」的架構，
不是 ACES JMh 或官方 output transform 的逐項重現。

### 13.2 exact reference 和 symbolic model 不完全相同

solver 使用 PWL approximation。即使 `pwl_max_abs_error` 很小，solver 找到的 input 仍然是近似模型的結果。
materialization 會在每個 solver candidate 上重新檢查 exact/PWL RGB error，超過 tolerance 就 fail closed。
此外，SAT candidate 還必須通過 exact Keras/reference label validation；SAT 本身不是 success。
因此實驗報告不能只看 label flip，還應看：

```text
pwl_error_at_x
gamut_mapped_pixel_count
hard_clipped_channel_count
hue drift
chroma delta
```

### 13.3 `bounds_mode` 在 ACES-like path 的語意

Affine path 可以使用 `strict` 來根據 coefficient 縮小有效 X interval；ACES-like path 則要求
`bounds_mode=clip`。CLI 與 `validate_global_real_config()` 都會拒絕 ACES-like strict mode，
因為 ACES-like 的 bounded output contract 是透過 concrete tone transform、local-MINDE gamut
mapping 和最終的有限 RGB clamp 來保證，而不是 affine strict interval。

因此 ACES-like 實驗應明確使用：

```text
--global-x-bounds-mode clip
```

### 13.4 PWL 近似是 per-image

不同 image 的 knots 可能不同。不能假設所有 case 都共用相同的 segment boundary，
也不能只從一個 case 的 PWL error 推論整個 dataset 的誤差。

### 13.5 「gamut mapped」不是「顏色完全錯誤」

gamut mapping 表示原本的 tone-adjusted 顏色超出 sRGB 可表示範圍，程式降低 chroma 來保留外觀。
它通常比逐 channel hard clip 更保色，但仍會造成 saturation 或其他 perceptual property 改變。

## 14. 最後的 code review checklist

依序檢查：

```text
[ ] CLI 可以接受兩種 ACES-like shift kind。
[ ] 非 ACES kind 仍走舊 affine path。
[ ] builder 對每個 case 建立 deterministic PWL table。
[ ] PWL knots 包含 x_min、0、x_max。
[ ] PWL validator 使用 adaptive 31-point dense grid，且記錄 segment_count/version。
[ ] PWL error 超標且達 segment cap 時會 fail closed。
[ ] runtime candidate 的 exact/PWL error 也會 fail closed。
[ ] ACES-like input mapping 涵蓋完整 RGB channel axis。
[ ] solver 只看到一個 shared global X。
[ ] 每個 RGB channel expression 都來自同一張 PWL table。
[ ] concrete value 和 symbolic expression 使用同一個 shift model。
[ ] exact diagnostics 與 PWL error 都被保存。
[ ] SAT candidate 會再經 exact reference label validation；SAT 不直接算 success。
[ ] success、timeout、exhausted、incomplete 的 metadata 語意清楚。
[ ] gamut mapping count 與 hard clipping count 沒有混淆。
[ ] ACES-like strict mode 被拒絕，只有 clip bounds mode 可用。
[ ] output path 包含 PWL 參數，避免不同設定互相覆蓋。
[ ] tests 覆蓋 identity、tone direction、out-of-gamut、PWL error、symbolic ite。
```

如果以上問題都能用程式中的函式、config 欄位或測試案例回答，就已經掌握這次改動的主要功能。

## 15. 相關文件

- [ACES-like research and design](aces-like-color-transform.md)
- [Technical overview](technical-overview.md)
- [ACES-like reference implementation](../libct/aces_like.py)
- [GlobalReal symbolic bridge](../libct/global_real.py)
- [GlobalReal builder](../tasks/builders/global_real.py)
- [ACES-like unit tests](../test/test_aces_like.py)
- [GlobalReal integration tests](../test/test_global_real.py)
