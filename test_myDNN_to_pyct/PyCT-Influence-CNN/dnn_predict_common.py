import os
import re
import json
import keras
import h5py
import numpy as np
import itertools
from dnnct.myDNN import NNModel

from libct.position import register_layer_number_mapping, to_Keras_layer_number

# keras_resnet 版本的 BN 帶有額外參數 freeze，而原生 Keras BN 沒有。如果不提供這個 custom_objects，載入 keras_resnet 存的 h5 很可能會因為 config 包含 freeze 參數而報「Unknown layer / unexpected argument」的錯。
# 所以程式作者用 compat 類包一層，以確保能順利還原 keras_resnet 的模型。
# 結論：名字被改成 ResNetBatchNormCompat 是load_keras_model() 先呼叫 _get_resnet_custom_objects()當中的邏輯，並不是 h5 內建


# 總之有這個客製層所以後續必須這樣修改: 
# 錯誤點在這行：

# NameError: name 'libct' is not defined  (layer "bn_conv1", BatchNormalization)
# 原因：keras_resnet.layers.BatchNormalization 在 TF Autograph 轉成圖時會引用全域變數 libct，但該模組的 global 沒這個符號，所以第一次 forward 就噴錯。需要手動把 libct 塞回去。
import libct
try:
    from keras_resnet.layers import BatchNormalization as kr_bn
    kr_bn.libct = libct
    # 確認libct是否生效
    print(getattr(kr_bn, "libct", None))
    print("#### 有成功執行到這邊 #####")
except Exception:
    pass

# 全域變數，用於儲存延遲載入所需的上下文
myModel = None
_PENDING_INIT_CONTEXT = None  # 用於儲存 init_model 傳入的參數，等待資料來臨

def _get_inbound_layers(layer):
    inbound = []
    for node in getattr(layer, "_inbound_nodes", []):
        inbound_layers = getattr(node, "inbound_layers", None)
        if inbound_layers is None:
            continue
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        inbound.extend(l for l in inbound_layers if l is not None)
    return inbound


def _collect_layers_and_inbound(model):
    layers = [l for l in model.layers if type(l).__name__ not in [
        'Dropout', 'InputLayer']]
    inbound_map = {}
    for layer in layers:
        inbound_map[layer.name] = [
            parent.name for parent in _get_inbound_layers(layer)]
    return layers, inbound_map


def _format_shape(shape):
    if shape is None:
        return "None"
    return str(shape)


def _describe_model_layers(model, layers):
    print("=== Keras model.layers order ===")
    for idx, layer in enumerate(model.layers):
        inbound = [parent.name for parent in _get_inbound_layers(layer)]
        print(
            f"[{idx:03d}] {layer.name:<30} {layer.__class__.__name__:<28} "
            f"output={_format_shape(getattr(layer, 'output_shape', None))} "
            f"inbound={inbound}"
        )


def _get_resnet_custom_objects():
    custom_objects = {}
    try:
        from keras_resnet.models import ResNet2D18
        from keras_resnet.layers import BatchNormalization as ResNetBatchNorm

        class ResNetBatchNormCompat(ResNetBatchNorm):
            def __init__(self, *args, freeze=False, **kwargs):
                super().__init__(*args, freeze=freeze, **kwargs)

            @classmethod
            def from_config(cls, config):
                config = dict(config)
                config.setdefault("freeze", False)
                return super(ResNetBatchNormCompat, cls).from_config(config)

        custom_objects["ResNet2D18"] = ResNet2D18
        custom_objects["BatchNormalization"] = ResNetBatchNormCompat
    except ModuleNotFoundError:
        pass
    return custom_objects


def _needs_resnet_fallback(exc):
    msg = str(exc)
    return ("Unknown layer: 'ResNet2D" in msg
            or "missing 1 required positional argument: 'inputs'" in msg)


def _read_model_config(model_path):
    try:
        with h5py.File(model_path, 'r') as f:
            if 'model_config' not in f.attrs:
                return None
            config = f.attrs['model_config']
            if isinstance(config, bytes):
                config = config.decode('utf-8')
            return json.loads(config)
    except Exception:
        return None


def _extract_specs_from_weights(model_path):
    specs = {}
    try:
        with h5py.File(model_path, 'r') as f:
            weights_group = f.get('model_weights')
            if weights_group is None:
                # 某些版本可能直接在 root
                if 'layer_names' in f.attrs:
                     weights_group = f
                else:
                     return specs

            input_channels = None
            dense_candidates = []

            def visitor(name, obj):
                nonlocal input_channels
                if not isinstance(obj, h5py.Dataset):
                    return
                # 修正：增加 W:0 和 weights:0 以相容不同後端
                if not (name.endswith('kernel:0') or name.endswith('W:0') or name.endswith('weights:0')):
                    return
                
                shape = obj.shape
                lname = name.lower()
                
                # 嘗試抓取第一層 Conv 的 channel (通常是 4D tensor)
                if len(shape) == 4 and input_channels is None:
                    # 假設格式 (H, W, In, Out) 或 (Out, In, H, W)
                    # 這邊簡單判斷：通常 In Channel 是第 3 個維度 (index 2)
                    input_channels = shape[2]
                
                elif len(shape) == 2:
                    hint = 0 if any(tag in lname for tag in (
                        'pred', 'logit', 'fc', 'dense', 'classifier')) else 1
                    dense_candidates.append((hint, int(shape[1])))

            weights_group.visititems(visitor)

            if input_channels is not None:
                # 這裡暫時無法得知長寬，只能得知 Channel
                specs['input_channel_only'] = int(input_channels)
            
            if dense_candidates:
                dense_candidates.sort(key=lambda tup: (tup[0], tup[1]))
                specs['num_classes'] = dense_candidates[0][1]

    except Exception:
        return specs

    return specs


def _extract_specs_from_h5(model_path):
    specs = {}
    config = _read_model_config(model_path)
    if config:
        class_name = config.get('class_name', '')
        matches = re.findall(r'(\d+)', class_name)
        if matches:
            specs['depth'] = int(matches[-1])

        conf = config.get('config', {})
        layers = conf.get('layers', [])
        for layer in layers:
            class_name = layer.get('class_name')
            layer_conf = layer.get('config', {})
            if specs.get('input_shape') is None and class_name == 'InputLayer':
                batch_shape = layer_conf.get('batch_input_shape')
                if batch_shape:
                    specs['input_shape'] = tuple(
                        None if dim is None else int(dim) for dim in batch_shape[1:])
            if class_name == 'Dense':
                units = layer_conf.get('units')
                if units is not None:
                    specs['num_classes'] = int(units)

    weight_specs = _extract_specs_from_weights(model_path)
    for key, value in weight_specs.items():
        specs.setdefault(key, value)

    return specs


def _infer_resnet_specs(model_path, input_shape_override=None, num_classes_override=None):
    specs = _extract_specs_from_h5(model_path)
    
    if input_shape_override is not None:
        specs["input_shape"] = input_shape_override
    if num_classes_override is not None:
        specs["num_classes"] = num_classes_override

    # 如果有抓到 channel 但沒抓到 input_shape，且沒 override，這裡會缺 input_shape
    # 這就是我們要在 predict 階段補救的地方

    if "depth" not in specs:
        basename = os.path.splitext(os.path.basename(model_path))[0]
        match = re.search(r"resnet(\d+)", basename)
        if match:
            specs["depth"] = int(match.group(1))
    
    # 檢查必要參數，若不足則返回 None
    if "input_shape" not in specs or "num_classes" not in specs or "depth" not in specs:
        return None
    return specs


def _build_resnet_model(depth, input_shape, num_classes):
    try:
        from keras_resnet import models as kr_models
    except ModuleNotFoundError as exc:
        raise ImportError(
            "載入 ResNet 權重需要 `keras-resnet` 套件，請先安裝 `pip install keras-resnet`。"
        ) from exc

    builder = None
    candidate_names = [f"ResNet2D{depth}", f"ResNet{depth}"]
    for name in candidate_names:
        builder = getattr(kr_models, name, None)
        if builder is not None:
            break

    if builder is None:
        raise ValueError(
            f"在 keras_resnet.models 中找不到對應深度 {depth} 的 ResNet builder")

    print(f"[Info] Rebuilding ResNet{depth} with shape {input_shape} and classes {num_classes}")
    inputs = keras.layers.Input(shape=input_shape)
    return builder(inputs=inputs, classes=num_classes, include_top=True, freeze_bn=False)


def load_keras_model(model_path, input_shape_override=None, num_classes_override=None, require_complete=True):
    """
    載入模型。
    require_complete: 若為 False，當無法推斷規格時不報錯，而是返回 None (用於延遲載入)。
    """
    custom_objects = _get_resnet_custom_objects()
    try:
        return keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False)
    except (ValueError, TypeError) as exc:
        if not _needs_resnet_fallback(exc):
            raise
        
        basename = os.path.splitext(os.path.basename(model_path))[0]
        specs = _infer_resnet_specs(
            model_path, input_shape_override=input_shape_override,
            num_classes_override=num_classes_override
        )
        
        if specs is None:
            if not require_complete:
                return None # 允許延遲載入
            raise ValueError(
                f"無法解析 {basename} 的輸入形狀或分類數，且未提供樣本資料。"
            ) from exc
            
        model = _build_resnet_model(
            specs["depth"], specs["input_shape"], specs["num_classes"])
        
        try:
            model.load_weights(model_path)
        except ValueError as w_exc:
             # 如果 dataset 的形狀跟權重真的不合，這裡會報錯，這是正確的行為
             raise ValueError(f"模型權重載入失敗，請確認資料維度是否與模型相符: {w_exc}")
             
        return model


def _setup_nn_model(model, verbose=True):
    """將 Keras 模型轉換為 NNModel 的邏輯封裝"""
    global myModel
    if verbose:
        model.summary()
        
    layers, inbound_map = _collect_layers_and_inbound(model)
    _describe_model_layers(model, layers)
    
    myModel = NNModel()
    input_layer_names = [
        layer.name for layer in model.layers if type(layer).__name__ == "InputLayer"]
    myModel.register_input_names(input_layer_names)

    # 設定 input_shape (移除 batch 維度)
    if model.input_shape and len(model.input_shape) > 0:
         myModel.input_shape = model.input_shape[1:]
    
    myLayerCount = 0
    for i, layer in enumerate(layers):
        inbound_names = inbound_map.get(layer.name, [])
        numberOfMyLayers = myModel.addLayer(
            layer, inbound_names=inbound_names)
        for j in range(numberOfMyLayers):
            register_layer_number_mapping(i, myLayerCount)
            myLayerCount += 1
            
    if verbose:
        print('Number of layers in my model:', len(myModel.layers))
        print('Number of layers in original Keras model:', len(layers))


def init_model(model_path, verbose=True, input_shape_override=None, num_classes_override=None):
    global myModel, _PENDING_INIT_CONTEXT
    
    # 嘗試載入模型，如果因為缺規格失敗，則允許返回 None
    model = load_keras_model(model_path,
                             input_shape_override=input_shape_override,
                             num_classes_override=num_classes_override,
                             require_complete=False) # 關鍵：允許不完整
    
    if model is None:
        print(f"[Warning] init_model: 無法從 {os.path.basename(model_path)} 獲得完整規格。")
        print("          將推遲至收到第一筆資料時，根據資料形狀進行模型初始化。")
        
        # 儲存參數，等待 predict 時使用
        _PENDING_INIT_CONTEXT = {
            'model_path': model_path,
            'verbose': verbose,
            'input_shape_override': input_shape_override,
            'num_classes_override': num_classes_override
        }
        myModel = None
        return

    _setup_nn_model(model, verbose)


def _infer_input_shape_from_data(data):
    """
    從資料字典 (e.g. {'v_0_0_0': 0.5, ...}) 推斷輸入形狀。
    回傳格式: (height, width, channel)
    """
    max_indices = {} # dim_index -> max_value
    
    for key in data.keys():
        if not key.startswith("v_"):
            continue
        # 解析 key: v_0_15_3 -> indices [0, 15, 3]
        try:
            parts = key[2:].split('_')
            indices = [int(p) for p in parts]
            
            for dim, idx in enumerate(indices):
                if dim not in max_indices or idx > max_indices[dim]:
                    max_indices[dim] = idx
        except ValueError:
            continue
            
    if not max_indices:
        return None
        
    # Shape = max_index + 1
    # 確保維度連續，例如 keys 是 3 維，則回傳 (d0, d1, d2)
    sorted_dims = sorted(max_indices.keys())
    shape = tuple(max_indices[d] + 1 for d in sorted_dims)
    return shape


def _ensure_model_loaded(data):
    """確保模型已載入，若尚未載入則利用 data 推斷形狀並載入"""
    global myModel, _PENDING_INIT_CONTEXT
    
    if myModel is not None:
        return

    if _PENDING_INIT_CONTEXT is None:
        raise RuntimeError("模型尚未初始化，且無待處理的初始化上下文 (Init context missing)。")

    print("[Info] 偵測到模型尚未初始化，正在根據輸入資料推斷形狀...")
    
    # 1. 從資料推斷形狀
    inferred_shape = _infer_input_shape_from_data(data)
    if inferred_shape is None:
        raise ValueError("無法從輸入資料推斷形狀，請檢查資料格式是否為 'v_x_y_z'。")
    
    print(f"[Info] 推斷輸入形狀為: {inferred_shape}")

    # 2. 取出原本的參數並加上推斷出的 shape
    ctx = _PENDING_INIT_CONTEXT
    model = load_keras_model(
        ctx['model_path'],
        input_shape_override=inferred_shape, # 使用推斷出的形狀
        num_classes_override=ctx['num_classes_override'],
        require_complete=True # 這次必須成功
    )
    
    # 3. 完成初始化
    _setup_nn_model(model, verbose=ctx['verbose'])
    
    # 4. 清除 Context (避免重複初始化)
    _PENDING_INIT_CONTEXT = None


def _forward_from_flat_data(data):
    # 新增：確保模型已載入
    _ensure_model_loaded(data)
    
    input_shape = myModel.input_shape
    iter_args = (range(dim) for dim in input_shape)
    
    # 注意：這裡假設 input_shape 與 data keys 吻合
    # 如果 data 少了某些 pixel (稀疏)，這裡會填 0
    X = np.zeros(input_shape).tolist()
    
    data_name_prefix = "v_"
    
    # 根據維度動態填值
    for i in itertools.product(*iter_args):
        # 組合 key 字串
        key_suffix = "_".join(map(str, i))
        key = f"{data_name_prefix}{key_suffix}"
        
        if key in data:
             val = data[key]
             # 依照維度深度賦值
             if len(i) == 2:
                 X[i[0]][i[1]] = val
             elif len(i) == 3:
                 X[i[0]][i[1]][i[2]] = val
             elif len(i) == 4:
                 X[i[0]][i[1]][i[2]][i[3]] = val
    
    return myModel.forward(X)


def predict_logits(**data):
    return _forward_from_flat_data(data)


def predict(**data):
    out_val = _forward_from_flat_data(data)

    if len(out_val) == 1:
        if out_val[0] > 0.5:
            ret_class = 1
        else:
            ret_class = 0
    else:
        max_val, ret_class = out_val[0], 0
        for i, cl_val in enumerate(out_val):
            if cl_val > max_val:
                max_val, ret_class = cl_val, i

    print("[DEBUG]predicted class:", ret_class)
    return ret_class