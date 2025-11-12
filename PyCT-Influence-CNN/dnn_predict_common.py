import keras
from dnnct.myDNN import NNModel
import numpy as np
import itertools

from libct.position import register_layer_number_mapping, to_Keras_layer_number


def _get_inbound_layers(layer):  # 取得某層的上一層神經層的資訊
    inbound = []
    for node in getattr(layer, "_inbound_nodes", []):  # keras把上一層的所有資訊存在_inbound_nodes
        inbound_layers = getattr(
            node, "inbound_layers", None)  # 取得上一層layer name
        if inbound_layers is None:
            continue
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        # 保證神經層依照資料流動順序維護神經層關係
        inbound.extend(l for l in inbound_layers if l is not None)
    return inbound

# 拓撲排序的核心是 indegree ＝ 0 的節點：只要前驅都處理完，就能進 queue。
# 如果同時有好幾個候選要進 queue，layer_index 只是拿來決定先拿誰，目的是保持 deterministic。


def _collect_execution_order(model):  # 取得模型中所有神經層的執行順序(拓樸排序)
    all_layers = list(model.layers)
    layer_index = {layer: idx for idx, layer in enumerate(all_layers)}
    reachable = set()
    inbound_map = {}

    def mark_reachable(layer):
        if layer is None or layer in reachable:
            return
        reachable.add(layer)
        for parent in _get_inbound_layers(layer):
            mark_reachable(parent)

    for output_tensor in model.outputs:
        history = getattr(output_tensor, "_keras_history", None)
        if history:
            mark_reachable(history[0])

    if not reachable:
        reachable = set(all_layers)

    graph = {layer: set() for layer in reachable}
    indegree = {layer: 0 for layer in reachable}
    for layer in reachable:
        for parent in _get_inbound_layers(layer):
            if parent in reachable:
                graph[parent].add(layer)
                indegree[layer] += 1

    # kahn's algorithm
    # indegree代表依賴項目 ＃ 每次從DAG中移除indegree為0的節點
    queue = [layer for layer, deg in indegree.items() if deg == 0]
    queue.sort(key=lambda l: layer_index.get(l, 0))

    execution_order = []
    excluded = {"Dropout", "InputLayer"}
    seen = set()

    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        if type(current).__name__ not in excluded:
            execution_order.append(current)
            inbound_map[current.name] = [
                parent.name for parent in _get_inbound_layers(current)]
        for nxt in graph.get(current, ()):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
        queue.sort(key=lambda l: layer_index.get(l, 0))

    return execution_order, inbound_map


def _format_shape(shape):
    if shape is None:
        return "None"
    return str(shape)


# 純粹對照keras 模型定義層 與 執行的神經層執行順序
def _describe_model_graph(model, execution_order):
    print("=== Keras model.layers order ===")
    for idx, layer in enumerate(model.layers):
        inbound = [parent.name for parent in _get_inbound_layers(layer)]
        print(
            f"[{idx:03d}] {layer.name:<30} {layer.__class__.__name__:<28} "
            f"output={_format_shape(getattr(layer, 'output_shape', None))} "
            f"inbound={inbound}"
        )

    print("=== Collected execution order ===")
    for idx, layer in enumerate(execution_order):
        inbound = [parent.name for parent in _get_inbound_layers(layer)]
        print(
            f"[{idx:03d}] {layer.name:<30} {layer.__class__.__name__:<28} "
            f"output={_format_shape(getattr(layer, 'output_shape', None))} "
            f"inbound={inbound}"
        )


myModel = None


def init_model(model_path):
    global myModel
    model = keras.models.load_model(model_path)
    model.summary()
    layers, inbound_map = _collect_execution_order(model)
    _describe_model_graph(model, layers)
    if not layers:
        layers = [l for l in model.layers if type(
            l).__name__ not in ['Dropout']]
    myModel = NNModel()
    input_layer_names = [
        layer.name for layer in model.layers if type(layer).__name__ == "InputLayer"]
    myModel.register_input_names(input_layer_names)

    # 1: is because 1st dim of input shape of Keras model is batch size (None)
    myModel.input_shape = model.input_shape[1:]
    myLayerCount = 0
    for i, layer in enumerate(layers):
        inbound_names = inbound_map.get(layer.name, [])
        numberOfMyLayers = myModel.addLayer(
            layer, inbound_names=inbound_names)
        # maintain the mapping of layers between Keras model and my model
        for j in range(numberOfMyLayers):
            register_layer_number_mapping(i, myLayerCount)
            myLayerCount += 1
    model.summary()
    print('Number of layers in my model:', len(myModel.layers))
    print('Number of layers in original Keras model:', len(layers))
    print('Correspondence between layers in Keras model and my model:')
    for myLayerNumber in range(myLayerCount):
        print('My: ', myLayerNumber, 'Keras: ',
              to_Keras_layer_number(myLayerNumber))
    for myLayer in myModel.layers:
        print(type(myLayer))


myInput = None


def init_input(input):
    global myInput
    myInput = input


def predict_2(**weights):
    # TODO: make a new model with the weights
    # TODO: run the globally defined input through the model
    pass


def predict(**data):
    # cast the flattened input dict 'data' back to the original shape and save to X
    input_shape = myModel.input_shape

    iter_args = (range(dim) for dim in input_shape)
    X = np.zeros(input_shape).tolist()
    data_name_prefix = "v_"
    for i in itertools.product(*iter_args):
        if len(i) == 2:
            X[i[0]][i[1]] = data[f"{data_name_prefix}{i[0]}_{i[1]}"]
        elif len(i) == 3:
            X[i[0]][i[1]][i[2]
                          ] = data[f"{data_name_prefix}{i[0]}_{i[1]}_{i[2]}"]
        elif len(i) == 4:
            X[i[0]][i[1]][i[2]][i[3]
                                ] = data[f"{data_name_prefix}{i[0]}_{i[1]}_{i[2]}_{i[3]}"]
    # calculate the output of the model

    out_val = myModel.forward(X)

    # 用一顆神經元做二分類
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
