from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import predictor_runtime as predictor


MODEL_PATH = ROOT / 'model' / 'simple_mnist_m6_09585.h5'


def test_collect_input_names_uses_model_inputs_when_input_layer_is_absent() -> None:
    model = predictor.load_model_with_compat(str(MODEL_PATH))

    input_names = predictor._collect_input_names(model)

    assert 'input_1' in input_names


def test_get_inbound_layers_flattens_non_list_and_skips_none() -> None:
    node1 = SimpleNamespace(inbound_layers=SimpleNamespace(name="parent_a"))
    node2 = SimpleNamespace(inbound_layers=[None, SimpleNamespace(name="parent_b")])
    node3 = SimpleNamespace(inbound_layers=None)
    layer = SimpleNamespace(_inbound_nodes=[node1, node2, node3])

    inbound = predictor._get_inbound_layers(layer)

    assert [item.name for item in inbound] == ["parent_a", "parent_b"]


def test_collect_input_names_deduplicates_layer_and_input_names() -> None:
    InputLayer = type("InputLayer", (), {})
    layer_a = InputLayer()
    layer_a.name = "input_a"
    layer_b = InputLayer()
    layer_b.name = "input_b"
    model = SimpleNamespace(
        inputs=[SimpleNamespace(name="input_a:0"), SimpleNamespace(name="input_a:0")],
        layers=[layer_a, layer_b],
        input_names=["input_b", "input_c"],
    )

    input_names = predictor._collect_input_names(model)

    assert input_names == ["input_a", "input_b", "input_c"]


def test_collect_layers_and_inbound_skips_excluded_layers_and_resolves_grandparents() -> None:
    InputLayer = type("InputLayer", (), {})
    DenseLayer = type("Dense", (), {})
    DropoutLayer = type("Dropout", (), {})
    EmbeddingLayer = type("Embedding", (), {})

    input_layer = InputLayer()
    input_layer.name = "input_1"
    input_layer._inbound_nodes = []

    hidden_parent = DenseLayer()
    hidden_parent.name = "hidden_parent"
    hidden_parent._inbound_nodes = [SimpleNamespace(inbound_layers=input_layer)]

    excluded_parent = DropoutLayer()
    excluded_parent.name = "drop_parent"
    excluded_parent._inbound_nodes = [SimpleNamespace(inbound_layers=hidden_parent)]

    child = DenseLayer()
    child.name = "child"
    child._inbound_nodes = [
        SimpleNamespace(inbound_layers=excluded_parent),
        SimpleNamespace(inbound_layers=[hidden_parent, input_layer]),
    ]

    excluded_embedding = EmbeddingLayer()
    excluded_embedding.name = "embed"
    excluded_embedding._inbound_nodes = []

    model = SimpleNamespace(layers=[input_layer, hidden_parent, excluded_parent, child, excluded_embedding])

    layers, inbound_map = predictor._collect_layers_and_inbound(model)

    assert [layer.name for layer in layers] == ["hidden_parent", "child"]
    assert inbound_map == {"hidden_parent": ["input_1"], "child": ["hidden_parent", "input_1"]}


def test_init_model_bootstraps_real_mnist_model_without_missing_input_key() -> None:
    predictor.myModel = None
    predictor.loaded_model_path = None

    predictor.init_model(str(MODEL_PATH))

    assert predictor.myModel is not None
    assert predictor.myModel.keras_to_cache_key['input_1'] == 'layer_input'


def test_init_model_returns_early_when_same_model_already_loaded(monkeypatch) -> None:
    predictor.myModel = object()
    predictor.loaded_model_path = str(MODEL_PATH)
    monkeypatch.setattr(
        predictor,
        "load_model_with_compat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected reload")),
    )

    predictor.init_model(str(MODEL_PATH))


def test_predict_raises_when_model_is_not_initialized() -> None:
    predictor.myModel = None

    with pytest.raises(RuntimeError, match="Model not initialized"):
        predictor.predict(v_0_0=1.0)


def test_predict_binary_scalar_output_thresholds_to_class() -> None:
    predictor.myModel = SimpleNamespace(
        input_shape=(1, 2),
        forward=lambda tensor_input: [0.8],
    )

    result = predictor.predict(v_0_0=0.1, v_0_1=0.2)

    assert result == 1


def test_predict_binary_nested_output_thresholds_to_class() -> None:
    predictor.myModel = SimpleNamespace(
        input_shape=(1, 1, 2),
        forward=lambda tensor_input: [[0.2]],
    )

    result = predictor.predict(v_0_0_0=0.1, v_0_0_1=0.2)

    assert result == 0


def test_predict_multiclass_returns_argmax() -> None:
    predictor.myModel = SimpleNamespace(
        input_shape=(1, 1, 3),
        forward=lambda tensor_input: [0.1, 0.9, 0.3],
    )

    result = predictor.predict(v_0_0_0=1.0, v_0_0_1=2.0, v_0_0_2=3.0)

    assert result == 1


def test_predict_builds_2d_tensor_input_keys_correctly() -> None:
    captured = {}

    def fake_forward(tensor_input):
        captured["input"] = tensor_input
        return [1.0]

    predictor.myModel = SimpleNamespace(
        input_shape=(2, 2),
        forward=fake_forward,
    )

    predictor.predict(v_0_0=1.0, v_0_1=2.0, v_1_0=3.0, v_1_1=4.0)

    assert captured["input"] == [[1.0, 2.0], [3.0, 4.0]]


def test_predict_builds_3d_tensor_input_keys_correctly() -> None:
    captured = {}

    def fake_forward(tensor_input):
        captured["input"] = tensor_input
        return [1.0]

    predictor.myModel = SimpleNamespace(
        input_shape=(1, 2, 2),
        forward=fake_forward,
    )

    predictor.predict(v_0_0_0=1.0, v_0_0_1=2.0, v_0_1_0=3.0, v_0_1_1=4.0)

    assert captured["input"] == [[[1.0, 2.0], [3.0, 4.0]]]


def test_predict_builds_4d_tensor_input_keys_correctly() -> None:
    captured = {}

    def fake_forward(tensor_input):
        captured["input"] = tensor_input
        return [1.0]

    predictor.myModel = SimpleNamespace(
        input_shape=(1, 1, 2, 2),
        forward=fake_forward,
    )

    predictor.predict(
        v_0_0_0_0=1.0,
        v_0_0_0_1=2.0,
        v_0_0_1_0=3.0,
        v_0_0_1_1=4.0,
    )

    assert captured["input"] == [[[[1.0, 2.0], [3.0, 4.0]]]]


def test_init_model_clears_session_when_switching_model_path(monkeypatch, tmp_path: Path) -> None:
    calls = []
    other_model = tmp_path / "other_model.h5"
    other_model.write_bytes(b"fake")
    real_loader = predictor.load_model_with_compat

    predictor.myModel = object()
    predictor.loaded_model_path = "existing-model.h5"

    monkeypatch.setattr(predictor.keras.backend, "clear_session", lambda: calls.append("clear"))
    monkeypatch.setattr(
        predictor,
        "load_model_with_compat",
        lambda *_args, **_kwargs: real_loader(str(MODEL_PATH)),
    )

    predictor.init_model(str(other_model))

    assert calls == ["clear"]
