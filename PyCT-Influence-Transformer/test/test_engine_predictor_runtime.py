from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import math
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import predictor_runtime as predictor
from libct.constraint import Constraint
from libct.path import PathToConstraint
from libct.position import register_current_indices, register_current_layer_number
from libct.utils import ConcolicObject


MODEL_PATH = ROOT / 'model' / 'simple_mnist_m6_09585.h5'


class _BranchEngine:
    symbolic_enabled = True

    def __init__(self) -> None:
        self.path = PathToConstraint()
        self.pushed = []

    def push_constraint(self, constraint, position) -> None:
        self.pushed.append((constraint, position))


def _reset_predictor_state() -> None:
    predictor.myModel = None
    predictor.loaded_model_path = None
    predictor.loaded_model_key = None
    predictor.searchModel = None
    predictor.search_model_key = None
    predictor.referenceModel = None
    predictor.reference_model_path = None
    predictor._MODEL_CACHE.clear()
    predictor._KERAS_MODEL_CACHE.clear()


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
    _reset_predictor_state()

    predictor.init_model(str(MODEL_PATH))

    assert predictor.myModel is not None
    assert predictor.myModel.keras_to_cache_key['input_1'] == 'layer_input'
    assert predictor.loaded_model_key == (str(MODEL_PATH.resolve()), False, 0.75)
    assert predictor.referenceModel is not None


def test_init_model_returns_early_when_same_model_already_loaded(monkeypatch) -> None:
    _reset_predictor_state()
    cached_model = object()
    cached_reference = object()
    resolved_path = str(MODEL_PATH.resolve())
    predictor.myModel = cached_model
    predictor.loaded_model_path = resolved_path
    predictor.loaded_model_key = (resolved_path, False, 0.75)
    predictor._MODEL_CACHE[(resolved_path, False, 0.75)] = cached_model
    predictor._KERAS_MODEL_CACHE[resolved_path] = cached_reference
    monkeypatch.setattr(
        predictor,
        "load_model_with_compat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected reload")),
    )

    predictor.init_model(str(MODEL_PATH))


def test_predict_raises_when_model_is_not_initialized() -> None:
    _reset_predictor_state()

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


def test_predict_rejects_non_finite_validation_input() -> None:
    predictor.myModel = SimpleNamespace(
        input_shape=(1, 1, 1),
        forward=lambda tensor_input: [0.1, 0.9],
    )

    with pytest.raises(ValueError, match="Validation input contains"):
        predictor.predict(v_0_0_0=math.nan)


def test_predict_reference_rejects_non_finite_output_instead_of_class_zero() -> None:
    predictor.referenceModel = SimpleNamespace(
        input_shape=(None, 1, 1, 1),
        predict=lambda _batch, verbose=0: [[math.nan, math.nan, math.nan]],
    )

    with pytest.raises(ValueError, match="Keras reference model output contains"):
        predictor.predict_reference(v_0_0_0=0.5)


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


def test_init_model_distinguishes_cache_entries_by_ternary_config(monkeypatch) -> None:
    _reset_predictor_state()
    calls = []
    real_loader = predictor.load_model_with_compat

    def fake_loader(model_path):
        calls.append(model_path)
        model = real_loader(str(MODEL_PATH))
        model._name = Path(model_path).stem
        return model

    monkeypatch.setattr(predictor, "load_model_with_compat", fake_loader)

    predictor.init_model(str(MODEL_PATH), ternary_simplification=False, ternary_threshold_scale=0.75)
    first_model = predictor.myModel
    predictor.init_model(str(MODEL_PATH), ternary_simplification=True, ternary_threshold_scale=0.75)
    second_model = predictor.myModel
    predictor.init_model(str(MODEL_PATH), ternary_simplification=True, ternary_threshold_scale=1.5)
    third_model = predictor.myModel

    assert len(calls) == 1
    assert first_model is not second_model
    assert second_model is not third_model
    assert len(predictor._MODEL_CACHE) == 3
    assert len(predictor._KERAS_MODEL_CACHE) == 1


def test_init_model_reuses_cache_only_when_model_and_ternary_config_match(monkeypatch) -> None:
    _reset_predictor_state()
    load_calls = []
    real_loader = predictor.load_model_with_compat

    monkeypatch.setattr(
        predictor,
        "load_model_with_compat",
        lambda model_path: load_calls.append(model_path) or real_loader(str(MODEL_PATH)),
    )

    predictor.init_model(str(MODEL_PATH), ternary_simplification=True, ternary_threshold_scale=0.75)
    first_model = predictor.myModel
    predictor.init_model(str(MODEL_PATH), ternary_simplification=True, ternary_threshold_scale=0.75)
    second_model = predictor.myModel

    assert len(load_calls) == 1
    assert first_model is second_model


def test_init_model_assigns_search_and_reference_models() -> None:
    _reset_predictor_state()

    predictor.init_model(str(MODEL_PATH), ternary_simplification=True, ternary_threshold_scale=1.5, role="search")

    assert predictor.searchModel is not None
    assert predictor.referenceModel is not None
    assert predictor.search_model_key == (str(MODEL_PATH.resolve()), True, 1.5)
    assert predictor.reference_model_path == str(MODEL_PATH.resolve())


def test_predict_search_and_reference_use_role_specific_models() -> None:
    reference_calls = []
    predictor.searchModel = SimpleNamespace(
        input_shape=(1, 2),
        forward=lambda tensor_input: [0.1, 0.8],
    )
    predictor.referenceModel = SimpleNamespace(
        input_shape=(None, 1, 2),
        predict=lambda batch, verbose=0: reference_calls.append((batch.copy(), verbose))
        or np.array([[0.9, 0.2]], dtype=np.float32),
    )

    assert predictor.predict_search(v_0_0=0.1, v_0_1=0.2) == 1
    assert predictor.predict_reference(v_0_0=0.1, v_0_1=0.2) == 0
    assert reference_calls[0][0].shape == (1, 1, 2)
    assert reference_calls[0][1] == 0


def test_predict_search_runs_forward_through_final_binary_class_branch(monkeypatch) -> None:
    Constraint.global_constraints.clear()
    PathToConstraint.root_constraint = None
    register_current_layer_number(7)
    register_current_indices((0,))
    engine = _BranchEngine()
    forwarded = []

    def forward(tensor_input):
        forwarded.append(tensor_input)
        return [tensor_input[0][0]]

    monkeypatch.setattr(
        predictor,
        "searchModel",
        SimpleNamespace(input_shape=(1, 1), forward=forward),
    )
    symbolic_input = ConcolicObject(0.8, "x_VAR", engine)

    assert predictor.predict_search(v_0_0=symbolic_input) == 1
    assert len(forwarded) == 1
    assert forwarded[0][0][0] is symbolic_input
    assert [position for _, position in engine.pushed] == [(7, (0,))]


def test_predict_search_runs_forward_through_final_multiclass_branch(monkeypatch) -> None:
    Constraint.global_constraints.clear()
    PathToConstraint.root_constraint = None
    register_current_layer_number(8)
    register_current_indices((1,))
    engine = _BranchEngine()
    forwarded = []

    def forward(tensor_input):
        forwarded.append(tensor_input)
        return [0.1, tensor_input[0][0]]

    monkeypatch.setattr(
        predictor,
        "searchModel",
        SimpleNamespace(input_shape=(1, 1), forward=forward),
    )
    symbolic_input = ConcolicObject(0.9, "x_VAR", engine)

    assert predictor.predict_search(v_0_0=symbolic_input) == 1
    assert len(forwarded) == 1
    assert forwarded[0][0][0] is symbolic_input
    assert [position for _, position in engine.pushed] == [(8, (1,))]


def test_predict_reference_binary_output_uses_threshold() -> None:
    predictor.referenceModel = SimpleNamespace(
        input_shape=(None, 1, 1),
        predict=lambda _batch, verbose=0: np.array([[0.8]], dtype=np.float32),
    )

    assert predictor.predict_reference(v_0_0=0.25) == 1


def test_predict_reference_rejects_non_finite_input() -> None:
    predictor.referenceModel = SimpleNamespace(
        input_shape=(None, 1, 1),
        predict=lambda _batch, verbose=0: np.array([[0.8]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Keras reference input contains"):
        predictor.predict_reference(v_0_0=math.inf)
