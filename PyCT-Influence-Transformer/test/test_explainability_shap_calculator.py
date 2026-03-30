from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import explainability.shap_calculator as shap_mod


class _FakeLayer:
    pass


class _FakeSequential:
    def __init__(self, layer_count: int = 3) -> None:
        self.layers = [_FakeLayer() for _ in range(layer_count)]


class _FakeFunctional:
    def __init__(self, layer_count: int = 3) -> None:
        self.layers = [_FakeLayer() for _ in range(layer_count)]


class _Shape:
    def __init__(self, dims) -> None:
        self._dims = tuple(dims)
        self.rank = len(self._dims)

    def __getitem__(self, index):
        return self._dims[index]


class _Tensor:
    def __init__(self, array) -> None:
        self.array = np.asarray(array, dtype=np.float32)
        self.shape = _Shape(self.array.shape)

    def __getitem__(self, index):
        result = self.array[index]
        if isinstance(result, np.ndarray):
            return _Tensor(result)
        return float(result)

    def __sub__(self, other):
        other_arr = other.array if isinstance(other, _Tensor) else other
        return _Tensor(self.array - other_arr)

    def __mul__(self, other):
        other_arr = other.array if isinstance(other, _Tensor) else other
        return _Tensor(self.array * other_arr)

    def numpy(self):
        return self.array


def _make_calculator(monkeypatch, tmp_path: Path, model) -> shap_mod.ShapValuesCalculator:
    monkeypatch.setattr(shap_mod, "Sequential", _FakeSequential)
    monkeypatch.setattr(shap_mod, "_load_model_with_compat", lambda *args, **kwargs: model)
    return shap_mod.ShapValuesCalculator(
        model_path=str(tmp_path / "demo_model.h5"),
        background_dataset=np.zeros((2, 2, 2), dtype=np.float32),
        input_data=np.zeros((1, 2, 2), dtype=np.float32),
        idx=3,
        output_root=str(tmp_path),
    )


def test_ensure_returns_in_memory_values_without_refresh(monkeypatch, tmp_path: Path) -> None:
    calculator = _make_calculator(monkeypatch, tmp_path, _FakeSequential())
    calculator._shap_values = {"0_0": 1.0}
    monkeypatch.setattr(calculator, "_compute_shap_values", lambda: (_ for _ in ()).throw(AssertionError("unexpected")))

    result = calculator.ensure()

    assert result == {"0_0": 1.0}


def test_ensure_loads_cache_when_present(monkeypatch, tmp_path: Path) -> None:
    calculator = _make_calculator(monkeypatch, tmp_path, _FakeSequential())
    calculator.cache_path.write_text(json.dumps({"values": {"0_0": 1.5}}), encoding="utf-8")

    result = calculator.ensure(assume_cached=True)

    assert result == {"0_0": 1.5}


def test_ensure_recomputes_when_cache_is_corrupt(monkeypatch, tmp_path: Path) -> None:
    calculator = _make_calculator(monkeypatch, tmp_path, _FakeSequential())
    calculator.cache_path.write_text("{bad json", encoding="utf-8")
    monkeypatch.setattr(calculator, "_compute_shap_values", lambda: {"0_1": 2.0})

    result = calculator.ensure()

    assert result == {"0_1": 2.0}
    assert json.loads(calculator.cache_path.read_text(encoding="utf-8")) == {"0_1": 2.0}


def test_save_cache_writes_meta_wrapper_when_meta_exists(monkeypatch, tmp_path: Path) -> None:
    calculator = _make_calculator(monkeypatch, tmp_path, _FakeSequential())
    calculator._cache_meta = {"background_seed": 7}

    calculator._save_cache({"0_0": 0.25})

    payload = json.loads(calculator.cache_path.read_text(encoding="utf-8"))
    assert payload["__meta__"] == {"background_seed": 7}
    assert payload["values"] == {"0_0": 0.25}


def test_read_background_meta_ignores_invalid_env_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PYCT_BG_PER_CLASS", "abc")
    monkeypatch.setenv("PYCT_BG_SEED", "def")

    calculator = _make_calculator(monkeypatch, tmp_path, _FakeSequential())

    assert calculator._cache_meta == {}


def test_compute_shap_values_functional_model_uses_input_level_plus_branch_influence(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    calculator = _make_calculator(monkeypatch, tmp_path, _FakeFunctional())
    monkeypatch.setattr(
        calculator,
        "_calculate_layer_shap_values",
        lambda shap_values, model, background_dataset, input_data, layer_number: calls.append(layer_number)
        or shap_values.update({"layer": float(layer_number)}),
    )
    monkeypatch.setattr(
        calculator,
        "_calculate_functional_layer_branch_influence",
        lambda shap_values: shap_values.update({"branch": 1.0}),
    )

    result = calculator._compute_shap_values()

    assert calculator._layerwise_enabled is False
    assert calls == [0]
    assert result["branch"] == 1.0


def test_compute_shap_values_sequential_model_walks_layers(monkeypatch, tmp_path: Path) -> None:
    calls = []
    calculator = _make_calculator(monkeypatch, tmp_path, _FakeSequential(layer_count=3))
    monkeypatch.setattr(
        calculator,
        "_calculate_layer_shap_values",
        lambda shap_values, model, background_dataset, input_data, layer_number: calls.append(layer_number),
    )
    monkeypatch.setattr(calculator, "apply_one_layer", lambda model, input_data: input_data)
    monkeypatch.setattr(calculator, "apply_one_layer_to_dataset", lambda model, dataset: dataset)
    monkeypatch.setattr(calculator, "without_first_layer", lambda model: model)

    calculator._compute_shap_values()

    assert calculator._layerwise_enabled is True
    assert calls == [0, 1, 2]


def test_calculate_layer_shap_values_gradient_records_reduced_values(monkeypatch, tmp_path: Path) -> None:
    calculator = _make_calculator(monkeypatch, tmp_path, _FakeSequential())
    shap_values = {}

    class _FakeGradientExplainer:
        def __init__(self, model, background):
            assert len(background) == 1
            assert np.array_equal(background[0], np.array([[0.0, 0.0]], dtype=np.float32))

        def shap_values(self, input_data):
            return np.array([[[1.0, 3.0]]], dtype=np.float32)

    monkeypatch.setattr(shap_mod.shap, "GradientExplainer", _FakeGradientExplainer)

    calculator._calculate_layer_shap_values(
        shap_values,
        model="model",
        background_dataset=np.array([[0.0, 0.0]], dtype=np.float32),
        input_data=np.array([[0.2, 0.4]], dtype=np.float32),
        layer_number=1,
    )

    assert shap_values == {"0_0": 1.0, "0_1": 3.0}


def test_calculate_layer_shap_values_kernel_flattens_multidim_input(monkeypatch, tmp_path: Path) -> None:
    calculator = _make_calculator(monkeypatch, tmp_path, _FakeSequential())
    calculator.explainer_type = "kernel"
    shap_values = {}

    class _FakeKernelExplainer:
        def __init__(self, model, background):
            assert model == "flat-model"
            assert np.array_equal(background, np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32))

        def shap_values(self, kernel_input):
            assert kernel_input.shape == (1, 4)
            return np.array([[10.0, 11.0, 12.0, 13.0]], dtype=np.float32)

    monkeypatch.setattr(shap_mod.shap, "KernelExplainer", _FakeKernelExplainer)
    monkeypatch.setattr(
        calculator,
        "_flatten_everything",
        lambda model, input_data, background_dataset: (
            "flat-model",
            np.array([9.0, 8.0, 7.0, 6.0], dtype=np.float32),
            np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32),
        ),
    )

    calculator._calculate_layer_shap_values(
        shap_values,
        model="model",
        background_dataset=np.zeros((1, 2, 2, 1), dtype=np.float32),
        input_data=np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32),
        layer_number=1,
    )

    assert shap_values == {
        "0_0_0_0": 10.0,
        "0_0_1_0": 11.0,
        "0_1_0_0": 12.0,
        "0_1_1_0": 13.0,
    }


def test_reduce_gradient_shap_values_averages_list_and_batch_axes() -> None:
    gradients = [
        np.array([[[1.0, 3.0], [5.0, 7.0]]], dtype=np.float32),
        np.array([[[2.0, 4.0], [6.0, 8.0]]], dtype=np.float32),
    ]

    reduced = shap_mod.ShapValuesCalculator._reduce_gradient_shap_values(
        gradients,
        input_data=np.zeros((1, 2, 2), dtype=np.float32),
    )

    assert np.array_equal(reduced, np.array([[1.5, 3.5], [5.5, 7.5]], dtype=np.float32))


def test_calculate_functional_layer_branch_influence_records_per_neuron_values(monkeypatch, tmp_path: Path) -> None:
    layer = SimpleNamespace(output="layer-output")
    model = SimpleNamespace(inputs="inputs", output="logits", layers=[layer])
    calculator = _make_calculator(monkeypatch, tmp_path, model)
    calculator._tracked_layers = [layer]
    calculator._model = model
    calculator.input = np.array([[5.0, 7.0]], dtype=np.float32)
    calculator.background_dataset = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    shap_values = {}

    class _FakeFeatureModel:
        def __call__(self, tensor, training=False):
            if tensor.array.shape[0] == 2:
                return [_Tensor([[1.0, 2.0], [3.0, 4.0]]), _Tensor([[0.1, 0.9]])]
            return [_Tensor([[5.0, 7.0]]), _Tensor([[0.2, 0.8]])]

    class _FakeGradientTape:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def watch(self, act):
            return None

        def gradient(self, target, acts):
            return [_Tensor([[0.5, 2.0]])]

    monkeypatch.setattr(shap_mod, "Model", lambda inputs, outputs: _FakeFeatureModel())
    monkeypatch.setattr(shap_mod.tf, "convert_to_tensor", lambda array, dtype=None: _Tensor(array))
    monkeypatch.setattr(shap_mod.tf, "GradientTape", _FakeGradientTape)
    monkeypatch.setattr(shap_mod.tf, "reduce_mean", lambda tensor, axis=None: _Tensor(np.mean(tensor.array, axis=axis)))
    monkeypatch.setattr(shap_mod.tf, "argmax", lambda tensor: int(np.argmax(tensor.array)))
    monkeypatch.setattr(shap_mod.tf, "cast", lambda value, dtype=None: value)
    monkeypatch.setattr(
        shap_mod.tf,
        "gather",
        lambda tensor, index, axis=0: _Tensor(np.take(tensor.array, index, axis=axis)),
    )
    monkeypatch.setattr(shap_mod.tf, "int32", "int32")
    monkeypatch.setattr(shap_mod.tf, "float32", "float32")

    calculator._calculate_functional_layer_branch_influence(shap_values)

    assert shap_values == {"0_0": 1.5, "0_1": 8.0}


def test_without_first_layer_and_apply_helpers_cover_sequential_paths(monkeypatch) -> None:
    added = []

    class _SeqFactory:
        def __init__(self, layers=None) -> None:
            self.layers = list(layers or [])
            self.built_with = None

        def add(self, layer) -> None:
            self.layers.append(layer)
            added.append(layer)

        def build(self, input_shape=None) -> None:
            self.built_with = input_shape

        def predict(self, data):
            return f"pred:{data}"

    layer0 = SimpleNamespace(input_shape=(None, 2), input="in0", output="out0")
    layer1 = SimpleNamespace(input_shape=(None, 3), input="in1", output="out1")
    layer2 = SimpleNamespace(input_shape=(None, 4), input="in2", output="out2")
    model = _SeqFactory([layer0, layer1, layer2])
    monkeypatch.setattr(shap_mod, "Sequential", _SeqFactory)

    tail = shap_mod.ShapValuesCalculator.without_first_layer(model)
    first_only = shap_mod.ShapValuesCalculator.get_model_with_only_first_layer(model)
    dataset_pred = shap_mod.ShapValuesCalculator.apply_one_layer_to_dataset(model, "dataset")
    input_pred = shap_mod.ShapValuesCalculator.apply_one_layer(model, "input")

    assert [layer.input_shape for layer in tail.layers] == [(None, 3), (None, 4)]
    assert first_only.layers == [layer0]
    assert dataset_pred == "pred:dataset"
    assert input_pred == "pred:input"


def test_cache_path_uses_output_root_and_model_stem(monkeypatch, tmp_path: Path) -> None:
    calculator = _make_calculator(monkeypatch, tmp_path, _FakeSequential())

    assert calculator.cache_path == tmp_path / "demo_model" / "shap_value_3.json"


def test_infer_layer_count_from_cached_values_uses_max_prefix() -> None:
    assert shap_mod._infer_layer_count_from_cached_values({"0_0": 1.0, "2_1_3": 2.0}) == 3
    assert shap_mod._infer_layer_count_from_cached_values({"foo": 1.0}) == 1


def test_load_cached_shap_values_supports_plain_and_wrapped_payloads(tmp_path: Path) -> None:
    plain_path = tmp_path / "plain.json"
    wrapped_path = tmp_path / "wrapped.json"
    plain_path.write_text(json.dumps({"0_0": 1, "1_0": 2.5}), encoding="utf-8")
    wrapped_path.write_text(json.dumps({"values": {"0_0": 1, "1_0": 3}}), encoding="utf-8")

    assert shap_mod._load_cached_shap_values(plain_path) == {"0_0": 1.0, "1_0": 2.5}
    assert shap_mod._load_cached_shap_values(wrapped_path) == {"0_0": 1.0, "1_0": 3.0}


def test_shap_values_comparator_fast_path_uses_cached_values(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "demo_model"
    cache_dir.mkdir(parents=True)
    (cache_dir / "shap_value_4.json").write_text(json.dumps({"values": {"0_1": 1.25, "1_0_0": 2.0}}), encoding="utf-8")

    class _UnexpectedCalculator:
        get_position_key = staticmethod(shap_mod.ShapValuesCalculator.get_position_key)

        def __init__(self, **kwargs) -> None:
            raise AssertionError("unexpected calculator construction")

    monkeypatch.setattr(shap_mod, "ShapValuesCalculator", _UnexpectedCalculator)

    comparator = shap_mod.ShapValuesComparator(
        model_path=str(tmp_path / "demo_model.h5"),
        background_dataset=np.zeros((1, 1), dtype=np.float32),
        input=np.zeros((1, 1), dtype=np.float32),
        idx=4,
        shap_value_pre_calculated=True,
        output_root=str(tmp_path),
    )

    assert comparator.model is None
    assert comparator.layer_count == 2
    assert comparator.get_shap_influence(0, (1,)) == 1.25


def test_shap_values_comparator_falls_back_to_calculator_when_fast_path_fails(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "demo_model"
    cache_dir.mkdir(parents=True)
    (cache_dir / "shap_value_5.json").write_text("{bad json", encoding="utf-8")

    class _FakeCalculator:
        get_position_key = staticmethod(shap_mod.ShapValuesCalculator.get_position_key)

        def __init__(self, **kwargs) -> None:
            self.model = "model"
            self.layer_count = 3

        def ensure(self, assume_cached: bool, force_refresh: bool):
            assert assume_cached is True
            assert force_refresh is False
            return {"0_0": 0.5}

    monkeypatch.setattr(shap_mod, "ShapValuesCalculator", _FakeCalculator)

    comparator = shap_mod.ShapValuesComparator(
        model_path=str(tmp_path / "demo_model.h5"),
        background_dataset=np.zeros((1, 1), dtype=np.float32),
        input=np.zeros((1, 1), dtype=np.float32),
        idx=5,
        shap_value_pre_calculated=True,
        output_root=str(tmp_path),
    )

    assert comparator.model == "model"
    assert comparator.layer_count == 3
    assert comparator.get_shap_influence(0, (0,)) == 0.5


def test_shap_values_comparator_lookup_falls_back_to_alt_and_spatial_keys(monkeypatch, tmp_path: Path) -> None:
    class _FakeCalculator:
        get_position_key = staticmethod(shap_mod.ShapValuesCalculator.get_position_key)

        def __init__(self, **kwargs) -> None:
            self.model = "model"
            self.layer_count = 4

        def ensure(self, assume_cached: bool, force_refresh: bool):
            return {"0_2_3": 1.0, "1_2_9": 2.0, "2_5": 3.0}

    monkeypatch.setattr(shap_mod, "ShapValuesCalculator", _FakeCalculator)
    comparator = shap_mod.ShapValuesComparator(
        model_path=str(tmp_path / "demo_model.h5"),
        background_dataset=np.zeros((1, 1), dtype=np.float32),
        input=np.zeros((1, 1), dtype=np.float32),
        idx=6,
        shap_value_pre_calculated=False,
        output_root=str(tmp_path),
    )

    assert comparator.get_shap_influence(1, (2, 3)) == 1.0
    assert comparator.get_shap_influence(2, (2, 9, 1)) == 2.0
    assert comparator.get_shap_influence(3, (5, 7)) == float("-inf")
    assert comparator.get_shap_influence(0, [(2, 3), (9,)]) == 0.5


def test_shap_values_comparator_compare_and_pop_helpers(monkeypatch, tmp_path: Path) -> None:
    class _FakeCalculator:
        get_position_key = staticmethod(shap_mod.ShapValuesCalculator.get_position_key)

        def __init__(self, **kwargs) -> None:
            self.model = "model"
            self.layer_count = 3

        def ensure(self, assume_cached: bool, force_refresh: bool):
            return {"0_0": 0.1, "0_1": 0.9}

    monkeypatch.setattr(shap_mod, "ShapValuesCalculator", _FakeCalculator)
    comparator = shap_mod.ShapValuesComparator(
        model_path=str(tmp_path / "demo_model.h5"),
        background_dataset=np.zeros((1, 1), dtype=np.float32),
        input=np.zeros((1, 1), dtype=np.float32),
        idx=7,
        shap_value_pre_calculated=False,
        output_root=str(tmp_path),
    )
    low = ("c1", (0, (0,)))
    high = ("c2", (0, (1,)))
    positioned = [high, low]

    assert comparator.compare(high, low) == 0.8
    assert shap_mod.pop_first_constraint(positioned[:]) == "c2"
    assert shap_mod.pop_last_constraint(positioned[:]) == "c1"
    assert shap_mod.pop_the_most_important_constraint(positioned[:], comparator.compare) == "c2"


def test_flatten_unflatten_and_position_key_helpers(monkeypatch, tmp_path: Path) -> None:
    calculator = _make_calculator(monkeypatch, tmp_path, _FakeSequential())

    assert np.array_equal(calculator._flatten(np.array([[1, 2], [3, 4]])), np.array([1, 2, 3, 4]))
    assert calculator._unflatten_index(5, (2, 3)) == (1, 2)
    assert calculator.get_position_key(2, (1, 3)) == "2_1_3"
