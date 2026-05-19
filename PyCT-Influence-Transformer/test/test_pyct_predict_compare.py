from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyct.predict_compare as compare_mod


class _FakeKerasModel:
    def __init__(self, predictions, input_shape=(None, 2)) -> None:
        self.predictions = np.asarray(predictions, dtype=np.float32)
        self.input_shape = input_shape
        self.predict_calls = []

    def predict(self, inputs, verbose=0):
        self.predict_calls.append((np.asarray(inputs), verbose))
        return self.predictions[: len(inputs)]


class _FakePythonModel:
    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)
        self.forward_calls = []

    def forward(self, tensor):
        self.forward_calls.append(tensor)
        return self.outputs[len(self.forward_calls) - 1]


class _FakeLayerOutputModel:
    def __init__(self, outputs_by_layer) -> None:
        self.outputs_by_layer = outputs_by_layer

    def getLayOutput(self, idx):
        return self.outputs_by_layer[idx]


def test_parse_args_resolves_model_name_and_rejects_both_model_inputs() -> None:
    args = compare_mod.parse_args(["--model-name", "demo"])

    assert compare_mod.resolve_model_path(args) == Path("model") / "demo.h5"

    with pytest.raises(SystemExit):
        compare_mod.parse_args(["--model-name", "demo", "--model-path", "model/demo.h5"])


def test_resolve_log_file_defaults_to_timestamp_and_model_name(monkeypatch) -> None:
    monkeypatch.setattr(compare_mod, "current_timestamp", lambda: "20260519_143205")

    args = compare_mod.parse_args(["--model-name", "demo-model"])

    assert compare_mod.resolve_log_file(args) == (
        Path("predict_compare_log") / "20260519_143205_demo-model.log"
    )


def test_resolve_log_file_defaults_to_model_path_stem(monkeypatch) -> None:
    monkeypatch.setattr(compare_mod, "current_timestamp", lambda: "20260519_143205")

    args = compare_mod.parse_args(["--model-path", "/tmp/models/foo.h5"])

    assert compare_mod.resolve_log_file(args) == (
        Path("predict_compare_log") / "20260519_143205_foo.log"
    )


def test_resolve_log_file_uses_explicit_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(compare_mod, "current_timestamp", lambda: "20260519_143205")
    explicit = tmp_path / "custom.log"

    args = compare_mod.parse_args(["--model-name", "demo", "--log-file", str(explicit)])

    assert compare_mod.resolve_log_file(args) == explicit


def test_parse_args_rejects_non_positive_counts() -> None:
    with pytest.raises(SystemExit):
        compare_mod.parse_args(["--model-name", "demo", "--first-n", "0"])

    with pytest.raises(SystemExit):
        compare_mod.parse_args(["--model-name", "demo", "--samples", "0"])


def test_generate_random_inputs_uses_samples_seed_and_model_shape() -> None:
    model = SimpleNamespace(input_shape=(None, 2, 3, 1))

    first = compare_mod.generate_random_inputs(model, samples=4, seed=123)
    second = compare_mod.generate_random_inputs(model, samples=4, seed=123)

    assert first.shape == (4, 2, 3, 1)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)


def test_generate_random_inputs_rejects_multiple_or_dynamic_inputs() -> None:
    with pytest.raises(ValueError, match="multiple model inputs"):
        compare_mod.generate_random_inputs(
            SimpleNamespace(input_shape=[(None, 2), (None, 2)]),
            samples=1,
            seed=1,
        )

    with pytest.raises(ValueError, match="fully-defined input shape"):
        compare_mod.generate_random_inputs(
            SimpleNamespace(input_shape=(None, None, 2)),
            samples=1,
            seed=1,
        )


def test_load_inputs_routes_dataset_and_random_sample_counts(monkeypatch) -> None:
    calls = []
    dataset_inputs = np.asarray([[1.0, 2.0]], dtype=np.float32)
    random_inputs = np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    monkeypatch.setattr(
        compare_mod,
        "load_dataset_inputs",
        lambda dataset, first_n: calls.append(("dataset", dataset, first_n)) or dataset_inputs,
    )
    monkeypatch.setattr(
        compare_mod,
        "generate_random_inputs",
        lambda model, samples, seed: calls.append(("random", samples, seed)) or random_inputs,
    )

    dataset_args = compare_mod.parse_args(
        ["--model-name", "demo", "--dataset", "cifar10", "--first-n", "3"]
    )
    random_args = compare_mod.parse_args(
        [
            "--model-name",
            "demo",
            "--input-source",
            "random",
            "--first-n",
            "3",
            "--samples",
            "2",
            "--seed",
            "99",
        ]
    )

    assert compare_mod.load_inputs(dataset_args, object()) is dataset_inputs
    assert compare_mod.load_inputs(random_args, object()) is random_inputs
    assert calls == [("dataset", "cifar10", 3), ("random", 2, 99)]


def test_load_dataset_inputs_uses_dataset_adapter_x_test(monkeypatch) -> None:
    fake_package = ModuleType("datasets")
    fake_mnist = ModuleType("datasets.mnist")

    class FakeMnistDataset:
        def __init__(self) -> None:
            self.x_test = np.arange(12, dtype=np.float32).reshape(3, 2, 2)

    fake_mnist.MnistDataset = FakeMnistDataset
    monkeypatch.setitem(sys.modules, "datasets", fake_package)
    monkeypatch.setitem(sys.modules, "datasets.mnist", fake_mnist)

    inputs = compare_mod.load_dataset_inputs("mnist", 2)

    assert inputs.dtype == np.float32
    np.testing.assert_array_equal(inputs, np.arange(8, dtype=np.float32).reshape(2, 2, 2))


def test_compare_case_passes_only_when_outputs_close_and_classes_match() -> None:
    passing = compare_mod.compare_case(0, [0.1, 0.9], [0.100001, 0.900001])
    not_close = compare_mod.compare_case(1, [0.1, 0.9], [0.2, 0.8])
    class_mismatch = compare_mod.compare_case(2, [0.49], [0.51])
    close_class_mismatch = compare_mod.compare_case(
        3,
        [0.5, 0.500001],
        [0.500001, 0.5],
    )

    assert passing.passed is True
    assert passing.keras_class == 1
    assert passing.py_class == 1
    assert not_close.close is False
    assert not_close.passed is False
    assert class_mismatch.close is False
    assert class_mismatch.keras_class == 0
    assert class_mismatch.py_class == 1
    assert class_mismatch.passed is False
    assert close_class_mismatch.close is True
    assert close_class_mismatch.keras_class == 1
    assert close_class_mismatch.py_class == 0
    assert close_class_mismatch.passed is False


def test_find_first_layer_diff_reports_first_non_close_layer() -> None:
    py_model = _FakeLayerOutputModel(
        {
            0: [1.0, 2.0],
            1: [30.0, 4.0],
        }
    )
    layer_pairs = [
        compare_mod.LayerPair(
            keras_layer_index=0,
            keras_layer_name="dense_a",
            keras_layer_type="Dense",
            my_layer_index=0,
            keras_layer=object(),
        ),
        compare_mod.LayerPair(
            keras_layer_index=1,
            keras_layer_name="dense_b",
            keras_layer_type="Dense",
            my_layer_index=1,
            keras_layer=object(),
        ),
    ]
    keras_layer_outputs = [
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([[3.0, 4.0]], dtype=np.float32),
    ]

    layer_diff = compare_mod.find_first_layer_diff(
        0,
        py_model,
        layer_pairs=layer_pairs,
        keras_layer_outputs=keras_layer_outputs,
    )

    assert layer_diff is not None
    assert layer_diff.keras_layer_index == 1
    assert layer_diff.keras_layer_name == "dense_b"
    assert layer_diff.my_layer_index == 1
    assert layer_diff.metrics.max_abs_diff == 27.0


def test_log_file_records_compare_messages(tmp_path) -> None:
    log_file = tmp_path / "predict-compare.log"
    compare_mod.configure_logging("INFO", log_file=str(log_file))

    result = compare_mod.compare_case(0, [0.1, 0.9], [0.1, 0.9])
    compare_mod.log_case(result)
    compare_mod.log_summary(compare_mod.summarize_results([result]))

    content = log_file.read_text()
    assert "[PASS] idx=0" in content
    assert "Result: PASS" in content


def test_run_batches_keras_predict_and_forwards_each_python_input(monkeypatch) -> None:
    inputs = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    keras_model = _FakeKerasModel([[0.1, 0.9], [0.8, 0.2]])
    py_model = _FakePythonModel([[0.100001, 0.900001], [0.800001, 0.199999]])

    _install_fake_runtime(monkeypatch, keras_model, py_model)
    monkeypatch.setattr(compare_mod, "load_dataset_inputs", lambda dataset, first_n: inputs)

    args = compare_mod.parse_args(["--model-name", "demo", "--dataset", "mnist", "--first-n", "2"])
    exit_code = compare_mod.run(args)

    assert exit_code == 0
    assert len(keras_model.predict_calls) == 1
    np.testing.assert_array_equal(keras_model.predict_calls[0][0], inputs)
    assert keras_model.predict_calls[0][1] == 0
    assert py_model.forward_calls == [[1.0, 2.0], [3.0, 4.0]]


def test_run_fail_fast_stops_after_first_failed_case(monkeypatch) -> None:
    inputs = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    keras_model = _FakeKerasModel([[0.1, 0.9], [0.8, 0.2]])
    py_model = _FakePythonModel([[0.9, 0.1], [0.8, 0.2]])

    _install_fake_runtime(monkeypatch, keras_model, py_model)
    monkeypatch.setattr(compare_mod, "load_dataset_inputs", lambda dataset, first_n: inputs)

    args = compare_mod.parse_args(
        ["--model-name", "demo", "--dataset", "mnist", "--first-n", "2", "--fail-fast"]
    )
    exit_code = compare_mod.run(args)

    assert exit_code == 1
    assert len(keras_model.predict_calls) == 1
    assert py_model.forward_calls == [[1.0, 2.0]]


def test_main_logs_summary_and_returns_failure(monkeypatch, caplog) -> None:
    inputs = np.asarray([[1.0, 2.0]], dtype=np.float32)
    keras_model = _FakeKerasModel([[0.1, 0.9]])
    py_model = _FakePythonModel([[0.9, 0.1]])

    _install_fake_runtime(monkeypatch, keras_model, py_model)
    monkeypatch.setattr(compare_mod, "load_dataset_inputs", lambda dataset, first_n: inputs)

    caplog.set_level("INFO", logger="ct.predict_compare")
    exit_code = compare_mod.main(["--model-path", "model/demo.h5", "--first-n", "1"])

    assert exit_code == 1
    assert "[FAIL] idx=0" in caplog.text
    assert "Compared inputs: 1" in caplog.text
    assert "Result: FAIL" in caplog.text


def _install_fake_runtime(monkeypatch, keras_model, py_model) -> None:
    fake_loader = ModuleType("modeling.keras_loader")
    fake_loader.load_model_with_compat = lambda _path: keras_model
    monkeypatch.setitem(sys.modules, "modeling.keras_loader", fake_loader)

    fake_runtime = ModuleType("engine.predictor_runtime")
    fake_runtime.myModel = None

    def init_model(_path):
        fake_runtime.myModel = py_model

    fake_runtime.init_model = init_model
    monkeypatch.setitem(sys.modules, "engine.predictor_runtime", fake_runtime)

    import engine

    monkeypatch.setattr(engine, "predictor_runtime", fake_runtime, raising=False)
