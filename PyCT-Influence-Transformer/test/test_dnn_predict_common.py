from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dnn_predict_common as predictor


MODEL_PATH = ROOT / 'model' / 'simple_mnist_m6_09585.h5'


def test_collect_input_names_uses_model_inputs_when_input_layer_is_absent() -> None:
    model = predictor.load_model_with_compat(str(MODEL_PATH))

    input_names = predictor._collect_input_names(model)

    assert 'input_1' in input_names


def test_init_model_bootstraps_real_mnist_model_without_missing_input_key() -> None:
    predictor.myModel = None
    predictor.loaded_model_path = None

    predictor.init_model(str(MODEL_PATH))

    assert predictor.myModel is not None
    assert predictor.myModel.keras_to_cache_key['input_1'] == 'layer_input'
