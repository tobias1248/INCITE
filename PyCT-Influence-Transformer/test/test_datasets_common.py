from __future__ import annotations

from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.common import (
    enable_attack_pixels,
    select_background_per_class,
    tensor_to_in_dict_and_con_dict,
)


def test_tensor_to_in_dict_and_con_dict_flattens_tensor() -> None:
    sample = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    in_dict, con_dict = tensor_to_in_dict_and_con_dict(sample)

    assert in_dict == {
        "v_0_0": 1.0,
        "v_0_1": 2.0,
        "v_1_0": 3.0,
        "v_1_1": 4.0,
    }
    assert con_dict == {
        "v_0_0": 0,
        "v_0_1": 0,
        "v_1_0": 0,
        "v_1_1": 0,
    }


def test_enable_attack_pixels_updates_existing_coordinates_only() -> None:
    con_dict = {"v_0_0": 0, "v_0_1": 0}

    enable_attack_pixels(con_dict, [(0, 1), (9, 9)])

    assert con_dict == {"v_0_0": 0, "v_0_1": 1}


def test_select_background_per_class_balances_classes() -> None:
    x_test = np.arange(24, dtype=np.float32).reshape(6, 4)
    y_test = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    background = select_background_per_class(x_test, y_test, per_class=1, seed=7)

    assert background.shape == (3, 4)
    rows = {tuple(row.tolist()) for row in background}
    expected_rows = {tuple(x_test[i].tolist()) for i in range(6)}
    assert rows.issubset(expected_rows)
