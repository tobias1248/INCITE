from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from shap_cal import sort_pixel_converter


def _build_sample_shap_values() -> np.ndarray:
    """Create a deterministic SHAP tensor fixture for unit tests."""
    return np.array(
        [
            [
                [[[1.0, -0.5, 0.0]], [[0.0, 0.0, 0.1]]],
                [[[-0.25, 0.25, 0.0]], [[0.0, 0.0, 2.0]]],
            ],
            [
                [[[0.1, 0.0, -0.1]], [[0.0, 0.5, 0.5]]],
                [[[-1.0, 0.0, 0.0]], [[0.0, 0.0, 0.05]]],
            ],
        ],
        dtype=np.float32,
    )


class SortPixelConverterTest(unittest.TestCase):
    """Validate SHAP-to-coordinate conversion helpers."""

    def setUp(self) -> None:
        """Prepare shared SHAP tensor fixture."""
        self.shap_values = _build_sample_shap_values()

    def test_sort_pixel_coordinates_orders_by_absolute_sum(self) -> None:
        """Coordinates should be sorted by descending |SHAP| score."""
        expected = np.array(
            [
                [[1, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[0, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 0]],
            ],
            dtype=np.int64,
        )
        coords = sort_pixel_converter.sort_pixel_coordinates(self.shap_values)
        np.testing.assert_array_equal(coords, expected)

    def test_sort_pixel_coordinates_honors_topk(self) -> None:
        """Limiting top-k should truncate each sample's coordinate list."""
        coords = sort_pixel_converter.sort_pixel_coordinates(self.shap_values, topk=2)
        self.assertEqual(coords.shape, (2, 2, 3))
        np.testing.assert_array_equal(
            coords[0], np.array([[1, 1, 0], [0, 0, 0]], dtype=np.int64)
        )

    def test_convert_shap_file_round_trip(self) -> None:
        """File conversion should persist the exact coordinate tensor."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_path = tmp_root / "shap_values.npy"
            output_path = tmp_root / "sorted_pixels.npy"
            np.save(input_path, self.shap_values)

            produced_path = sort_pixel_converter.convert_shap_file(
                input_path, output_path, topk=3
            )
            self.assertEqual(produced_path, output_path)

            expected = sort_pixel_converter.sort_pixel_coordinates(self.shap_values, topk=3)
            saved = np.load(output_path)
            np.testing.assert_array_equal(saved, expected)


if __name__ == "__main__":
    unittest.main()
