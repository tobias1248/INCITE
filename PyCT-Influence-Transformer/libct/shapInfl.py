from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Literal, Tuple

import numpy as np
import shap
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Sequential, load_model

from libct.constraint import Constraint

if TYPE_CHECKING:
    from typing import Iterable

    class PositionedConstraint(Tuple[Constraint, Tuple[int, Tuple[int, ...]]]):
        ...


__all__ = [
    "ShapValuesCalculator",
    "ShapValuesComparator",
    "pop_last_constraint",
    "pop_first_constraint",
    "pop_the_most_important_constraint",
]


class ShapValuesCalculator:
    """Load, compute, and cache SHAP values for a given model/input pair."""

    def __init__(
        self,
        *,
        model_path: str,
        background_dataset: np.ndarray,
        input_data: np.ndarray,
        idx: int,
        explainer_type: Literal["gradient", "kernel"] = "gradient",
        output_root: str = "shap_value_all_layer",
    ) -> None:
        self.model_path = model_path
        self.model_name = Path(model_path).stem
        self.background_dataset = background_dataset
        self.input = input_data
        self.idx = idx
        self.explainer_type = explainer_type
        self.output_dir = Path(output_root) / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Avoid recompiling saved models (optimizer state may be incompatible across TF/Keras versions).
        self._model = load_model(model_path, compile=False)
        self._shap_values: Dict[str, float] | None = None
        self._cache_meta = self._read_background_meta()
        self._layer_count = (
            len(self._model.layers)
            if isinstance(self._model, Sequential)
            else len(self._model.layers) - 1
        )

    @property
    def model(self) -> Sequential | Model:
        return self._model

    @property
    def layer_count(self) -> int:
        return self._layer_count

    @property
    def cache_path(self) -> Path:
        return self.output_dir / f"shap_value_{self.idx}.json"

    def ensure(
        self,
        *,
        assume_cached: bool = False,
        force_refresh: bool = False,
    ) -> Dict[str, float]:
        """Return SHAP values, computing and persisting them when needed."""
        if self._shap_values is not None and not force_refresh:
            return self._shap_values

        cache_exists = self.cache_path.is_file()
        should_load_cache = (
            cache_exists
            and not force_refresh
            and (assume_cached or self._shap_values is None)
        )

        if should_load_cache:
            try:
                self._shap_values = self._load_cache()
                return self._shap_values
            except (json.JSONDecodeError, OSError):
                # fall back to recomputing if cache is corrupt
                pass

        shap_values = self._compute_shap_values()
        self._save_cache(shap_values)
        self._shap_values = shap_values
        return shap_values

    def _load_cache(self) -> Dict[str, float]:
        with self.cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "values" in data:
            self._cache_meta = data.get("__meta__") or self._cache_meta
            values = data.get("values", {})
        else:
            values = data
        if not isinstance(values, dict):
            raise TypeError(f"Expected SHAP cache {self.cache_path} to be a JSON dict.")
        return {str(k): float(v) for k, v in values.items()}

    def _save_cache(self, shap_values: Dict[str, float]) -> None:
        with self.cache_path.open("w", encoding="utf-8") as handle:
            if self._cache_meta:
                payload = {"__meta__": self._cache_meta, "values": shap_values}
                json.dump(payload, handle)
            else:
                json.dump(shap_values, handle)

    def _read_background_meta(self) -> Dict[str, int]:
        meta: Dict[str, int] = {}
        try:
            meta["background_per_class"] = int(os.environ.get("PYCT_BG_PER_CLASS", ""))
        except ValueError:
            pass
        try:
            meta["background_seed"] = int(os.environ.get("PYCT_BG_SEED", ""))
        except ValueError:
            pass
        return meta

    def _compute_shap_values(self) -> Dict[str, float]:
        shap_values: Dict[str, float] = {}
        trimmed_model = self._model
        transformed_background = self.background_dataset
        transformed_input = self.input

        for layer_number in range(self._layer_count):
            self._calculate_layer_shap_values(
                shap_values,
                trimmed_model,
                transformed_background,
                transformed_input,
                layer_number,
            )
            transformed_input = self.apply_one_layer(trimmed_model, transformed_input)
            transformed_background = self.apply_one_layer_to_dataset(
                trimmed_model, transformed_background
            )
            if layer_number == self._layer_count - 1:
                break
            trimmed_model = self.without_first_layer(trimmed_model)

        return shap_values

    def _calculate_layer_shap_values(
        self,
        shap_values: Dict[str, float],
        model: Sequential | Model,
        background_dataset: np.ndarray,
        input_data: np.ndarray,
        layer_number: int,
    ) -> None:
        if self.explainer_type == "gradient":
            bg_for_shap: Iterable[np.ndarray]
            if isinstance(background_dataset, list):
                bg_for_shap = background_dataset
            else:
                bg_for_shap = [background_dataset]

            explainer = shap.GradientExplainer(model, bg_for_shap)
            gradients = explainer.shap_values(input_data)
            gradients_summed = np.sum(gradients, axis=1)
            average_gradients = np.squeeze(np.mean(gradients_summed, axis=0))

            for indices, value in np.ndenumerate(average_gradients):
                shap_values[
                    self.get_position_key(layer_number - 1, indices)
                ] = float(value)
        else:
            should_flatten = len(input_data.shape) > 2
            original_shape: Tuple[int, ...] | None = None
            kernel_model = model
            kernel_input = input_data
            kernel_background = background_dataset

            if should_flatten:
                original_shape = input_data.shape
                (
                    kernel_model,
                    kernel_input,
                    kernel_background,
                ) = self._flatten_everything(model, input_data, background_dataset)
                kernel_input = np.expand_dims(kernel_input, axis=0)

            explainer = shap.KernelExplainer(kernel_model, kernel_background)
            kernel_shap_values = explainer.shap_values(kernel_input)

            for idx, value in np.ndenumerate(kernel_shap_values):
                flat_index = idx[1]
                if should_flatten and original_shape is not None:
                    indices = self._unflatten_index(flat_index, original_shape)
                else:
                    indices = (flat_index,)
                shap_values[
                    self.get_position_key(layer_number - 1, indices)
                ] = float(value)

    @staticmethod
    def without_first_layer(original_model: Sequential | Model) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            new_model = Sequential()
            for layer in original_model.layers[1:]:
                new_model.add(layer)
            new_model.build(original_model.layers[1].input_shape)
            return new_model

        new_input = original_model.layers[2].input
        new_output = original_model.layers[-1].output
        return Model(inputs=new_input, outputs=new_output)

    @staticmethod
    def get_model_with_only_first_layer(
        original_model: Sequential | Model,
    ) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            return Sequential(original_model.layers[:1])

        return Model(
            inputs=original_model.layers[1].input,
            outputs=original_model.layers[1].output,
        )

    @classmethod
    def apply_one_layer_to_dataset(
        cls, original_model: Sequential | Model, dataset: np.ndarray
    ) -> np.ndarray:
        model_with_only_first_layer = cls.get_model_with_only_first_layer(original_model)
        model_with_only_first_layer.build(original_model.layers[0].input_shape)
        return model_with_only_first_layer.predict(dataset)

    @classmethod
    def apply_one_layer(
        cls, original_model: Sequential | Model, original_input: np.ndarray
    ) -> np.ndarray:
        return cls.get_model_with_only_first_layer(original_model).predict(
            original_input
        )

    @staticmethod
    def get_position_key(layer_number: int, indices: Tuple[int, ...]) -> str:
        key = str(layer_number)
        for index in indices:
            key += f"_{index}"
        return key

    def _flatten_everything(
        self,
        model: Sequential | Model,
        input_data: np.ndarray,
        background_dataset: np.ndarray,
    ) -> Tuple[Sequential | Model, np.ndarray, np.ndarray]:
        flattened_input = self._flatten(input_data)
        flattened_background = np.array(
            [self._flatten(data_point) for data_point in background_dataset]
        )
        reshaped_model = self._prepend_unflatten_layer(model, input_data.shape[1:])
        return reshaped_model, flattened_input, flattened_background

    @staticmethod
    def _flatten(array: np.ndarray) -> np.ndarray:
        return np.reshape(array, [-1])

    @staticmethod
    def _prepend_unflatten_layer(
        model: Sequential | Model, original_input_shape: Tuple[int, ...]
    ) -> Sequential:
        new_input_shape = (int(np.prod(original_input_shape)),)
        new_model = Sequential()
        new_model.add(Input(new_input_shape))
        new_model.add(Reshape(original_input_shape))
        for layer in model.layers:
            new_model.add(layer)
        new_model.build()
        return new_model

    @staticmethod
    def _unflatten_index(
        flattened_index: int, original_input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        indices = []
        current_modular = original_input_shape[-1]
        current_divisor = 1
        reversed_shape = tuple(reversed(original_input_shape))
        for i, dim in enumerate(reversed_shape):
            indices.append(flattened_index % current_modular // current_divisor)
            current_divisor = current_modular
            if i + 1 < len(reversed_shape):
                current_modular *= reversed_shape[i + 1]
        return tuple(reversed(indices))


class ShapValuesComparator:
    """Comparator for prioritising constraints based on SHAP influence."""

    def __init__(
        self,
        *,
        model_path,
        background_dataset,
        input,
        idx,
        shap_value_pre_calculated,
        explainer_type: Literal["gradient", "kernel"] = "gradient",
        output_root: str = "shap_value_all_layer",
    ) -> None:
        self.calculator = ShapValuesCalculator(
            model_path=model_path,
            background_dataset=background_dataset,
            input_data=input,
            idx=idx,
            explainer_type=explainer_type,
            output_root=output_root,
        )
        self.shap_values = self.calculator.ensure(
            assume_cached=shap_value_pre_calculated,
            force_refresh=not shap_value_pre_calculated,
        )
        self.model = self.calculator.model
        self.layer_count = self.calculator.layer_count

    def compare(
        self,
        positioned_constraint_1: PositionedConstraint,
        positioned_constraint_2: PositionedConstraint,
    ) -> float:
        constraint_1, (row_number_1, indices_1) = positioned_constraint_1
        constraint_2, (row_number_2, indices_2) = positioned_constraint_2
        return self.get_shap_influence(row_number_1, indices_1) - self.get_shap_influence(
            row_number_2, indices_2
        )

    def get_shap_influence(
        self,
        layer_number: int,
        indices: tuple[int, ...] | list[tuple[int, ...]],
    ) -> float:
        if layer_number == self.layer_count - 1:
            return float("-inf")
        if isinstance(indices, list):
            total = 0.0
            for ids in indices:
                total += self._lookup(layer_number, ids)
            return total / len(indices)
        return self._lookup(layer_number, indices)

    def _lookup(self, layer_number: int, indices: tuple[int, ...]) -> float:
        key = self.get_position_key(layer_number, indices)
        if key in self.shap_values:
            return self.shap_values[key]
        alt_key = self.get_position_key(layer_number - 1, indices)
        if alt_key in self.shap_values:
            return self.shap_values[alt_key]
        if len(indices) > 1:
            spatial = indices[:-1]
            spatial_key = self.get_position_key(layer_number, spatial)
            if spatial_key in self.shap_values:
                return self.shap_values[spatial_key]
            spatial_alt_key = self.get_position_key(layer_number - 1, spatial)
            if spatial_alt_key in self.shap_values:
                return self.shap_values[spatial_alt_key]
        return 0.0

    @staticmethod
    def get_position_key(layer_number: int, indices: Tuple[int, ...]) -> str:
        return ShapValuesCalculator.get_position_key(layer_number, indices)

    @staticmethod
    def pop_last_constraint(
        positioned_constraints: list[PositionedConstraint],
    ) -> Constraint:
        return positioned_constraints.pop()[0]

    @staticmethod
    def pop_first_constraint(
        positioned_constraints: list[PositionedConstraint],
    ) -> Constraint:
        return positioned_constraints.pop(0)[0]

    @staticmethod
    def pop_the_most_important_constraint(
        positioned_constraints: list[PositionedConstraint],
        compare: Callable[[PositionedConstraint, PositionedConstraint], float],
    ) -> Constraint:
        positioned_constraints.sort(key=functools.cmp_to_key(compare))
        return ShapValuesComparator.pop_last_constraint(positioned_constraints)


def pop_last_constraint(
    positioned_constraints: list[PositionedConstraint],
) -> Constraint:
    return ShapValuesComparator.pop_last_constraint(positioned_constraints)


def pop_first_constraint(
    positioned_constraints: list[PositionedConstraint],
) -> Constraint:
    return ShapValuesComparator.pop_first_constraint(positioned_constraints)


def pop_the_most_important_constraint(
    positioned_constraints: list[PositionedConstraint],
    compare: Callable[[PositionedConstraint, PositionedConstraint], float],
) -> Constraint:
    return ShapValuesComparator.pop_the_most_important_constraint(
        positioned_constraints, compare
    )
