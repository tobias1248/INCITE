from __future__ import annotations

import functools
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Tuple

import numpy as np
import shap
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Sequential

from explainability.shap_contract import (
    DEFAULT_TARGET_CLASS_SHAP_ROOT,
    ShapCacheContractError,
    build_cache_identity,
    build_cache_metadata,
    load_target_class_cache,
    select_target_class_values,
)
from libct.constraint import Constraint
from modeling.keras_loader import load_model_with_compat as shared_load_model_with_compat

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

log = logging.getLogger("ct.shap")


def _load_model_with_compat(
    model_path: str,
    *,
    input_shape_override: Optional[Tuple[int, ...]] = None,
    num_classes_override: Optional[int] = None,
) -> Sequential | Model:
    return shared_load_model_with_compat(
        model_path,
        input_shape_override=input_shape_override,
        num_classes_override=num_classes_override,
    )


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
        output_root: str = DEFAULT_TARGET_CLASS_SHAP_ROOT,
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
        inferred_input_shape: Optional[Tuple[int, ...]] = None
        input_arr = np.asarray(input_data)
        if input_arr.ndim >= 2:
            inferred_input_shape = tuple(int(v) for v in input_arr.shape[1:])

        self._model = _load_model_with_compat(
            model_path,
            input_shape_override=inferred_input_shape,
        )
        self._target_class, self._class_count = self._predict_target_class()
        self._shap_values: Dict[str, float] | None = None
        background_settings = self._read_background_settings()
        self._cache_meta = build_cache_metadata(
            case_index=self.idx,
            model_path=self.model_path,
            input_data=np.asarray(self.input),
            background_dataset=np.asarray(self.background_dataset),
            explainer_type=self.explainer_type,
            target_class=self._target_class,
            class_count=self._class_count,
            background_per_class=background_settings.get("background_per_class"),
            background_seed=background_settings.get("background_seed"),
        )
        self._tracked_layers = [
            layer
            for layer in self._model.layers
            if type(layer).__name__ not in ("InputLayer", "Dropout", "Embedding")
        ]
        self._layer_count = (
            len(self._model.layers)
            if isinstance(self._model, Sequential)
            else len(self._model.layers) - 1
        )
        self._layerwise_enabled = isinstance(self._model, Sequential)
        self.last_timing: Dict[str, Any] = self._build_timing_metadata(
            computed=False,
            was_cached=False,
            compute_seconds=0.0,
        )

    @property
    def model(self) -> Sequential | Model:
        return self._model

    @property
    def layer_count(self) -> int:
        return self._layer_count

    @property
    def target_class(self) -> int:
        return self._target_class

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
            self.last_timing = self._build_timing_metadata(
                computed=False,
                was_cached=False,
                compute_seconds=0.0,
            )
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
                self.last_timing = self._build_timing_metadata(
                    computed=False,
                    was_cached=True,
                    compute_seconds=0.0,
                )
                return self._shap_values
            except ShapCacheContractError as exc:
                log.warning("Ignoring incompatible SHAP cache %s: %s", self.cache_path, exc)

        start = time.perf_counter()
        shap_values = self._compute_shap_values()
        elapsed = time.perf_counter() - start
        self._save_cache(shap_values)
        self._shap_values = shap_values
        self.last_timing = self._build_timing_metadata(
            computed=True,
            was_cached=False,
            compute_seconds=elapsed,
        )
        return shap_values

    def _build_timing_metadata(
        self,
        *,
        computed: bool,
        was_cached: bool,
        compute_seconds: float,
    ) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "computed": bool(computed),
            "was_cached": bool(was_cached),
            "compute_seconds": float(compute_seconds),
            "output_path": str(self.cache_path),
            "schema_version": self._cache_meta["schema_version"],
            "attribution_target": self._cache_meta["attribution_target"],
            "target_class": self._target_class,
        }

    def _load_cache(self) -> Dict[str, float]:
        _, values = load_target_class_cache(
            self.cache_path,
            expected_identity=self._cache_meta,
            case_index=self.idx,
        )
        return values

    def _save_cache(self, shap_values: Dict[str, float]) -> None:
        payload = {"__meta__": self._cache_meta, "values": shap_values}
        temp_path = self.cache_path.with_name(
            f".{self.cache_path.name}.{os.getpid()}.tmp"
        )
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        temp_path.replace(self.cache_path)

    def _read_background_settings(self) -> Dict[str, int]:
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

    def _predict_target_class(self) -> Tuple[int, int]:
        predictions = np.asarray(self._model.predict(self.input, verbose=0))
        if not np.isfinite(predictions.astype(np.float64, copy=False)).all():
            raise ValueError("Keras model output contains NaN or Inf")
        if predictions.ndim != 2 or predictions.shape[0] != 1 or predictions.shape[1] < 2:
            raise ValueError(
                "Target-class SHAP requires one multiclass prediction; "
                f"got output shape {predictions.shape}"
            )
        return int(np.argmax(predictions[0])), int(predictions.shape[1])

    def _compute_shap_values(self) -> Dict[str, float]:
        shap_values: Dict[str, float] = {}
        if not self._layerwise_enabled:
            # Functional/DAG models (e.g., ResNet) are not safe for "drop first layer"
            # slicing. Compute reliable input-level SHAP only.
            self._calculate_layer_shap_values(
                shap_values,
                self._model,
                self.background_dataset,
                self.input,
                0,
            )
            # Add per-layer branch influence values for Functional models.
            self._calculate_functional_layer_branch_influence(shap_values)
            return shap_values

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

    def _calculate_functional_layer_branch_influence(
        self,
        shap_values: Dict[str, float],
    ) -> None:
        # For Functional/DAG models, estimate per-neuron branch influence as
        # gradient * (activation - background_mean_activation).
        if not self._tracked_layers:
            return

        output_specs: list[tuple[int, object]] = []
        for layer_index, layer in enumerate(self._tracked_layers):
            layer_output = getattr(layer, "output", None)
            if layer_output is None or isinstance(layer_output, (list, tuple)):
                continue
            output_specs.append((layer_index, layer_output))

        if not output_specs:
            return

        feature_outputs = [tensor for _, tensor in output_specs]
        feature_outputs.append(self._model.output)
        feature_model = Model(inputs=self._model.inputs, outputs=feature_outputs)

        input_array = np.asarray(self.input, dtype=np.float32)
        background_array = np.asarray(self.background_dataset, dtype=np.float32)
        if input_array.ndim == background_array.ndim - 1:
            input_array = np.expand_dims(input_array, axis=0)

        x = tf.convert_to_tensor(input_array, dtype=tf.float32)
        background = tf.convert_to_tensor(background_array, dtype=tf.float32)

        background_out = feature_model(background, training=False)
        if not isinstance(background_out, (list, tuple)):
            background_out = [background_out]
        background_acts = list(background_out[:-1])

        with tf.GradientTape() as tape:
            input_out = feature_model(x, training=False)
            if not isinstance(input_out, (list, tuple)):
                input_out = [input_out]
            input_acts = list(input_out[:-1])
            logits = input_out[-1]

            for act in input_acts:
                tape.watch(act)
            target = tf.reduce_mean(
                tf.gather(logits, self._target_class, axis=1)
            )

        grads = tape.gradient(target, list(input_acts))

        for (layer_index, _), act_input, act_background, grad in zip(
            output_specs,
            input_acts,
            background_acts,
            grads,
        ):
            if grad is None:
                continue

            baseline = tf.reduce_mean(act_background, axis=0)
            influence = grad[0] * (act_input[0] - baseline)
            influence_np = np.asarray(influence.numpy())

            for indices, value in np.ndenumerate(influence_np):
                shap_values[self.get_position_key(layer_index, indices)] = float(value)

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
            target_values = select_target_class_values(
                gradients,
                target_class=self._target_class,
                batched_input_shape=tuple(int(dim) for dim in input_data.shape),
            )

            for indices, value in np.ndenumerate(target_values):
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
                original_shape = tuple(int(dim) for dim in input_data.shape[1:])
                (
                    kernel_model,
                    kernel_input,
                    kernel_background,
                ) = self._flatten_everything(model, input_data, background_dataset)
                kernel_input = np.expand_dims(kernel_input, axis=0)

            explainer = shap.KernelExplainer(kernel_model, kernel_background)
            kernel_shap_values = explainer.shap_values(kernel_input)
            target_values = select_target_class_values(
                kernel_shap_values,
                target_class=self._target_class,
                batched_input_shape=tuple(int(dim) for dim in kernel_input.shape),
            )

            for idx, value in np.ndenumerate(target_values):
                flat_index = idx[0]
                if should_flatten and original_shape is not None:
                    indices = self._unflatten_index(flat_index, original_shape)
                else:
                    indices = idx
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


def _infer_layer_count_from_cached_values(shap_values: Dict[str, float]) -> int:
    max_layer = -1
    for key in shap_values.keys():
        token = key.split("_", 1)[0]
        if token.lstrip("-").isdigit():
            max_layer = max(max_layer, int(token))
    return max_layer + 1 if max_layer >= 0 else 1


def _load_cached_shap_values(cache_path: Path) -> Dict[str, float]:
    _, values = load_target_class_cache(cache_path)
    return values


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
        output_root: str = DEFAULT_TARGET_CLASS_SHAP_ROOT,
    ) -> None:
        lookup_mode = str(os.environ.get("PYCT_SHAP_LOOKUP_MODE", "all")).strip().lower()
        self._input_only_lookup = lookup_mode in {"input-only", "input_only", "inputonly"}
        cache_path = Path(output_root) / Path(model_path).stem / f"shap_value_{idx}.json"
        if shap_value_pre_calculated and cache_path.is_file():
            try:
                expected_identity = build_cache_identity(
                    case_index=idx,
                    model_path=model_path,
                    input_data=np.asarray(input),
                    background_dataset=np.asarray(background_dataset),
                    explainer_type=explainer_type,
                )
                _, self.shap_values = load_target_class_cache(
                    cache_path,
                    expected_identity=expected_identity,
                    case_index=idx,
                )
                self.layer_count = _infer_layer_count_from_cached_values(self.shap_values)
                self.model = None
                self.calculator = None
                log.debug("Loaded SHAP cache without model load: %s", cache_path)
                return
            except Exception as exc:
                log.warning(
                    "Failed to load SHAP cache '%s' in fast path (%r); falling back to model-backed path.",
                    cache_path,
                    exc,
                )

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
            return 0.0
        if isinstance(indices, list):
            if not indices:
                return 0.0
            signed_mean = sum(
                self._lookup(layer_number, ids) for ids in indices
            ) / len(indices)
            return abs(signed_mean)
        return abs(self._lookup(layer_number, indices))

    def _lookup(self, layer_number: int, indices: tuple[int, ...]) -> float:
        key = self.get_position_key(layer_number, indices)
        if key in self.shap_values and (
            not self._input_only_lookup or layer_number == -1
        ):
            return self.shap_values[key]
        alt_key = self.get_position_key(layer_number - 1, indices)
        if alt_key in self.shap_values and (
            not self._input_only_lookup or layer_number - 1 == -1
        ):
            return self.shap_values[alt_key]
        if len(indices) > 1:
            spatial = indices[:-1]
            spatial_key = self.get_position_key(layer_number, spatial)
            if spatial_key in self.shap_values and (
                not self._input_only_lookup or layer_number == -1
            ):
                return self.shap_values[spatial_key]
            spatial_alt_key = self.get_position_key(layer_number - 1, spatial)
            if spatial_alt_key in self.shap_values and (
                not self._input_only_lookup or layer_number - 1 == -1
            ):
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
