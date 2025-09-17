from __future__ import annotations
import sys

import functools
import shap
from keras import Model
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import InputLayer
from tensorflow.keras.layers import Reshape, Input
from functools import reduce
from typing import Literal, Tuple
import numpy as np
import json
from keras.models import load_model

from libct.constraint import Constraint

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

    class PositionedConstraint(Tuple[Constraint, Tuple[int, Tuple[int, ...]]]):
        pass


def pop_last_constraint(
    positioned_constraints: list[PositionedConstraint],
) -> Constraint:
    return positioned_constraints.pop()[0]


def pop_first_constraint(
    positioned_constraints: list[PositionedConstraint],
) -> Constraint:
    return positioned_constraints.pop(0)[0]


def pop_the_most_important_constraint(
    positioned_constraints: list[PositionedConstraint],
    compare: Callable[[PositionedConstraint, PositionedConstraint], int],
) -> Constraint:
    positioned_constraints.sort(key=functools.cmp_to_key(compare))
    return pop_last_constraint(positioned_constraints)


class ShapValuesComparator:
    def __init__(
        self,
        model_path,
        background_dataset,
        input,
        idx,
        shap_value_pre_calculated,
        explainer_type: Literal["gradient", "kernel"] = "gradient",
    ) -> None:
        self.model_path = model_path
        self.model = load_model(model_path)
        self.background_dataset = background_dataset
        self.input = input
        self.idx = idx
        self.shap_value_pre_calculated = shap_value_pre_calculated
        self.explainer_type = explainer_type 

        # If the SHAP values have already been computed, read them directly.
        if self.shap_value_pre_calculated:
                with open(f'./shap_value_all_layer/{self.model_path.split("/")[-1].split(".")[0]}/shap_value_{self.idx}.json', 'r') as f:
                    self.shap_values = json.load(f)
        else:
            self.shap_values: dict[str, float] = dict()
            self.calculate_shap_values_for_all_neurons()

    def compare(
        self,
        positioned_constraint_1: PositionedConstraint,
        positioned_constraint_2: PositionedConstraint,
    ) -> float:
        constraint_1, (row_number_1, indices_1) = positioned_constraint_1
        constraint_2, (row_number_2, indices_2) = positioned_constraint_2
        # if row_number_1 > row_number_2: return -1
        # if row_number_1 < row_number_2: return 1
        return self.get_shap_influence(row_number_1, indices_1) - self.get_shap_influence(
            row_number_2, indices_2
        )

    def calculate_shap_values_for_all_neurons(self) -> None:
        trimmed_model = self.model
        transformed_background = self.background_dataset
        transformed_input = self.input
        number_of_layers = len(self.model.layers) if isinstance(self.model, Sequential) else len(self.model.layers) - 1
        for layer_number in range(0, number_of_layers):
            self.calculate_shap_values(
                trimmed_model, transformed_background, transformed_input, layer_number
            )
            transformed_input = ShapValuesComparator.apply_one_layer(
                trimmed_model, transformed_input
            )
            transformed_background = ShapValuesComparator.apply_one_layer_to_dataset(
                trimmed_model, transformed_background
            )
            if layer_number == number_of_layers - 1:
                break
            trimmed_model = ShapValuesComparator.without_first_layer(trimmed_model)
        file_path = f"./shap_value_all_layer/transformer_fashion_mnist_two_mha/shap_value_{self.idx}.json"
        with open(file_path, "w") as json_file:
            json.dump(self.shap_values, json_file)
        print(f"shap values saved to {file_path}")
        
        

    @staticmethod
    def without_first_layer(original_model: Sequential | Model) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            new_model = Sequential()
            for layer in original_model.layers[1:]:
                new_model.add(layer)
            new_model.build(original_model.layers[1].input_shape)
        elif isinstance(original_model, Model):
            # Get the output of the second layer
            print('$'*100, 'before')
            print(len(original_model.layers), ' layers')
            original_model.summary()
            
            new_input = original_model.layers[2].input
            # Get the output of the last layer
            new_output = original_model.layers[-1].output
            new_model = Model(inputs=new_input, outputs=new_output)
            print('$'*100, 'after')
            new_model.summary()

        return new_model

    @staticmethod
    def without_first_layer_fast(
        original_model: Sequential | Model,
    ) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            new_model = Sequential(original_model.layers[1:])
            new_model.build(original_model.layers[1].input_shape)
        elif isinstance(original_model, Model):
            # Get the output of the second layer
            new_input = original_model.layers[1].input
            # Get the output of the last layer
            new_output = original_model.layers[-1].output
            new_model = Model(inputs=new_input, outputs=new_output)
        return new_model

    @staticmethod
    def apply_one_layer_to_dataset(
        original_model: Sequential | Model, dataset: np.ndarray
    ) -> np.ndarray:
        model_with_only_first_layer = (
            ShapValuesComparator.get_model_with_only_first_layer(original_model)
        )
        model_with_only_first_layer.build(original_model.layers[0].input_shape)
        new_dataset = model_with_only_first_layer.predict(dataset)
        # new_dataset = np.array([model_with_only_first_layer.predict(data) for data in dataset ])
        return new_dataset

    @staticmethod
    def apply_one_layer(
        original_model: Sequential | Model, original_input: np.ndarray
    ) -> np.ndarray:
        return ShapValuesComparator.get_model_with_only_first_layer(
            original_model
        ).predict(original_input)

    @staticmethod
    def get_model_with_only_first_layer(
        original_model: Sequential | Model,
    ) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            model_with_only_first_layer = Sequential(original_model.layers[:1])
        elif isinstance(original_model, Model):
            model_with_only_first_layer = Model(
                inputs=original_model.layers[1].input, outputs=original_model.layers[1].output
            )
            print("model_with_only_first_layer:", model_with_only_first_layer.summary())
        return model_with_only_first_layer

    def calculate_shap_values(
        self, model, background_dataset, input, layer_number: int
    ) -> None:
        if self.explainer_type == "gradient":
            print("------background_dataset------", background_dataset.shape)
            print("------input------", input.shape)
            outputs=model.layers[0].output
            explainer = shap.GradientExplainer(model, background_dataset)
            shap_values = explainer.shap_values(input)
            
            shap_values_summed = np.sum(shap_values, axis=1)
            average_shap_values = np.squeeze(np.mean(shap_values_summed, axis=0))

            for indices, shap_value in np.ndenumerate(average_shap_values):
           
                self.shap_values[
                    ShapValuesComparator.get_position_key(layer_number - 1, indices)
                ] = shap_value
        else: # self.explainer_type == "kernel"
            
            shouldFlattenInputToUseKernelExplainer = len(input.shape) > 2
            if shouldFlattenInputToUseKernelExplainer:
                originalInputShape = input.shape
                model, input, background_dataset = self.flatten_everything(model, input, background_dataset)
                input = np.expand_dims(input, axis=0)
                model.summary()
            explainer = shap.KernelExplainer(model, background_dataset)
            
            
            shap_values = explainer.shap_values(input)
            
            for index, shap_value in np.ndenumerate(shap_values):
                index = index[1]
                if shouldFlattenInputToUseKernelExplainer:
                    indices = self.unflatten_index(index, originalInputShape)
                else:
                    indices = (index,)

                self.shap_values[
                    ShapValuesComparator.get_position_key(layer_number - 1, indices)
                ] = shap_value

    def get_shap_influence(self, layer_number: int, indices: tuple[int, ...] | list[tuple[int, ...]]) -> float:
       
        number_of_layers = len(self.model.layers) if isinstance(self.model, Sequential) else len(self.model.layers) - 1
        if layer_number == number_of_layers - 1:
            return float("-inf")
        if isinstance(indices, list):
            return sum([self.shap_values[ShapValuesComparator.get_position_key(layer_number, ids)] for ids in indices]) / len(indices)
        else:
            return self.shap_values[
                ShapValuesComparator.get_position_key(layer_number, indices)
            ]

    @staticmethod
    def get_position_key(layer_number: int, indices: tuple[int, ...]) -> str:
        key = str(layer_number)
        # if layer_number == 1:
        #     print("------key------", key)
        for i in indices:
            key += "_" + str(i)
        # if layer_number == 1:
        #     print("------key after + indices------", key)
        return key

    def print_shap_values(self) -> None:
        for key in self.shap_values:
            print(key, self.shap_values[key])
    def flatten_everything(self, model, input, background_dataset):
        print("input shape",input.shape)
        print("input shape[1:]",input.shape[1:])
        model = self.prepend_unflatten_layer(model, input.shape[1:])
        input = self.flatten(input)
        background_dataset = np.array([self.flatten(data_point) for data_point in background_dataset])
        return model, input, background_dataset
    
    def product_shape(self,shape):
        return reduce(lambda x, y: x*y, [d for d in shape])
    
    def prepend_unflatten_layer(self, model, original_input_shape):
        # add an layer as the new first layer to unflatten input
        new_input_shape = (self.product_shape(original_input_shape), )

        new_model = Sequential()
        new_model.add(Input(new_input_shape))
        new_model.add(Reshape(original_input_shape))
        for layer in model.layers:
            new_model.add(layer)

        new_model.build()

        return new_model

    def flatten(self,input):
        # flatten input
        newInput = np.reshape(input, [-1])
        return newInput
    def unflatten_index(self, flattened_index: int, original_input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        # recover indices from the flattened_index, given the original_input_shape
        indices = []
        current_modular = original_input_shape[-1]
        current_divisor = 1
        reversed_shape = tuple(reversed(original_input_shape))
        for (i, d) in enumerate(reversed_shape):
            indices.append(flattened_index % current_modular // current_divisor)
            current_divisor = current_modular
            if i+1 < len(reversed_shape):
                current_modular *= reversed_shape[i+1]
        return tuple(reversed(indices))


# usage:
# positioned_constraints: List[PositionedConstraint] = ...
# constraint: Constraint = pop_the_most_important_constraint(positioned_constraints, ShapValuesComparator().compare)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
input_shape = (28, 28, 1)
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# 將標籤進行 one-hot 編碼
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


class ShapValuesComparatorTester:
    def __init__(self):
        # construct a model with three Layers
        self.input_data = np.random.rand(1000, 10)
        self.model = Sequential()
        self.model.add(Dense(units=32, activation="relu", input_shape=(10,)))
        self.model.add(Dense(units=1, activation="sigmoid"))
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model.summary()
        self.labels = np.random.randint(2, size=(1000, 1))
        self.model.fit(self.input_data, self.labels, epochs=10, batch_size=32)
        self.background_dataset = self.input_data[
            np.random.choice(self.input_data.shape[0], 100, replace=False)
        ]
        # self.background_dataset = self.background_dataset.reshape((100, -1))
        # create an input of floats between 0 and 1
        self.input = self.input_data[:1]
        # create a comparator
        self.runtests()

    def runtests(self):
        # self.test_get_position_key()
        # self.test_without_first_layer()
        # self.test_apply_one_layer()
        # self.test_apply_one_layer_to_dataset()
        # self.test_get_model_with_only_first_layer()
        self.comparator = ShapValuesComparator(
            self.model, self.background_dataset, self.input
        )
        print(self.comparator.shap_values)
        # self.test_calculate_shap_values()
        # self.test_calculate_shap_values_for_all_neurons()
        # self.test_get_shap_influence()
        # self.test_compare()

    def test_get_position_key(self):
        assert ShapValuesComparator.get_position_key(1, (1, 2)) == "1_1_2"
        assert ShapValuesComparator.get_position_key(1, (1, 2, 3)) == "1_1_2_3"
        assert ShapValuesComparator.get_position_key(1, ()) == "1"

    def test_without_first_layer(self):
        # self.model.summary()
        new_model = ShapValuesComparator.without_first_layer(self.model)
        # new_model.summary()
        assert len(new_model.layers) == 2
        assert new_model.layers[0].input_shape == (None, 64)
        assert new_model.layers[1].input_shape == (None, 10)
        assert new_model.layers[0].output_shape == (None, 10)
        assert new_model.layers[0].get_config()["activation"] == "relu"
        assert new_model.layers[1].get_config()["activation"] == "softmax"

    def test_apply_one_layer(self):
        print("2-1")
        assert self.input.shape == (1, 64)
        print("2-2")
        assert self.model.layers[0].input_shape == (None, 64)
        print("2-3")
        assert self.model.layers[0].output_shape == (None, 32)
        print("2-4")
        output = ShapValuesComparator.apply_one_layer(self.model, self.input)
        print("2-5")
        assert output.shape == (1, 32)

    def test_apply_one_layer_to_dataset(self):
        assert self.background_dataset.shape == (100, 1, 64)
        assert self.model.layers[0].input_shape == (None, 64)
        assert self.model.layers[0].output_shape == (None, 32)
        print("3-1")
        output = ShapValuesComparator.apply_one_layer_to_dataset(
            self.model, self.background_dataset
        )
        assert output.shape == (100, 1, 32)

    def test_get_model_with_only_first_layer(self):
        model_with_only_first_layer = (
            ShapValuesComparator.get_model_with_only_first_layer(self.model)
        )
        assert len(model_with_only_first_layer.layers) == 1
        assert (
            model_with_only_first_layer.layers[0].input_shape
            == self.model.layers[0].input_shape
        )
        assert (
            model_with_only_first_layer.layers[0].output_shape
            == self.model.layers[0].output_shape
        )
        assert (
            model_with_only_first_layer.layers[0].activation
            == self.model.layers[0].activation
        )

    def test_calculate_shap_values(self):
        self.comparator.calculate_shap_values(
            self.model, self.background_dataset, self.input, 0
        )
        self.print_shap_values()
        assert len(self.comparator.shap_values) >= 64
        assert "0_0_0_1" in self.comparator.shap_values
        assert "0_0_0_63" in self.comparator.shap_values

    def print_shap_values(self):
        for k, v in self.comparator.shap_values.items():
            print(k, v)

    def test_get_shap_influence(self):
        self.comparator.calculate_shap_values(
            self.model, self.background_dataset, self.input, 1
        )
        assert (
            self.comparator.get_shap_influence(1, (0, 0, 0))
            == self.comparator.shap_values["1_0_0_0"]
        )
        assert (
            self.comparator.get_shap_influence(1, (0, 0, 31))
            == self.comparator.shap_values["1_0_0_31"]
        )

    def test_calculate_shap_values_for_all_neurons(self):
        assert len(self.comparator.shap_values) == 64 + 32 + 64 + 10
        assert self.comparator.shap_values.get("0_0_0_0") != None
        assert self.comparator.shap_values.get("0_0_0_63") != None
        assert self.comparator.shap_values.get("1_0_0_0") != None
        assert self.comparator.shap_values.get("1_0_0_31") != None
        assert self.comparator.shap_values.get("2_0_0_0") != None
        assert self.comparator.shap_values.get("2_0_0_63") != None
        assert self.comparator.shap_values.get("3_0_0_0") != None
        assert self.comparator.shap_values.get("3_0_0_9") != None

    def test_compare(self):
        mock_constraint_1 = Constraint(None, None)
        compare = self.comparator.compare
        positioned_constraint_1: PositionedConstraint = (
            mock_constraint_1,
            (1, (0, 0, 1)),
        )
        positioned_constraint_2: PositionedConstraint = (
            mock_constraint_1,
            (1, (0, 0, 2)),
        )
        if (
            self.comparator.shap_values["1_0_0_1"]
            > self.comparator.shap_values["1_0_0_2"]
        ):
            assert compare(positioned_constraint_1, positioned_constraint_2) > 0
        elif (
            self.comparator.shap_values["1_0_0_1"]
            < self.comparator.shap_values["1_0_0_2"]
        ):
            assert compare(positioned_constraint_1, positioned_constraint_2) < 0
        else:
            assert compare(positioned_constraint_1, positioned_constraint_2) == 0


if __name__ == "__main__":
    ShapValuesComparatorTester()
