from typing import Tuple, Dict, Union

class Position(Tuple[int, Tuple[int, ...]]):
    pass

my_layer_number_to_Keras_layer_number: Dict[int, int] = dict()

def register_layer_number_mapping(keras_layer_number: int, my_layer_number: int):
    global my_layer_number_to_Keras_layer_number
    my_layer_number_to_Keras_layer_number[my_layer_number] = keras_layer_number

def to_Keras_layer_number(my_layer_number: int) -> int:
    global my_layer_number_to_Keras_layer_number
    return my_layer_number_to_Keras_layer_number[my_layer_number]

current_layer_number: int 
current_indices_in_current_layer: Union[Tuple[int, ...], list[Tuple[int, ...]]]

def register_current_indices(indices: Union[Tuple[int, ...], list[Tuple[int, ...]]]):
    # print('------', indices)
    global current_indices_in_current_layer
    current_indices_in_current_layer = indices

def register_current_layer_number(keras_layer_number: int):
    # print('-------------', keras_layer_number)
    global current_layer_number
    current_layer_number = keras_layer_number

def get_current_position() -> Position:
    global current_layer_number
    global current_indices_in_current_layer
    return current_layer_number, current_indices_in_current_layer