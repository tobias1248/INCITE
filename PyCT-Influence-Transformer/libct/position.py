from __future__ import annotations

from typing import Any, Dict, Tuple, Union

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


def summarize_indices(
    indices: Union[Tuple[int, ...], list[Tuple[int, ...]], Any],
    *,
    preview: int = 3,
) -> str:
    """Return a compact, log-friendly representation of indices."""
    if isinstance(indices, list):
        length = len(indices)
        if length == 0:
            return "[] (len=0)"
        if length <= preview * 2:
            return f"{indices} (len={length})"
        head = ", ".join(str(item) for item in indices[:preview])
        tail = ", ".join(str(item) for item in indices[-preview:])
        return f"[{head}, ..., {tail}] (len={length})"
    return str(indices)


def summarize_position(position: Any, *, preview: int = 3) -> str:
    """Return a compact, log-friendly representation of a position tuple."""
    if position is None:
        return "None"
    if (
        isinstance(position, tuple)
        and len(position) == 2
    ):
        layer, indices = position
        return f"(layer={layer}, indices={summarize_indices(indices, preview=preview)})"
    return str(position)
