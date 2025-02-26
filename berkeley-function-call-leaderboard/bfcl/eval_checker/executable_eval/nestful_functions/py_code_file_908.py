from typing import Dict



def data_to_tuple(x: Dict[str, str]) -> tuple:

    """Converts a dictionary to a tuple representation.



    Args:

        x: The dictionary to convert.



    Raises:

        AssertionError: If the input data is not a dictionary.

    """

    assert isinstance(x, dict), "Input data must be a map"

    tuple_repr = []

    for key, value in x.items():

        tuple_repr.append(key)

        tuple_repr.append(value)

    return tuple(tuple_repr)

