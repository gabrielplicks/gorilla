from typing import List, Union



def get_types(sequence: List[Union[int, float, str]]) -> List[str]:

    """Returns a list of the types of elements in the input sequence.



    Args:

        sequence: A non-empty sequence of integers, floats, or strings.



    Returns:

        A list of strings representing the types of the elements in the input sequence.

    """

    types = []

    for element in sequence:

        element_type = type(element).__name__

        types.append(element_type)

    return types

