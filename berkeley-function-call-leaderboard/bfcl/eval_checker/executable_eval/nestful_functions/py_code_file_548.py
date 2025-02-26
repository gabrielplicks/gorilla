from typing import List



def to_integer_list(float_list: List[float]) -> List[int]:

    """Converts a list of numbers in float format to an equivalent list of integers.

    Args:

        float_list: A list of numbers in float format.

    Raises:

        Exception: If any element in the input list is not a valid float.

    """

    for element in float_list:

        if not isinstance(element, (int, float)) or element != element:

            raise Exception("Invalid input: {} is not a valid float".format(element))

    return [int(element) for element in float_list]

