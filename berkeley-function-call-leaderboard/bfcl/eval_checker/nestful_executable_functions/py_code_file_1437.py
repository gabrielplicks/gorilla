from typing import List



def multiply_list(input_list: List[float], constant: float) -> List[float]:

    """Multiplies every element in a list by a constant value.



    Args:

        input_list: The list of numbers to be multiplied.

        constant: The constant value to multiply each element by.

    """

    for i in range(len(input_list)):

        input_list[i] = input_list[i] * constant

    return input_list

