from typing import List



def sum_between_0_and_1(float_list: List[float]) -> float:

    """Sum up all numbers in a given list of floats that are between 0.0 and 1.0.



    Args:

        float_list: A list of floats.



    Returns:

        The sum of all floats in the list that are between 0.0 and 1.0.

    """

    float_sum = 0.0

    for float_value in float_list:

        if 0.0 <= float_value <= 1.0:

            float_sum += float_value

    return float_sum

