from typing import List



def max_product(values: List[int]) -> int:

    """Calculates the maximum product of a pair of values in a list of integers.



    Args:

        values: A list of integer values.



    Returns:

        The maximum product of a pair of values in the list.

    """

    max_value = float('-inf')

    max_index = -1



    for i, value in enumerate(values):

        if value > max_value:

            max_value = value

            max_index = i



    max_product = float('-inf')



    for i, value in enumerate(values):

        if i != max_index:

            product = value * max_value

            if product > max_product:

                max_product = product



    return max_product

