from typing import Union



def sum_of_consecutive_numbers(n: Union[int, float]) -> float:

    """Calculates the sum of all elements in the list [1, 2, ..., n].



    Args:

        n: A non-negative integer or float.

    """

    return n * (n + 1) / 2

