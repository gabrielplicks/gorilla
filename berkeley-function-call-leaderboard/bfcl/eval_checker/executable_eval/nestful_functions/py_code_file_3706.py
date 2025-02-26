from typing import Union



def sum_using_formula(n: Union[int, float]) -> float:

    """Calculates the sum of positive integers from 1 to `n` using the sum formula.



    Args:

        n: The upper limit of the summation.

    """

    return n * (n + 1) / 2



def sum_using_loop(n: Union[int, float]) -> float:

    """Calculates the sum of positive integers from 1 to `n` using a for-loop.



    Args:

        n: The upper limit of the summation.

    """

    sum_value = 0

    for i in range(1, int(n) + 1):

        sum_value += i

    return sum_value

