from typing import Union



def sum_of_first_perfect_squares(n: int) -> int:

    """Calculates the sum of the first `n` perfect squares.



    Args:

        n: The number of perfect squares to sum.



    Returns:

        The sum of the first `n` perfect squares.

    """

    result = 0

    for i in range(1, n + 1):

        result += i * i

    return result

