from typing import Tuple



def sum_of_squares_divisible_by_3(m: int, n: int) -> int:

    """Calculates the sum of the squares of all numbers between `m` and `n` (inclusive) that are divisible by 3.



    Args:

        m: The first integer.

        n: The second integer.



    Returns:

        The sum of the squares of all numbers between `m` and `n` (inclusive) that are divisible by 3.

    """

    result = 0

    for i in range(min(m, n), max(m, n) + 1):

        if i % 3 == 0:

            result += i ** 2

    return result

