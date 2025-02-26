from typing import Union



def cube_integer_sum(n: Union[int, float]) -> int:

    """Calculates the sum of the cubes of all integers from 1 to `n`.



    Args:

        n: The upper limit of the range of integers.

    """

    total = 0

    for i in range(1, n + 1):

        total += i ** 3

    return total

