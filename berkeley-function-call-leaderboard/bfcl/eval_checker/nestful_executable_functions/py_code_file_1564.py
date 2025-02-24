from typing import Union



def sum_using_for_loop(n: Union[int, float]) -> int:

    """Calculates the sum of all integers from 1 to `n` using a for-loop.

    Args:

        n: A positive integer.

    """

    total_sum = 0

    for i in range(1, int(n) + 1):

        total_sum += i

    return total_sum

