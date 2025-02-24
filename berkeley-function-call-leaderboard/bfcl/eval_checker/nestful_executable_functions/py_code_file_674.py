from typing import Union



def largest_10_divisor(n: int) -> Union[int, None]:

    """Calculates the largest integer that is less than or equal to `n` and divisible by 10.



    Args:

        n: The input integer.



    Returns:

        The largest integer that is less than or equal to `n` and divisible by 10.

    """

    while n % 10 != 0:

        n -= 1

    return n

