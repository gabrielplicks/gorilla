from typing import Union



def sqrt_rounded_up(x: Union[int, float]) -> int:

    """Calculates the square root of a number rounded up to the nearest integer.



    Args:

        x: The input number.



    Returns:

        The square root of the input number rounded up to the nearest integer.

    """

    i = 1

    curr_sq = 1

    while curr_sq < x:

        i += 1

        curr_sq = i * i

    return i

