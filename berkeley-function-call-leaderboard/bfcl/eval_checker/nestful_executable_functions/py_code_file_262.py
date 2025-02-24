from typing import Union



def babylonian_sqrt(x: Union[int, float]) -> int:

    """Calculates the square root of a number using the Babylonian method.



    Args:

        x: The number to calculate the square root of.



    Returns:

        The square root of `x` as an `int` value.

    """

    a0 = x / 2

    while True:

        a1 = (a0 + x / a0) / 2

        if abs(a0 - a1) < 1e-6:

            break

        a0 = a1

    return int(a1)

