from typing import Union



def my_sqrt(n: Union[int, float]) -> float:

    """Calculates the square root of a number using the Newton method.



    Args:

        n: The number to calculate the square root of.



    Returns:

        The square root of the number as a floating-point number.

    """

    root = n / 2

    while abs(root**2 - n) > 0.0000000001:

        root = (root + n / root) / 2

    return round(root, 10)

