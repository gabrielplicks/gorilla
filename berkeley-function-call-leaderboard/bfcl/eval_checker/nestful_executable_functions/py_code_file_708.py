from math import sqrt

from typing import Union



def approximate_square_root(n: Union[int, float]) -> Union[int, float]:

    """Calculates an approximation of the square root of a number using Newton's method.



    Args:

        n: The number to calculate the square root of.

    """

    x = n / 2

    while abs(x**2 - n) > 0.001:

        x = (x + n / x) / 2

    return round(x)

