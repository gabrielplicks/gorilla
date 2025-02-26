from typing import Union



def sqrt_newton(x: Union[int, float]) -> float:

    """Calculates the square root of a number using Newton's method.



    Args:

        x: The number to calculate the square root of.



    Returns:

        The square root of `x`.

    """

    y = 1.0

    while True:

        new_y = 0.5 * (y + x / y)

        if abs(new_y - y) < 0.001:

            return new_y

        y = new_y

