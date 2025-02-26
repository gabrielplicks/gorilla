from typing import Union



def newton_raphson_sqrt(number: Union[int, float], iterations: int) -> Union[int, float]:

    """Calculates the square root of a positive number using the Newton-Raphson method.



    Args:

        number: The number to find the square root of.

        iterations: The number of iterations to perform.



    Returns:

        The approximate square root of number.

    """

    x = number / 2  # Initial guess

    for _ in range(iterations):

        x = 0.5 * (x + number / x)

    return x

