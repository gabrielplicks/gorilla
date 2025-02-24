from typing import Union



def square_root_approx(number: float, decimal_places: Union[int, float] = 2) -> float:

    """Approximates the square root of a given positive number using Heron's method.

    Args:

        number: The number whose square root you want to approximate.

        decimal_places: The number of decimal places in the result.

    """

    approximation = number / 2

    while True:

        next_approximation = (approximation + number / approximation) / 2

        if abs(approximation - next_approximation) < 0.0001:

            break

        approximation = next_approximation

    return round(approximation, decimal_places)

