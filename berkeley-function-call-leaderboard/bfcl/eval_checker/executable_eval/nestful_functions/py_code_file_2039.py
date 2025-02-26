from typing import Tuple



def division_floor(x: int, y: int) -> int:

    """Calculates the floor division of two numbers `x` and `y`.



    Args:

        x: The numerator.

        y: The denominator.



    Returns:

        The floor division of `x` and `y`.

    """

    return (x - x % y) // y

