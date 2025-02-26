from typing import Tuple



def floor_division(a: int, b: int) -> int:

    """Calculates the floor division of two non-negative integers.



    Args:

        a: The numerator.

        b: The denominator.



    Returns:

        The floor division of `a` and `b`.

    """

    return a - (a % b)

