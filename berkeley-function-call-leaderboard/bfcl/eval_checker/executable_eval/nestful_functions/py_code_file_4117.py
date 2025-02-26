from typing import Tuple



def one_to_zero(a: int, b: int) -> Tuple[int, int]:

    """Converts a one-based coordinate into a zero-based coordinate.

    Args:

        a: The x-coordinate of the one-based coordinate.

        b: The y-coordinate of the one-based coordinate.

    """

    x = a - 1

    y = b - 1

    return x, y

