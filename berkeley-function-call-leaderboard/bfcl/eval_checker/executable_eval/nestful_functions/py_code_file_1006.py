from typing import List



def perform_numeric_operations() -> List[int]:

    """

    Performs a series of numeric operations and documents each local variable.

    Returns a list of results.

    """

    x: int = 2 + 1  # x = 3

    """The result of adding 2 and 1."""

    y: int = x * 4  # y = 12

    """The result of multiplying x by 4."""

    z: int = y - 5  # z = 7

    """The result of subtracting 5 from y."""

    return [x, y, z]

