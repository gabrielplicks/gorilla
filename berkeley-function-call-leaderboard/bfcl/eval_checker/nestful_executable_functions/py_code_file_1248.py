from typing import Tuple



def add_and_shift(x: int, y: int) -> int:

    """Adds two integers and shifts the result right by one bit.



    Args:

        x: The first integer.

        y: The second integer.



    Returns:

        The result of adding `x` and `y` together and then shifting the result right by one bit.

    """

    return (x + y) >> 1

