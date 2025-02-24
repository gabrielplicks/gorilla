from math import ceil

from typing import TypeVar



T = TypeVar("T", int, float)



def align_integer(x: T, y: T) -> T:

    """Aligns `y` to the left so that it is a multiple of `x`.



    Args:

        x: The integer to align `y` to.

        y: The integer to be aligned.



    Returns:

        The smallest integer that is greater than or equal to `y` and divisible by `x`.

    """

    return x * ceil(y / x)

