from typing import TypeVar



T = TypeVar("T", int, bytes)



def bit_not_inverse(x: T) -> T:

    """

    Calculates the bitwise inverse of a positive integer `x`.



    Args:

        x: The positive integer to calculate the bitwise inverse of.



    Returns:

        The bitwise inverse of `x`.

    """

    return ~x & 0xFFFFFFFF

