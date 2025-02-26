from typing import Union



def is_power_of_two_or_two_plus_one(x: Union[int, float]) -> bool:

    """Determines if an integer is a power of two or a power of two plus one.

    Args:

        x: The integer to check.

    """

    return x == 1 or ((x & (x - 1) == 0) | (x & (x - 2) == 2))

